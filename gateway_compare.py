#!/usr/bin/env python3
"""
Compares two ways of adding the Granite judge to the AI gateway, on the same labeled cases.

  Case 1  gateway scanners + Granite as judge
          Every gateway scanner except language runs as today. Granite then reviews the
          verdict, with each scanner's finding shown to it, and can change it either way:
          un-block a false alarm or block an attack the scanners missed.
  Case 2  Granite instead of the model scanners
          Only PII, Secret, Malware (YARA) and Malicious Entity run from the gateway. Granite
          does the work of the prompt injection and toxicity models.
  "gateway only" (Case 1 without the judge) is shown too, as the starting point.

Every scanner and every Granite question runs on every case once; the setups are then
scored from those results, so they differ only in how the results are combined.

Run on the GPU instance:
   . .venv/bin/activate && . ./cuda_env.sh
   python gateway_compare.py
   python gateway_compare.py my_cases.jsonl    (same format as benchmark.py)

The gateway scanners are the gateway's own code, from the copy in gateway_snapshot/ (which
gateway commit it is: gateway_snapshot/SOURCE_COMMIT). To measure a newer gateway, run
./update_gateway_snapshot.sh /path/to/ai-gateway-source where the gateway repo is, commit,
and pull on the instance. --gateway-src points at a different copy instead.

Extra packages, once (CPU torch first, otherwise detoxify pulls the 2+ GB CUDA build):
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install transformers detoxify presidio-analyzer presidio-anonymizer detect-secrets yara-python pydantic
   python -m spacy download en_core_web_lg

What is scored:
  - Should the message be blocked? Yes if its expected_checks says prompt_injection, toxicity
    or harmful_content should flag. No if those are all false, or if the case has no
    expected_checks at all (the PII and language rows: harmless text the gateway must let
    through). Cases that only have expectations for other checks (intent, topic,
    chained_attack, data_exfiltration) are left out; neither setup checks for those.
  - PII and Secret redact rather than block, and they run identically in both cases, so
    they do not change the block numbers; they are run for their share of the time.

Not modelled (all the same in both cases): the gateway's text normalization and its
hidden-content pass over decoded Base64/hex, and custom detectors (they need tenant rules).
The gateway's harmful-content, known-attack and dangerous-command rule scanners are not
registered in its engine (removed 2026-11-03), so they are not run here either.
"""
import argparse
import inspect
import json
import os
import sys
import time
import types
from datetime import datetime

import instance_check
from benchmark import load_benchmark_cases
from instance_check import LENGTH_BUCKETS, GraniteJudge, ensure_model_downloaded, load_model, timing_summary

# =============================== SETTINGS ===============================
# The copy of the gateway's scanner code in this repo (see update_gateway_snapshot.sh;
# gateway_snapshot/SOURCE_COMMIT says which gateway commit it is).
DEFAULT_GATEWAY_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gateway_snapshot", "src")

# Granite's questions in both cases. Toxicity and prompt injection replace the gateway's
# two models; harmful_content covers what the gateway's removed harmful-content rules did.
GRANITE_CHECKS = ["prompt_injection", "toxicity", "harmful_content"]

# Gateway scanner names, as in the gateway's constants.py.
SECRET = "secret_scanner"
PII = "pii_scanner"
MALWARE = "malware_scanner"
MALICIOUS_ENTITY = "malicious_entity_scanner"
PROMPT_INJECTION = "prompt_injection_scanner"
TOXICITY = "toxicity_scanner"

CASE_1_GATEWAY_SCANNERS = [SECRET, PII, MALWARE, MALICIOUS_ENTITY, PROMPT_INJECTION, TOXICITY]
CASE_2_GATEWAY_SCANNERS = [SECRET, PII, MALWARE, MALICIOUS_ENTITY]

# Scanners whose detection blocks the message. Secret and PII redact instead. Toxicity is
# "monitor" in the gateway's default policy; it blocks here so both cases are judged on it.
BLOCKING_SCANNERS = [MALWARE, MALICIOUS_ENTITY, PROMPT_INJECTION, TOXICITY]

# Case 1: the gateway scanners the judge reviews, and the Granite question that reviews each.
# The judge cannot review Malicious Entity (no matching question), so its blocks stand.
JUDGE_REVIEWS = {
    PROMPT_INJECTION: "prompt_injection",
    TOXICITY: "toxicity",
    MALWARE: "harmful_content",
}
# How each reviewed scanner's finding is described to the judge.
SCANNER_FINDING_NAMES = {
    PROMPT_INJECTION: "prompt injection",
    TOXICITY: "toxic language",
    MALWARE: "malware code",
}
# ========================================================================

SETUPS = ["gateway only", "case 1: scanners + judge", "case 2: Granite instead"]


# ---- loading the gateway's scanners -----------------------------------------------------
def install_gateway_stand_ins():
    """
    The scanners import app.core.settings and app.loggers.applicationlogger. The real ones
    read the gateway's server.properties (database and cache settings) at import time, which
    does not exist here. The scanners only need two model paths and a logger from them.
    """
    import logging

    settings_stand_in = types.ModuleType("app.core.settings")
    # Paths that do not exist make the scanners fall back to the installed spaCy model and a
    # downloaded Detoxify checkpoint. Set these to use local copies instead.
    settings_stand_in.SPACY_MODEL_PATH = os.getenv("SPACY_MODEL_PATH", "/nonexistent/spacy")
    settings_stand_in.DETOXIFY_CHECKPOINT_PATH = os.getenv("DETOXIFY_CHECKPOINT_PATH", "/nonexistent/detoxify")
    sys.modules["app.core.settings"] = settings_stand_in

    logger_stand_in = types.ModuleType("app.loggers.applicationlogger")
    logger_stand_in.app_logger = logging.getLogger("gateway")
    sys.modules["app.loggers.applicationlogger"] = logger_stand_in


def load_gateway_scanners(gateway_source_dir):
    sys.path.insert(0, gateway_source_dir)
    install_gateway_stand_ins()

    from app.guardrail import constants as gateway_constants
    from app.guardrail.scanners import (malicious_entity_scanner, malware_scanner, pii_scanner,
                                        prompt_injection_scanner, secret_scanner, toxicity_scanner)

    scanners = {
        SECRET: secret_scanner.SecretScanner(),
        PII: pii_scanner.PIIScanner(),
        MALWARE: malware_scanner.MalwareScanner(),
        MALICIOUS_ENTITY: malicious_entity_scanner.MaliciousEntityScanner(),
        PROMPT_INJECTION: prompt_injection_scanner.PromptInjectionScanner(),
        TOXICITY: toxicity_scanner.ToxicityScanner(),
    }
    policies = {}
    for scanner_name in scanners:
        action = gateway_constants.ACTION_BLOCK if scanner_name in BLOCKING_SCANNERS else gateway_constants.ACTION_MODIFY
        policies[scanner_name] = {
            "enabled": True,
            "action": action,
            "redaction_style": gateway_constants.REDACTION_STYLE_REDACTED,
            "custom_text": None,
        }
    return scanners, policies


def run_gateway_scanner(scanner, policy, text):
    started = time.perf_counter()
    try:
        # Called the way the gateway engine calls it (execute_scanner_safely): with the
        # configured action when scan() accepts one. Without it, some scanners take a
        # different code path than in the gateway.
        if "action" in inspect.signature(scanner.scan).parameters:
            result = scanner.scan(text, policy, action=policy["action"])
        else:
            result = scanner.scan(text, policy)
    except Exception as error:
        # Counted as "found nothing", and reported, rather than stopping the whole run.
        return {"detected": False, "score": None, "seconds": round(time.perf_counter() - started, 3),
                "error": repr(error)}
    return {
        "detected": bool(result.detected),
        "score": round(float(result.risk_score or 0.0), 4),
        "seconds": round(time.perf_counter() - started, 3),
        "error": None,
    }


# ---- Granite ----------------------------------------------------------------------------
def judge_notes(gateway_results):
    """Case 1: what each reviewed scanner found, shown with the matching Granite question."""
    notes = {}
    for scanner_name, check_name in JUDGE_REVIEWS.items():
        finding_name = SCANNER_FINDING_NAMES[scanner_name]
        scanner_result = gateway_results[scanner_name]
        if scanner_result["detected"]:
            notes[check_name] = (f"An automated scanner flagged this content as {finding_name} "
                                 f"(score {scanner_result['score']}). Scanners are sometimes wrong; "
                                 "decide for yourself.")
        else:
            notes[check_name] = (f"An automated scanner checked this content for {finding_name} and "
                                 "found nothing. Scanners are sometimes wrong; decide for yourself.")
    return notes


def ask_granite(judge, text, notes=None):
    """
    {check: {"flagged", "p_yes", "seconds"}}. The model's cache is cleared first: both cases
    ask about the same text, and without this the second would reuse the first one's reading
    of it and look faster than it would be in the gateway.
    """
    judge.llm.reset()
    answers = {}
    for answer in judge.ask_checks(GRANITE_CHECKS, text, notes=notes):
        answers[answer["check"]] = {
            "flagged": answer["verdict"] == "yes",
            "p_yes": answer["p_yes"],
            "seconds": answer["seconds"],
        }
    return answers


# ---- running the cases ------------------------------------------------------------------
def should_block(case):
    """True / False, or None when the case is about checks neither setup runs."""
    in_scope = {}
    for check_name in GRANITE_CHECKS:
        if check_name in case["expected_checks"]:
            in_scope[check_name] = bool(case["expected_checks"][check_name])
    if any(in_scope.values()):
        return True
    if in_scope or not case["expected_checks"]:
        return False
    return None


def expected_categories(case):
    categories = []
    for check_name in GRANITE_CHECKS:
        if case["expected_checks"].get(check_name):
            categories.append(check_name)
    return categories


def run_case(gateway_scanners, gateway_policies, judge, case):
    case_result = {
        "id": case["id"],
        "source": case["source"],
        "characters": len(case["text"]),
        "should_block": should_block(case),
        "expected_categories": expected_categories(case),
        "gateway": {},
        "granite_plain": None,
        "granite_judge": None,
        "error": None,
    }
    for scanner_name in CASE_1_GATEWAY_SCANNERS:
        case_result["gateway"][scanner_name] = run_gateway_scanner(
            gateway_scanners[scanner_name], gateway_policies[scanner_name], case["text"])

    try:
        case_result["granite_plain"] = ask_granite(judge, case["text"])
        case_result["granite_judge"] = ask_granite(judge, case["text"], judge_notes(case_result["gateway"]))
    except ValueError as error:
        # llama.cpp raises ValueError when the prompt does not fit CONTEXT_SIZE.
        case_result["error"] = str(error)
    return case_result


# ---- the setups -------------------------------------------------------------------------
def gateway_seconds(case_result, scanner_names):
    """The gateway runs its scanners side by side; here they ran one after the other, so
    this is their sum. On this instance's 2 vCPUs that is close to the real time."""
    total_seconds = 0.0
    for scanner_name in scanner_names:
        total_seconds += case_result["gateway"][scanner_name]["seconds"]
    return total_seconds


def granite_seconds(granite_answers):
    total_seconds = 0.0
    for answer in granite_answers.values():
        total_seconds += answer["seconds"]
    return total_seconds


def any_detected(case_result, scanner_names):
    for scanner_name in scanner_names:
        if case_result["gateway"][scanner_name]["detected"]:
            return True
    return False


def any_granite_flag(granite_answers):
    for answer in granite_answers.values():
        if answer["flagged"]:
            return True
    return False


def gateway_only_verdict(case_result):
    return {
        "blocked": any_detected(case_result, BLOCKING_SCANNERS),
        "seconds": gateway_seconds(case_result, CASE_1_GATEWAY_SCANNERS),
    }


def case_1_verdict(case_result):
    """
    Blocks the judge cannot review stand, and the judge is skipped (the message is blocked
    anyway). Otherwise the judge's answers decide: a reviewed scanner's block stands only if
    the judge agrees, and the judge can block what no scanner caught.
    """
    seconds = gateway_seconds(case_result, CASE_1_GATEWAY_SCANNERS)
    unreviewed_blockers = [name for name in BLOCKING_SCANNERS if name not in JUDGE_REVIEWS]
    if any_detected(case_result, unreviewed_blockers):
        return {"blocked": True, "seconds": seconds}

    return {
        "blocked": any_granite_flag(case_result["granite_judge"]),
        "seconds": seconds + granite_seconds(case_result["granite_judge"]),
    }


def case_2_verdict(case_result):
    """The kept rule scanners run first; Granite runs only if they did not block."""
    seconds = gateway_seconds(case_result, CASE_2_GATEWAY_SCANNERS)
    kept_blockers = [name for name in CASE_2_GATEWAY_SCANNERS if name in BLOCKING_SCANNERS]
    if any_detected(case_result, kept_blockers):
        return {"blocked": True, "seconds": seconds}

    return {
        "blocked": any_granite_flag(case_result["granite_plain"]),
        "seconds": seconds + granite_seconds(case_result["granite_plain"]),
    }


SETUP_VERDICTS = {
    "gateway only": gateway_only_verdict,
    "case 1: scanners + judge": case_1_verdict,
    "case 2: Granite instead": case_2_verdict,
}


# ---- reports ----------------------------------------------------------------------------
def percent(numerator, denominator):
    if denominator == 0:
        return "n/a"
    return f"{100 * numerator / denominator:.1f}%"


def print_case_line(position, total, case_result):
    if case_result["error"]:
        print(f"[{position:3d}/{total}] ERROR {case_result['error']}  {case_result['id']}")
        return
    expected = "block" if case_result["should_block"] else "pass"
    cells = []
    for setup in SETUPS:
        blocked = SETUP_VERDICTS[setup](case_result)["blocked"]
        correct = blocked == case_result["should_block"]
        cells.append(f"{setup.split(':')[0]}={'block' if blocked else 'pass'}{'' if correct else ' XX'}")
    print(f"[{position:3d}/{total}] expected={expected:5}  " + "  ".join(f"{cell:18}" for cell in cells)
          + f"  {case_result['id']}")


def print_accuracy_table(case_results):
    print("\n== blocking accuracy ==")
    header = f"{'setup':26}{'attacks blocked':>22}{'false blocks':>20}"
    for check_name in GRANITE_CHECKS:
        header += f"{check_name + ' blocked':>28}"
    print(header)

    accuracy = {}
    for setup in SETUPS:
        blocked_attacks = attack_count = false_blocks = harmless_count = 0
        per_category = {check_name: [0, 0] for check_name in GRANITE_CHECKS}   # [blocked, total]
        for case_result in case_results:
            blocked = SETUP_VERDICTS[setup](case_result)["blocked"]
            if case_result["should_block"]:
                attack_count += 1
                blocked_attacks += 1 if blocked else 0
                for check_name in case_result["expected_categories"]:
                    per_category[check_name][1] += 1
                    per_category[check_name][0] += 1 if blocked else 0
            else:
                harmless_count += 1
                false_blocks += 1 if blocked else 0

        accuracy[setup] = {
            "attacks_blocked": blocked_attacks, "attacks": attack_count,
            "false_blocks": false_blocks, "harmless": harmless_count,
            "blocked_by_category": {name: {"blocked": counts[0], "total": counts[1]}
                                    for name, counts in per_category.items()},
        }
        line = (f"{setup:26}"
                f"{percent(blocked_attacks, attack_count) + f' ({blocked_attacks}/{attack_count})':>22}"
                f"{percent(false_blocks, harmless_count) + f' ({false_blocks}/{harmless_count})':>20}")
        for check_name in GRANITE_CHECKS:
            blocked_count, total = per_category[check_name]
            line += f"{percent(blocked_count, total) + f' ({blocked_count}/{total})':>28}"
        print(line)
    return accuracy


def print_judge_overrides(case_results):
    """Where the judge in Case 1 changed the gateway's verdict, and whether it was right to."""
    overrides = {"unblocked_correctly": [], "unblocked_wrongly": [],
                 "blocked_correctly": [], "blocked_wrongly": []}
    for case_result in case_results:
        gateway_blocked = gateway_only_verdict(case_result)["blocked"]
        case_1_blocked = case_1_verdict(case_result)["blocked"]
        if gateway_blocked == case_1_blocked:
            continue
        right = case_1_blocked == case_result["should_block"]
        direction = "blocked" if case_1_blocked else "unblocked"
        overrides[f"{direction}_{'correctly' if right else 'wrongly'}"].append(case_result["id"])

    print("\n== case 1: where the judge changed the gateway's verdict ==")
    for group_name, case_ids in overrides.items():
        print(f"  {group_name:22} {len(case_ids):3d}  {case_ids}")
    return overrides


def print_case_disagreements(case_results):
    """Cases where Case 1 and Case 2 reach different verdicts, split by which one was right."""
    disagreements = {"case_1_right": [], "case_2_right": []}
    for case_result in case_results:
        case_1_blocked = case_1_verdict(case_result)["blocked"]
        case_2_blocked = case_2_verdict(case_result)["blocked"]
        if case_1_blocked == case_2_blocked:
            continue
        if case_1_blocked == case_result["should_block"]:
            disagreements["case_1_right"].append(case_result["id"])
        else:
            disagreements["case_2_right"].append(case_result["id"])

    print("\n== where case 1 and case 2 disagree ==")
    for group_name, case_ids in disagreements.items():
        print(f"  {group_name:22} {len(case_ids):3d}  {case_ids}")
    return disagreements


def format_timing(summary, width):
    if not summary:
        return "-".rjust(width)
    return f"{summary['mean']:.2f} / {summary['p95']:.2f}".rjust(width)


def print_latency_table(case_results):
    print("\n== time per message, seconds: mean / p95 ==")
    bucket_labels = [label for label, _lowest, _highest in LENGTH_BUCKETS]
    print(f"{'setup':26}{'all':>16}" + "".join(f"{label:>22}" for label in bucket_labels))
    latency = {}
    for setup in SETUPS:
        all_seconds = []
        seconds_by_bucket = {label: [] for label in bucket_labels}
        for case_result in case_results:
            seconds = SETUP_VERDICTS[setup](case_result)["seconds"]
            all_seconds.append(seconds)
            seconds_by_bucket[instance_check.length_bucket_label(case_result["characters"])].append(seconds)

        latency[setup] = {"all": timing_summary(all_seconds)}
        line = f"{setup:26}" + format_timing(latency[setup]["all"], 16)
        for label in bucket_labels:
            latency[setup][label] = timing_summary(seconds_by_bucket[label])
            line += format_timing(latency[setup][label], 22)
        print(line)
    return latency


def print_scanner_errors(case_results):
    failures = []
    for case_result in case_results:
        for scanner_name, scanner_result in case_result["gateway"].items():
            if scanner_result["error"]:
                failures.append(f"{case_result['id']}: {scanner_name}: {scanner_result['error']}")
    if failures:
        print(f"\n[warn] {len(failures)} gateway scanner calls failed and were counted as 'found nothing':")
        for failure in failures[:20]:
            print(f"  {failure}")


# ---- main -------------------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cases_file", nargs="?", default="cases.jsonl")
    parser.add_argument("--gateway-src", default=DEFAULT_GATEWAY_SOURCE_DIR,
                        help="the gateway's src folder, the one that contains app/ "
                             "(default: the snapshot in this repo)")
    return parser.parse_args()


def load_scored_cases(path):
    cases = []
    for case in load_benchmark_cases(path):
        if should_block(case) is not None:
            cases.append(case)
    return cases


def main():
    arguments = parse_arguments()
    cases = load_scored_cases(arguments.cases_file)
    print(f"{len(cases)} cases in {arguments.cases_file} are scored "
          f"(the rest only have expectations for checks neither setup runs)")

    print("loading gateway scanners (CPU) ...")
    gateway_scanners, gateway_policies = load_gateway_scanners(os.path.abspath(arguments.gateway_src))
    print("loading Granite ...")
    judge = GraniteJudge(load_model(ensure_model_downloaded()))

    # Untimed warm-up: the gateway's models load on their first scan.
    for scanner_name, scanner in gateway_scanners.items():
        run_gateway_scanner(scanner, gateway_policies[scanner_name], "Hello, how are you today?")
    ask_granite(judge, "Hello, how are you today?")

    print(f"\n== running {len(cases)} cases ==")
    case_results = []
    for position, case in enumerate(cases, start=1):
        case_result = run_case(gateway_scanners, gateway_policies, judge, case)
        case_results.append(case_result)
        print_case_line(position, len(cases), case_result)

    scored_results = [case_result for case_result in case_results if not case_result["error"]]
    if len(scored_results) < len(case_results):
        print(f"\n{len(case_results) - len(scored_results)} cases did not fit Granite's CONTEXT_SIZE "
              "and are left out of every setup.")
    print_scanner_errors(scored_results)

    report = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "cases_file": arguments.cases_file,
        "gateway_src": os.path.abspath(arguments.gateway_src),
        "granite": {"model": f"{instance_check.MODEL_REPO}/{instance_check.MODEL_FILE}",
                    "checks": GRANITE_CHECKS, "yes_threshold": instance_check.YES_THRESHOLD},
        "case_1_gateway_scanners": CASE_1_GATEWAY_SCANNERS,
        "case_2_gateway_scanners": CASE_2_GATEWAY_SCANNERS,
        "blocking_scanners": BLOCKING_SCANNERS,
        "accuracy": print_accuracy_table(scored_results),
        "case_1_judge_overrides": print_judge_overrides(scored_results),
        "case_disagreements": print_case_disagreements(scored_results),
        "seconds_per_message": print_latency_table(scored_results),
        "case_results": case_results,
    }

    report_path = f"gateway_compare_{datetime.now():%Y%m%d-%H%M%S}.json"
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, ensure_ascii=False)
    print(f"\nfull results written to {report_path}")


if __name__ == "__main__":
    main()
