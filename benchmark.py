#!/usr/bin/env python3
"""
Benchmark the Granite judge on labeled cases, alone and under concurrent load.

Run:   python benchmark.py                          (cases.jsonl, then 2 and 4 concurrent users)
       python benchmark.py my_cases.jsonl           (your own file, .jsonl or .csv)
       python benchmark.py --concurrency 2,4,8      (choose the concurrent-user levels)
       python benchmark.py --concurrency 0          (accuracy only, no load test)

Input, one case per JSONL line or CSV row:
  text             the message to judge (required)
  label            attack or benign (required; yes/no, true/false and 1/0 also work, and
                   the column may be called "expected"). Attack means prompt injection, a
                   harmful request, a chained attack or data exfiltration.
  expected_checks  JSONL only: {"check name": true/false} - whether each judge check
                   should flag the message (optional; feeds the per-check table)
  id, source, category   copied into the per-case results (optional)
  context          topic/intent policy for this case (optional)
  history          JSONL only: earlier turns as [{"role": "user", "content": "..."}]. When
                   it is not empty, chained_attack also runs.

Each case runs ATTACK_CHECKS (plus chained_attack when there is history) and counts as an
attack when any of them says yes, the way a gateway would block it. Checks named in
expected_checks also run, for the per-check table.

Concurrency: the model serves one request at a time, so requests from concurrent users wait
in a queue. The load test replays every case with N users sending at once and reports how
much latency grows and whether any verdict changed. All checks for one message are served
together, so the model can reuse the already-read message between checks.

Model settings, YES_THRESHOLD and the latency targets come from instance_check.py.
"""
import argparse
import csv
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import instance_check
from instance_check import (CHECK_QUESTIONS, LENGTH_BUCKETS, GraniteJudge, ensure_model_downloaded,
                            load_model, percentile)

DEFAULT_CASES_FILE = "cases.jsonl"
DEFAULT_CONCURRENCY_LEVELS = "2,4"

# Checks whose "yes" means the message is an attack. toxicity, topic and intent are policy
# checks: cases.jsonl labels toxic, off-topic and off-purpose text benign, and scores those
# checks in the per-check table instead. chained_attack is added when a case has history.
ATTACK_CHECKS = ["prompt_injection", "harmful_content", "data_exfiltration"]

ATTACK_LABELS = {"yes", "true", "1", "attack", "malicious"}
BENIGN_LABELS = {"no", "false", "0", "benign", "safe"}


# ---- loading --------------------------------------------------------------------------
def parse_label(raw_label, case_id):
    label = str(raw_label).strip().lower()
    if label in ATTACK_LABELS:
        return True
    if label in BENIGN_LABELS:
        return False
    raise ValueError(f"case {case_id}: cannot read label {raw_label!r} as attack or benign")


def read_raw_cases(path):
    if path.lower().endswith(".csv"):
        with open(path, encoding="utf-8", newline="") as csv_file:
            return list(csv.DictReader(csv_file))

    raw_cases = []
    with open(path, encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            if line.strip():
                raw_cases.append(json.loads(line))
    return raw_cases


def load_benchmark_cases(path):
    cases = []
    for row_number, raw_case in enumerate(read_raw_cases(path), start=1):
        case_id = raw_case.get("id") or str(row_number)

        raw_label = raw_case.get("label", raw_case.get("expected"))
        if raw_label is None:
            raise ValueError(f"case {case_id}: needs a 'label' (or 'expected') field")

        expected_checks = raw_case.get("expected_checks") or {}
        for check_name in expected_checks:
            if check_name not in CHECK_QUESTIONS:
                raise ValueError(f"case {case_id}: unknown check {check_name!r} in expected_checks")

        cases.append({
            "id": case_id,
            "source": raw_case.get("source"),
            "category": raw_case.get("category"),
            "text": raw_case["text"],
            "is_attack": parse_label(raw_label, case_id),
            "expected_checks": expected_checks,
            "context": raw_case.get("context") or None,
            "history": raw_case.get("history") or [],
        })
    return cases


# ---- running one case -----------------------------------------------------------------
def attack_checks_for_case(case):
    if case["history"]:
        return ATTACK_CHECKS + ["chained_attack"]
    return ATTACK_CHECKS


def checks_for_case(case):
    """Attack checks first, then any extra checks the case has expectations for."""
    checks_to_run = list(attack_checks_for_case(case))
    for check_name in case["expected_checks"]:
        if check_name not in checks_to_run:
            checks_to_run.append(check_name)
    return checks_to_run


def run_case(judge, case):
    """Runs every check for the case. Returns a result dict; "error" is set if it could not run."""
    case_result = {
        "id": case["id"],
        "source": case["source"],
        "category": case["category"],
        "characters": len(case["text"]),
        "is_attack": case["is_attack"],
        "expected_checks": case["expected_checks"],
        "flagged": None,
        "correct": None,
        "attack_seconds": None,
        "checks": [],
        "error": None,
    }
    try:
        # All checks in one call, so a long text is read once per window, not once per check.
        case_result["checks"] = judge.ask_checks(
            checks_for_case(case), case["text"], case["history"], case["context"])
    except ValueError as error:
        # llama.cpp raises ValueError when the prompt does not fit CONTEXT_SIZE.
        case_result["error"] = str(error)
        return case_result

    attack_check_names = attack_checks_for_case(case)
    flagged = False
    attack_seconds = 0.0
    for check_result in case_result["checks"]:
        if check_result["check"] not in attack_check_names:
            continue
        attack_seconds += check_result["seconds"]
        if check_result["verdict"] == "yes":
            flagged = True

    case_result["flagged"] = flagged
    case_result["correct"] = flagged == case["is_attack"]
    case_result["attack_seconds"] = round(attack_seconds, 3)
    return case_result


def run_cases_one_at_a_time(judge, cases):
    case_results = []
    for case_number, case in enumerate(cases, start=1):
        case_result = run_case(judge, case)
        case_results.append(case_result)
        if case_result["error"]:
            status = "ERROR"
        elif case_result["correct"]:
            status = "ok"
        else:
            status = "WRONG"
        print(f"  [{case_number}/{len(cases)}] {case['id']}: {status}", flush=True)
    return case_results


# ---- concurrent load ------------------------------------------------------------------
def run_cases_with_concurrent_users(judge, cases, user_count):
    """
    user_count users send cases at the same time; each sends its next case as soon as the
    previous answer arrives. A single-worker executor is the queue in front of the model:
    it serves one case at a time, in arrival order. Latency is measured from the moment a
    user sends a case until its answer comes back, so it includes the time spent waiting.
    """
    model_queue = ThreadPoolExecutor(max_workers=1)

    def send_and_wait(case):
        sent_at = time.perf_counter()
        case_result = model_queue.submit(run_case, judge, case).result()
        case_result["latency_seconds"] = round(time.perf_counter() - sent_at, 3)
        return case_result

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=user_count) as users:
        case_results = list(users.map(send_and_wait, cases))
    wall_seconds = time.perf_counter() - started

    model_queue.shutdown()
    return case_results, wall_seconds


def describe_load_run(user_count, case_results, wall_seconds, baseline_results):
    baseline_by_id = {}
    for baseline_result in baseline_results:
        baseline_by_id[baseline_result["id"]] = baseline_result

    ran_results = [case_result for case_result in case_results if not case_result["error"]]
    changed_ids = []
    for case_result in ran_results:
        baseline_result = baseline_by_id[case_result["id"]]
        if baseline_result["error"]:
            continue
        baseline_verdicts = [check_result["verdict"] for check_result in baseline_result["checks"]]
        verdicts = [check_result["verdict"] for check_result in case_result["checks"]]
        if verdicts != baseline_verdicts:
            changed_ids.append(case_result["id"])

    latencies = [case_result["latency_seconds"] for case_result in ran_results]
    correct_count = sum(1 for case_result in ran_results if case_result["correct"])
    return {
        "users": user_count,
        "cases": len(ran_results),
        "correct": correct_count,
        "changed_verdict_ids": changed_ids,
        "latency_p50": round(statistics.median(latencies), 3) if latencies else None,
        "latency_p95": round(percentile(latencies, 95), 3) if latencies else None,
        "latency_max": round(max(latencies), 3) if latencies else None,
        "cases_per_second": round(len(case_results) / wall_seconds, 2),
    }


def describe_baseline_run(case_results, wall_seconds):
    """The one-user run in the same shape as a load run, as the first row of the load table."""
    ran_results = [case_result for case_result in case_results if not case_result["error"]]
    latencies = []
    for case_result in ran_results:
        latencies.append(sum(check_result["seconds"] for check_result in case_result["checks"]))
    return {
        "users": 1,
        "cases": len(ran_results),
        "correct": sum(1 for case_result in ran_results if case_result["correct"]),
        "changed_verdict_ids": [],
        "latency_p50": round(statistics.median(latencies), 3) if latencies else None,
        "latency_p95": round(percentile(latencies, 95), 3) if latencies else None,
        "latency_max": round(max(latencies), 3) if latencies else None,
        "cases_per_second": round(len(case_results) / wall_seconds, 2),
    }


# ---- summarising ----------------------------------------------------------------------
def format_percent(numerator, denominator):
    if denominator == 0:
        return "n/a"
    return f"{100 * numerator / denominator:.1f}%"


def format_mean_seconds(seconds_list):
    if not seconds_list:
        return "n/a"
    return f"{statistics.mean(seconds_list):.3f} s"


def count_confusion(pairs):
    """pairs: (should_flag, flagged, seconds). Returns TP/FN/FP/TN counts and timings."""
    counts = {"tp": 0, "fn": 0, "fp": 0, "tn": 0,
              "all_seconds": [], "positive_seconds": [], "negative_seconds": []}
    for should_flag, flagged, seconds in pairs:
        counts["all_seconds"].append(seconds)
        if should_flag:
            counts["positive_seconds"].append(seconds)
            counts["tp" if flagged else "fn"] += 1
        else:
            counts["negative_seconds"].append(seconds)
            counts["fp" if flagged else "tn"] += 1
    counts["total"] = counts["tp"] + counts["fn"] + counts["fp"] + counts["tn"]
    counts["correct"] = counts["tp"] + counts["tn"]
    return counts


def attack_pairs(case_results):
    pairs = []
    for case_result in case_results:
        if not case_result["error"]:
            pairs.append((case_result["is_attack"], case_result["flagged"], case_result["attack_seconds"]))
    return pairs


def check_pairs(case_results, check_name):
    pairs = []
    for case_result in case_results:
        if case_result["error"] or check_name not in case_result["expected_checks"]:
            continue
        for check_result in case_result["checks"]:
            if check_result["check"] == check_name:
                should_flag = case_result["expected_checks"][check_name]
                pairs.append((should_flag, check_result["verdict"] == "yes", check_result["seconds"]))
    return pairs


def results_in_bucket(case_results, lowest, highest):
    matching_results = []
    for case_result in case_results:
        too_short = case_result["characters"] < lowest
        too_long = highest is not None and case_result["characters"] > highest
        if not too_short and not too_long:
            matching_results.append(case_result)
    return matching_results


def print_attack_table(case_results):
    counts = count_confusion(attack_pairs(case_results))
    print("Correct\tAccuracy\tTP\tFN\tFP\tTN\tAttack recall\tMean time")
    print("\t".join([
        f"{counts['correct']}/{counts['total']}",
        format_percent(counts["correct"], counts["total"]),
        str(counts["tp"]), str(counts["fn"]), str(counts["fp"]), str(counts["tn"]),
        format_percent(counts["tp"], counts["tp"] + counts["fn"]),
        format_mean_seconds(counts["all_seconds"]),
    ]))


def print_length_table(case_results):
    print("Length\tCases\tOverall accuracy; mean time\tAttack recall; mean time\tBenign accuracy; mean time")
    for label, lowest, highest in LENGTH_BUCKETS:
        counts = count_confusion(attack_pairs(results_in_bucket(case_results, lowest, highest)))
        if counts["total"] == 0:
            continue
        print("\t".join([
            label,
            str(counts["total"]),
            f"{format_percent(counts['correct'], counts['total'])}; {format_mean_seconds(counts['all_seconds'])}",
            f"{format_percent(counts['tp'], counts['tp'] + counts['fn'])}; {format_mean_seconds(counts['positive_seconds'])}",
            f"{format_percent(counts['tn'], counts['fp'] + counts['tn'])}; {format_mean_seconds(counts['negative_seconds'])}",
        ]))


def print_check_table(case_results):
    print("Check\tCases\tCorrect\tAccuracy\tTP\tFN\tFP\tTN\tRecall\tMean time")
    for check_name in CHECK_QUESTIONS:
        counts = count_confusion(check_pairs(case_results, check_name))
        if counts["total"] == 0:
            continue
        print("\t".join([
            check_name,
            str(counts["total"]),
            f"{counts['correct']}/{counts['total']}",
            format_percent(counts["correct"], counts["total"]),
            str(counts["tp"]), str(counts["fn"]), str(counts["fp"]), str(counts["tn"]),
            format_percent(counts["tp"], counts["tp"] + counts["fn"]),
            format_mean_seconds(counts["all_seconds"]),
        ]))


def print_load_table(load_runs):
    print("Concurrent users\tCases\tAccuracy\tChanged verdicts\tp50 latency\tp95 latency\tMax latency\tCases/s")
    for load_run in load_runs:
        print("\t".join([
            str(load_run["users"]),
            str(load_run["cases"]),
            format_percent(load_run["correct"], load_run["cases"]),
            str(len(load_run["changed_verdict_ids"])),
            f"{load_run['latency_p50']:.3f} s",
            f"{load_run['latency_p95']:.3f} s",
            f"{load_run['latency_max']:.3f} s",
            f"{load_run['cases_per_second']:.2f}",
        ]))


def print_load_verdicts(load_runs):
    """PASS/WARN/FAIL per load level against the targets in instance_check.py."""
    target_seconds = instance_check.TARGET_SECONDS_PER_MESSAGE
    timeout_seconds = instance_check.GATEWAY_TIMEOUT_SECONDS
    for load_run in load_runs[1:]:
        users = load_run["users"]
        if load_run["changed_verdict_ids"]:
            print(f"[FAIL] {users} users: verdicts changed vs 1 user for {', '.join(load_run['changed_verdict_ids'])}")
        else:
            print(f"[PASS] {users} users: every verdict matches the 1-user run")

        p95 = load_run["latency_p95"]
        if load_run["latency_max"] > timeout_seconds:
            print(f"[FAIL] {users} users: slowest case {load_run['latency_max']:.2f}s is over the {timeout_seconds}s gateway timeout")
        elif p95 > target_seconds:
            print(f"[WARN] {users} users: p95 latency {p95:.2f}s is over the {target_seconds}s target")
        else:
            print(f"[PASS] {users} users: p95 latency {p95:.2f}s is within the {target_seconds}s target")


def print_errors(case_results):
    failed_results = [case_result for case_result in case_results if case_result["error"]]
    if not failed_results:
        return
    print(f"\n{len(failed_results)} case(s) could not run and are left out of the numbers above:")
    for case_result in failed_results:
        print(f"  {case_result['id']} ({case_result['characters']} characters): {case_result['error']}")
    print("  If they are too long, raise CONTEXT_SIZE in instance_check.py.")


# ---- main -----------------------------------------------------------------------------
def parse_concurrency_levels(raw_levels):
    levels = []
    for raw_level in raw_levels.split(","):
        level = int(raw_level)
        if level > 1 and level not in levels:
            levels.append(level)
    return levels


def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmark the Granite judge on labeled cases.")
    parser.add_argument("cases_file", nargs="?", default=DEFAULT_CASES_FILE,
                        help=f"JSONL or CSV file with labeled cases (default {DEFAULT_CASES_FILE})")
    parser.add_argument("--concurrency", default=DEFAULT_CONCURRENCY_LEVELS,
                        help="comma-separated numbers of concurrent users to test, e.g. 2,4,8; "
                             "0 skips the load test (default %(default)s)")
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    concurrency_levels = parse_concurrency_levels(arguments.concurrency)

    cases = load_benchmark_cases(arguments.cases_file)
    print(f"loaded {len(cases)} cases from {arguments.cases_file}; attack checks: {', '.join(ATTACK_CHECKS)}")

    print("loading model ...")
    judge = GraniteJudge(load_model(ensure_model_downloaded()))
    judge.ask("prompt_injection", "Hello, how are you today?")   # warm-up, not timed

    print("\n== 1 user ==")
    started = time.perf_counter()
    case_results = run_cases_one_at_a_time(judge, cases)
    load_runs = [describe_baseline_run(case_results, time.perf_counter() - started)]

    load_results = {}
    for user_count in concurrency_levels:
        print(f"\n== {user_count} concurrent users ==", flush=True)
        level_results, wall_seconds = run_cases_with_concurrent_users(judge, cases, user_count)
        load_runs.append(describe_load_run(user_count, level_results, wall_seconds, case_results))
        load_results[user_count] = level_results

    print("\n== attacks ==")
    print_attack_table(case_results)
    print("\n== by length ==")
    print_length_table(case_results)
    print("\n== per check ==")
    print_check_table(case_results)
    if concurrency_levels:
        print("\n== concurrent load ==")
        print_load_table(load_runs)
        print_load_verdicts(load_runs)
    print_errors(case_results)

    report_path = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report = {
        "input": arguments.cases_file,
        "attack_checks": ATTACK_CHECKS,
        "load_runs": load_runs,
        "cases": case_results,
        "load_case_results": load_results,
    }
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)
    print(f"\nper-case results written to {report_path}")


if __name__ == "__main__":
    main()
