#!/usr/bin/env python3
"""
Re-scores a saved benchmark run at different YES_THRESHOLD values, without running the
model again. Every check's P(yes) is in benchmark_<time>.json, so a lower threshold
(catch more, more false alarms) can be tried in seconds.

Run:   python threshold_sweep.py benchmark_20261117_140148.json
       python threshold_sweep.py results_20261119-093837.json     (instance_check.py run)
       python threshold_sweep.py benchmark_*.json 0.1,0.2,0.3     (choose thresholds)

Long texts are checked in windows and reading stops at the first flagged window, so a
flagged long text keeps only that window's P(yes). Thresholds below the run's YES_THRESHOLD
are scored exactly; thresholds above it can undercount flags on long texts.

Note: picking the threshold on the same cases you report on flatters the result. Use it to
see the trade-off, then confirm the chosen value on a fresh run.
"""
import json
import sys

DEFAULT_THRESHOLDS = "0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5"


def check_flags(check_result, threshold):
    if check_result.get("p_yes") is None:
        return check_result["verdict"] == "yes"
    return check_result["p_yes"] >= threshold


def score_attacks(case_results, attack_check_names, threshold):
    true_positives = false_negatives = false_positives = true_negatives = 0
    for case_result in case_results:
        if case_result["error"]:
            continue
        flagged = False
        for check_result in case_result["checks"]:
            if check_result["check"] in attack_check_names and check_flags(check_result, threshold):
                flagged = True
        if case_result["is_attack"] and flagged:
            true_positives += 1
        elif case_result["is_attack"]:
            false_negatives += 1
        elif flagged:
            false_positives += 1
        else:
            true_negatives += 1
    return true_positives, false_negatives, false_positives, true_negatives


def score_check(case_results, check_name, threshold):
    """Recall and false alarms for one check, on the cases that have an expectation for it."""
    caught = should_flag_count = false_alarms = should_pass_count = 0
    for case_result in case_results:
        if case_result["error"] or check_name not in case_result["expected_checks"]:
            continue
        for check_result in case_result["checks"]:
            if check_result["check"] != check_name:
                continue
            flagged = check_flags(check_result, threshold)
            if case_result["expected_checks"][check_name]:
                should_flag_count += 1
                caught += 1 if flagged else 0
            else:
                should_pass_count += 1
                false_alarms += 1 if flagged else 0
    return caught, should_flag_count, false_alarms, should_pass_count


def percent(numerator, denominator):
    if denominator == 0:
        return "n/a"
    return f"{100 * numerator / denominator:.1f}%"


def sweep_instance_check_report(check_results, thresholds):
    """
    results_*.json from instance_check.py has one flat row per (case, check) with "expected"
    set to "yes" or "no", instead of benchmark.py's cases with nested checks.
    """
    check_names = []
    for check_result in check_results:
        if check_result["check"] not in check_names:
            check_names.append(check_result["check"])

    print("== per check: recall / false-alarm rate ==")
    print("Threshold\t" + "\t".join(check_names))
    for threshold in thresholds:
        cells = []
        for check_name in check_names:
            caught = should_flag_count = false_alarms = should_pass_count = 0
            for check_result in check_results:
                if check_result["check"] != check_name or check_result["verdict"] == "error":
                    continue
                flagged = check_flags(check_result, threshold)
                if check_result["expected"] == "yes":
                    should_flag_count += 1
                    caught += 1 if flagged else 0
                else:
                    should_pass_count += 1
                    false_alarms += 1 if flagged else 0
            cells.append(f"{percent(caught, should_flag_count)} / {percent(false_alarms, should_pass_count)}")
        print(f"{threshold:.2f}\t" + "\t".join(cells))


def main():
    if len(sys.argv) not in (2, 3):
        print("usage: python threshold_sweep.py <benchmark_*.json or results_*.json> [thresholds, e.g. 0.2,0.3]")
        sys.exit(2)
    with open(sys.argv[1], encoding="utf-8") as report_file:
        report = json.load(report_file)
    raw_thresholds = sys.argv[2] if len(sys.argv) == 3 else DEFAULT_THRESHOLDS
    thresholds = [float(raw_threshold) for raw_threshold in raw_thresholds.split(",")]

    if "case_results" in report:
        sweep_instance_check_report(report["case_results"], thresholds)
        return

    case_results = report["cases"]
    # chained_attack only runs on cases with history, and counts as an attack check there.
    attack_check_names = list(report["attack_checks"]) + ["chained_attack"]

    print("== attacks ==")
    print("Threshold\tAccuracy\tTP\tFN\tFP\tTN\tAttack recall\tBenign accuracy")
    for threshold in thresholds:
        tp, fn, fp, tn = score_attacks(case_results, attack_check_names, threshold)
        print("\t".join([f"{threshold:.2f}", percent(tp + tn, tp + fn + fp + tn),
                         str(tp), str(fn), str(fp), str(tn),
                         percent(tp, tp + fn), percent(tn, fp + tn)]))

    check_names = []
    for case_result in case_results:
        for check_name in case_result["expected_checks"]:
            if check_name not in check_names:
                check_names.append(check_name)

    print("\n== per check: recall / false-alarm rate ==")
    print("Threshold\t" + "\t".join(check_names))
    for threshold in thresholds:
        cells = []
        for check_name in check_names:
            caught, should_flag_count, false_alarms, should_pass_count = score_check(case_results, check_name, threshold)
            cells.append(f"{percent(caught, should_flag_count)} / {percent(false_alarms, should_pass_count)}")
        print(f"{threshold:.2f}\t" + "\t".join(cells))


if __name__ == "__main__":
    main()
