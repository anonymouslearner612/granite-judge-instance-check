"""
Time per message-length bucket for each model in a model_compare.py run.

    python3 length_timing.py comparison_20261120_105335

Reads the per-model result files (<model>.json) the run left in that folder; no model is
loaded. A case's time is the sum of all its checks, the same measure as "s/case" in
comparison.txt: the time a gateway would wait for every check on one message.
"""
import json
import os
import sys

from instance_check import LENGTH_BUCKETS, length_bucket_label, timing_summary


def model_names_in_run_order(run_directory):
    """The models in the order the run benchmarked them, as listed in comparison.json."""
    with open(os.path.join(run_directory, "comparison.json"), encoding="utf-8") as summary_file:
        summaries = json.load(summary_file)
    model_names = []
    for summary in summaries:
        if summary["status"] == "ok":
            model_names.append(summary["model"])
    return model_names


def seconds_by_bucket(case_results):
    """{bucket label: [seconds for each case that ran]}"""
    bucket_seconds = {}
    for label, _lowest, _highest in LENGTH_BUCKETS:
        bucket_seconds[label] = []
    for case_result in case_results:
        if case_result["error"]:
            continue
        case_seconds = sum(check_result["seconds"] for check_result in case_result["checks"])
        bucket_seconds[length_bucket_label(case_result["characters"])].append(case_seconds)
    return bucket_seconds


def format_seconds(timing, key):
    if key not in timing:
        return "-"
    return f"{timing[key]:.2f}"


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python3 length_timing.py <comparison run folder>")
    run_directory = sys.argv[1]

    print("Model\tLength\tCases\tMean s\tMedian s\tp95 s\tMax s")
    for model_name in model_names_in_run_order(run_directory):
        with open(os.path.join(run_directory, f"{model_name}.json"), encoding="utf-8") as report_file:
            report = json.load(report_file)
        for label, seconds_list in seconds_by_bucket(report["cases"]).items():
            timing = timing_summary(seconds_list)
            print("\t".join([
                model_name, label, str(len(seconds_list)),
                format_seconds(timing, "mean"), format_seconds(timing, "median"),
                format_seconds(timing, "p95"), format_seconds(timing, "max"),
            ]))
        print()


if __name__ == "__main__":
    main()
