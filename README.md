# Granite judge instance check

Checks whether a GPU instance (built for an AWS `g6f.large`) can run a quantized
**Granite 4.0 Micro** (`Q4_K_M`, about 2.1 GB) as an LLM guardrail judge. It also
checks whether the judge can do each guardrail feature.

## Quick start (on the instance)

```bash
curl -fsSL https://raw.githubusercontent.com/anonymouslearner612/granite-judge-instance-check/main/get.sh | bash
cd ~/granite-judge-instance-check
./setup.sh      # first time only
./run.sh
```

To pick up later changes, run `get.sh` again or `git pull` in the folder, then `./run.sh`.

## Try your own messages

```bash
./try.sh        # or: . .venv/bin/activate && python try_prompt.py
```

Type a message and press Enter. It prints the verdict, P(yes) and time for each check.
Earlier messages are kept as the conversation, so from the second message on
`chained_attack` also runs. `/reset` starts a new conversation and `/quit` exits. The
topic/intent policy and `YES_THRESHOLD` come from the `SETTINGS` block in
`instance_check.py`.

## Benchmark

```bash
. .venv/bin/activate
python benchmark.py                     # all cases in cases.jsonl
python benchmark.py my_cases.jsonl      # or your own file (.jsonl or .csv)
```

`cases.jsonl` holds 217 labeled cases (61 attacks, 156 benign): 180 from deepset prompt
injections, XSTest, malicious code, Civil Comments, ai4privacy PII and language
identification, plus 37 synthetic per-check cases (`source: synthetic_checks`). The
synthetic ones have a `check` field, so only that check runs on them, and `attack` there
means "this check should flag it". Each case needs `text` and `label` (`attack`/`benign`; `yes`/`no`,
`true`/`false` and `1`/`0` also work, and the column may be called `expected`).
Optional fields are `id`, `source`, `category`, `check` (run only that check),
`context` and `history`.

Without `check`, the checks in `ATTACK_CHECKS` (prompt injection, harmful content, data
exfiltration) run, and the case counts as flagged if any of them says yes. Toxicity, topic
and intent are left out because the labels treat toxic, PII and off-topic text as benign.

It prints an overall summary and a breakdown by message length, both tab-separated so
they paste into a spreadsheet:

```
Correct  Accuracy  TP  FN  FP  TN  Attack recall  Mean time
Length  Cases  Overall accuracy; mean time  Attack recall; mean time  Benign accuracy; mean time
```

Per-case results go to `benchmark_<time>.json`. A case that does not fit `CONTEXT_SIZE`
is reported and left out of the numbers. A 10,000-character message is about 2,500
tokens, which fits in the default 4096.

## Instance prerequisites

- **NVIDIA GRID driver 18.4 or later.** G6f fractional GPUs need it; `nvidia-smi`
  must work before you start.
- **CUDA compiler (`nvcc`) 12.x.** `setup.sh` compiles `llama-cpp-python` on the
  instance (15-40 minutes), because the prebuilt CUDA wheels crash with "Illegal
  instruction" on the AMD CPUs in G6/G6f. On Amazon Linux 2023 it installs the compiler
  from NVIDIA's repo by itself; elsewhere, install it first.
- Python 3.10+ with `venv`, and `git`.
- About 5 GB of free disk for the env and model.

## What it measures

| Area | How |
|---|---|
| Hardware | GPU name, GPU memory, driver, vCPUs, RAM |
| Model fit | Loads the whole model onto the GPU and records the GPU memory it took |
| Feature accuracy | The synthetic cases in `cases.jsonl` (rows with a `check` field) for prompt injection, toxicity, harmful content, topic, intent, chained (multi-turn) attack and data exfiltration |
| Latency per message | Runs all `PIPELINE_CHECKS` on each sample message, the way a gateway would |
| Long input | One call near `CONTEXT_SIZE` to find peak GPU memory and the slowest case |

It prints a PASS / WARN / FAIL line for each item and writes `results_<time>.json`.

## Tuning

Edit the `SETTINGS` block at the top of `instance_check.py`:

- **Model does not fit:** lower `CONTEXT_SIZE` (for example 2048), and keep
  `QUANTIZED_KV_CACHE = True` and `FLASH_ATTENTION = True`.
- **CPU baseline:** set `GPU_LAYERS = 0`.
- **More cases:** add lines to `cases.jsonl` with `text` and `label` (`attack`/`benign`).
  Add `check` to make it a per-check case, used by both scripts, and optionally
  `history` and `context`.

## Notes

- One check is one yes/no call. Checks run on the same message reuse the
  already-read content, so later checks are much cheaper than the first.
- Throughput is measured for a single stream; requests are not batched.
- The per-check cases are synthetic; the other 180 come from public datasets.
