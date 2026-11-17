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
./try.sh
```

Type a message and press Enter. It prints the verdict, P(yes) and time for each check.
Earlier messages are kept as the conversation, so from the second message on
`chained_attack` also runs. `/reset` starts a new conversation and `/quit` exits. The
topic/intent policy and `YES_THRESHOLD` come from the `SETTINGS` block in
`instance_check.py`.

## Benchmark

```bash
./bench.sh                       # cases.jsonl, then 2 and 4 concurrent users
./bench.sh --concurrency 2,4,8   # choose the concurrent-user levels
./bench.sh --concurrency 0       # accuracy only, no load test
./bench.sh my_cases.jsonl        # your own file (.jsonl or .csv)
```

Use the `.sh` wrappers rather than calling `python` directly. They also source
`cuda_env.sh`, which points the loader at the CUDA libraries in `.venv`. Without it you
get `libcudart.so.12: cannot open shared object file`.

### The cases

`cases.jsonl` holds 327 cases (93 attacks, 234 benign):

- 180 from public datasets: deepset prompt injections, XSTest, malicious code, Civil
  Comments, ai4privacy PII and language identification.
- 85 synthetic cases (`source: synthetic_checks`), at least 17 for each judge check, with
  a mix of short and long (1,000+ character) messages.
- 62 synthetic long cases (`source: synthetic_long`), so each length bucket from 1,000
  characters up has 15 attacks and 15 benign cases. They are real-looking work documents
  (policies, release notes, support threads, meeting notes, a postmortem). Attacks add a
  hidden prompt-injection or exfiltration instruction at the start, middle or end, or a
  harmful request after the documents. The longest is about 11,900 characters, which still
  fits `CONTEXT_SIZE = 4096` tokens.

| Length | Attacks | Benign |
|---|---|---|
| under 1,000 characters | 48 | 189 |
| 1,000-4,999 | 15 | 15 |
| 5,000-9,999 | 15 | 15 |
| 10,000+ | 15 | 15 |

Every row has:

- `label`: `attack` or `benign` for the message as a whole. Attack means prompt
  injection, a harmful request, a chained (multi-turn) attack or data exfiltration. Toxic,
  off-topic and off-purpose messages are policy problems, not attacks, so they are
  `benign` here.
- `expected_checks`: whether each judge check should flag it, for example
  `{"toxicity": true}`. In the dataset rows this comes from `expected_scanners`: `malware`
  becomes `harmful_content`, and XSTest's unsafe "contrast" prompts are `harmful_content: true`.
  PII and language rows have none, because no judge check scores them.
- Optional `id`, `source`, `category`, `context` (the topic/intent policy) and `history`.
  `instance_check.py` runs every synthetic row for each check in its `expected_checks`.

### What it runs and prints

Each case runs the checks in `ATTACK_CHECKS` (prompt injection, harmful content, data
exfiltration, plus chained attack when there is history), and counts as an attack if any
of them says yes. Checks named in `expected_checks` also run. It prints these tables,
tab-separated so they paste into a spreadsheet:

```
Correct  Accuracy  TP  FN  FP  TN  Attack recall  Mean time
Length  Cases  Overall accuracy; mean time  Attack recall; mean time  Benign accuracy; mean time
Check  Cases  Correct  Accuracy  TP  FN  FP  TN  Recall  Mean time
Concurrent users  Cases  Accuracy  Changed verdicts  p50 latency  p95 latency  Max latency  Cases/s
```

### Concurrent load

The model serves one request at a time, so it sits behind a queue. The load test replays
every case with N users sending at once, each sending the next message as soon as its
answer arrives. Latency is measured from send to answer, so it includes time spent
waiting in the queue. All checks for one message are served together, so the model still
reuses the already-read message between checks.

For each level it reports PASS/FAIL on whether any verdict changed versus 1 user, and on
p95 latency against `TARGET_SECONDS_PER_MESSAGE` and `GATEWAY_TIMEOUT_SECONDS` from
`instance_check.py`. Verdicts should never change because the requests are served one
after another. Latency grows roughly with the number of users and throughput stays flat.
These numbers show whether a batching server (llama.cpp's `llama-server --parallel`) is
worth adding.

Per-case results go to `benchmark_<time>.json`. A case that does not fit `CONTEXT_SIZE`
is reported and left out of the numbers.

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
| Feature accuracy | The synthetic rows in `cases.jsonl`, once per check in `expected_checks`, for prompt injection, toxicity, harmful content, topic, intent, chained (multi-turn) attack and data exfiltration. The summary has one line per check and one per message length (`LENGTH_BUCKETS`) |
| Latency per message | Runs all `PIPELINE_CHECKS` on each sample message, the way a gateway would |
| Long input | One call near `CONTEXT_SIZE` to find peak GPU memory and the slowest case |

It prints a PASS / WARN / FAIL line for each item and writes `results_<time>.json`.

## Tuning

Edit the `SETTINGS` block at the top of `instance_check.py`:

- **g6f.large defaults:** the L4-3Q slice reports 2,856 MiB, but only about 2.1 GB is
  usable after CUDA's own overhead. The whole model (41 layers) does not fit with its KV
  cache and compute buffers, so the defaults are `GPU_LAYERS = 36`, `BATCH_SIZE = 256`,
  `CONTEXT_SIZE = 4096`, flash attention and a q8 KV cache. The 5 layers left on the CPU
  cost little, because the judge reads one output token per call. 38 layers loads but
  crashes with a CUDA error mid-run, because CUDA needs more working memory later.
- **Model does not fit** ("Failed to create llama_context"): this is almost always GPU
  memory. Lower `GPU_LAYERS` (e.g. 36), `BATCH_SIZE` (e.g. 128) or `CONTEXT_SIZE` (e.g.
  2048), and keep `QUANTIZED_KV_CACHE = True` and `FLASH_ATTENTION = True`.
- **CPU baseline:** set `GPU_LAYERS = 0`.
- **More cases:** add lines to `cases.jsonl` with `text` and `label` (`attack`/`benign`).
  Add `check` to make it a per-check case, used by both scripts, and optionally
  `history` and `context`.

## Notes

- One check is one yes/no call. Checks run on the same message reuse the
  already-read content, so later checks are much cheaper than the first.
- Throughput is measured for a single stream; requests are not batched.
- The 85 per-check cases are synthetic; the other 180 come from public datasets.
