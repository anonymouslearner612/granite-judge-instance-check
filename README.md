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
| Feature accuracy | Synthetic yes/no cases in `cases.jsonl` for prompt injection, toxicity, harmful content, topic, intent, chained (multi-turn) attack and data exfiltration |
| Latency per message | Runs all `PIPELINE_CHECKS` on each sample message, the way a gateway would |
| Long input | One call near `CONTEXT_SIZE` to find peak GPU memory and the slowest case |

It prints a PASS / WARN / FAIL line for each item and writes `results_<time>.json`.

## Tuning

Edit the `SETTINGS` block at the top of `instance_check.py`:

- **Model does not fit:** lower `CONTEXT_SIZE` (for example 2048), and keep
  `QUANTIZED_KV_CACHE = True` and `FLASH_ATTENTION = True`.
- **CPU baseline:** set `GPU_LAYERS = 0`.
- **More cases:** add lines to `cases.jsonl` with fields `check`, `expected`
  (`yes`/`no`), `text`, and optionally `history` and `context`.

## Notes

- One check is one yes/no call. Checks run on the same message reuse the
  already-read content, so later checks are much cheaper than the first.
- Throughput is measured for a single stream; requests are not batched.
- All test cases are synthetic.
