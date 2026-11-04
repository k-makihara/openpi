# Long input diagnosis

Submit one job first, using existing normalization assets for the selected 224 config:

```bash
cd /fsx/koshi.makihara/openpi
sbatch scripts/slurm_diagnose_train_input.sh --hz 10
# Alternatively, reproduce job 315's 30 Hz input pattern:
# sbatch scripts/slurm_diagnose_train_input.sh --hz 30
```

Defaults: H200 x8, 96 CPUs, batch 512, workers 32, grouped/PyAV, 128 warmup + 1200 measured steps, 5-second monitoring, 12-hour limit. Override `--steps`, `--warmup`, `--interval`. No checkpoint writes or stats recomputation. Existing model cache and normalization assets are required. Input errors fail explicitly instead of silently retrying another sample.

Outputs: `outputs/input_diagnose_JOBID/`:

- `train.log`: initialization and per-step results.
- `events.jsonl`: flushed events before/after each batch fetch, device readiness, and compute. Unix timestamps match resource records, including during an unfinished long fetch.
- `resources.jsonl`: main process and descendants (workers identified by PID/comm/parent), CPU percent (100% = one core), RSS, swap, faults, CPU affinity, wait channel, I/O counters, host CPU/iowait, memory, reclaim/swap counters and pressure. Lustre client stats included where readable.
- `settings.json`: node/job and accounting availability. `exit.json`: child exit status.
- `result.json`, `result.steps.csv`: completed-run timing summary and steps; progressive JSONL survives incomplete runs.

Host counters and process I/O counters are cumulative; compare deltas. RSS includes shared pages and cannot be summed as physical memory. Linux task delay accounting can be disabled; zero I/O delay does not exclude FSx waits. Host pressure is node-wide, not attributed solely to the job. Lustre counters may be unavailable or count other clients on that node.

This reuses the synchronized GPU benchmark, not the asynchronous production training loop. `next_seconds` includes the loader's host/device array construction; `readiness_seconds` measures the subsequent JAX readiness wait. Neither is pure disk I/O. No-checkpoint execution isolates input stalls before checkpoint interference. Failure to reproduce does not prove production is healthy.
