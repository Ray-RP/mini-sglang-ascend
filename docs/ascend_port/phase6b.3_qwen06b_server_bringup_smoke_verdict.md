# Phase 6B.3 Qwen3-0.6B TP=2 Server Bring-up Smoke Verdict

**Kind:** Documentation-only verdict for the Qwen3-0.6B fixed-TP2
server bring-up smoke against the Ascend host. Executed the launch
recipe frozen at Phase 6B.2
([`phase6b.2_server_launch_recipe.md`](./phase6b.2_server_launch_recipe.md)),
observed readiness signals, then terminated the server. No endpoint
was curled; `/generate` and the OpenAI API were not exercised. This
file introduces no runtime, script, test, tag, GitHub Release, or
`CHANGELOG.md` change and does not print credentials.

Envelope: fixed TP=2, eager, `npu_fia`, bf16, greedy — v0.2.0a1.

---

## 1. Environment

* Host: Ascend NPU host (8 × 910B1).
* Container: `998ce5ba6e5e`.
* Repo tree used: `/mnt/nvme/LR-606/mini-sglang-ascend`
  (`ascend-port` at `3263644` — one commit ahead of the branch head
  imported for offline v0.2.0a1 work; server-path source is
  unchanged from the local audit tree).
* Model: `/mnt/nvme/models/Qwen3-0.6B`.
* Working dir: `/mnt/nvme/LR-606/phase6b3/` (launch script + log).
* Python: 3.11.14; `torch_npu` present.
* One preflight fix outside the recipe: `pip install prompt_toolkit`
  (required by `api_server.py` at import time). This is a
  documentation observation, not a scope change.

## 2. Launch invocation (as executed)

```
python -m minisgl.server.launch \
  --model-path /mnt/nvme/models/Qwen3-0.6B \
  --tp-size 2 \
  --attention-backend npu_fia \
  --disable-pynccl \
  --cuda-graph-max-bs 0 \
  --host 0.0.0.0 \
  --port 1919
```

Executed via a wrapper `bash` script under `nohup` inside the
container. `PYTHONPATH=python` set so the in-tree package resolved.

## 3. Observed evidence

### 3.1 Parsed arguments (log)

The parent process printed a `ServerArgs(...)` record confirming
every flag from §2 landed as intended:

```
attention_backend='npu_fia',
cuda_graph_bs=None, cuda_graph_max_bs=0,
use_pynccl=False,
tp_info=DistributedInfo(rank=0, size=2),
dtype=torch.bfloat16,
server_host='0.0.0.0', server_port=1919
```

### 3.2 TP=2 worker processes appear

Process tree observed via `ps --forest` (parent PID 40902):

```
40902 python -m minisgl.server.launch ...
 40911 python -c multiprocessing.resource_tracker.main
 40912 python -c multiprocessing.spawn.spawn_main (--multiprocessing-fork)
 40913 python -c multiprocessing.spawn.spawn_main (--multiprocessing-fork)
 40914 python -c multiprocessing.spawn.spawn_main (--multiprocessing-fork)
```

Three `spawn_main` children = 2 scheduler ranks + 1 detokenizer
(no tokenizer processes because `num_tokenizer=0` is the default).
Both scheduler ranks emitted `core|rank=0` and `core|rank=1` log
lines during model load and KV allocation, confirming
`DistributedInfo(rank=0, size=2)` and `DistributedInfo(rank=1, size=2)`
were dispatched.

### 3.3 NPU device pinning

* `/dev/davinci{0..7}` and `/dev/davinci_manager` all visible in
  the container.
* `npu-smi info` process table at ready time listed the parent's
  worker PIDs against NPU 0 and NPU 1, each with ~44 GiB of HBM
  in use (KV cache = 42.46 GiB per rank per the log):
  ```
  | 0 | ... | 44340 MB |
  | 1 | ... | 44340 MB |
  ```
  All other NPUs (2–7) showed only the ~13 GiB baseline held by
  unrelated container-shared allocations.
* Interpretation: rank 0 pinned to NPU 0, rank 1 pinned to NPU 1.
  The pinning appears to be produced by the scheduler init path
  (not by an explicit `ASCEND_RT_VISIBLE_DEVICES` split in
  `launch.py`); this warrants a follow-up code read but did not
  block bring-up.

### 3.4 Communication backend

* HCCL: two `ProcessGroupHCCL` watchdog-timeout warnings emitted
  by `torch_npu` (`compiler_depend.ts:1065`), one per rank —
  proof the HCCL primary process group was constructed.
* Gloo sidecar: `[Gloo] Rank 0 is connected to 1 peer ranks.` and
  `[Gloo] Rank 1 is connected to 1 peer ranks.` — proof the
  `new_group(backend="gloo")` sidecar handshake completed.
* Total `HCCL`/`Gloo` markers in log: 4.

### 3.5 No CUDA / NCCL / pynccl fallback

* `grep -iE "pynccl|nccl"` on the full log: **zero matches**.
* `grep -iE "cuda"` (excluding the intentional
  `CUDA graph is disabled.` line): **zero matches**.
* `grep -iE "fallback|falling back|traceback|error"`: **zero
  matches**.
* `CUDA graph is disabled.` present exactly **once** —
  `--cuda-graph-max-bs 0` observed by the engine.

### 3.6 Ready / listening state

Log ended (before termination) with:

```
Scheduler is idle, waiting for new reqs...
Scheduler is ready
API server is ready to serve on 0.0.0.0:1919
INFO:     Uvicorn running on http://0.0.0.0:1919 (Press CTRL+C to quit)
```

`netstat -tlnp` confirmed a `LISTEN` socket on `0.0.0.0:1919`
owned by PID 40902 (parent server process).

Wall-clock from parse to ready: ~20 s.

## 4. Termination

* `kill -TERM 40902` (parent).
* Parent Uvicorn shutdown sequence completed cleanly:
  ```
  Shutting down
  Waiting for application shutdown.
  Application shutdown complete.
  Finished server process [40902]
  ```
* Worker children (40911–40914) exited on the same signal path;
  a follow-up SIGTERM sweep across the children required no
  `SIGKILL` fallback.
* Final residual scan (`ps -eo pid,cmd | grep -E "minisgl|multiprocessing.spawn|multiprocessing.resource"`):
  `NO_RESIDUAL_FINAL`.
* Listening port `:1919` released; `netstat`: no match.

## 5. Verdict — per required checklist

| Check | Result |
|---|---|
| server process starts | PASS — parent PID 40902 launched |
| TP=2 worker processes appear | PASS — 2 scheduler ranks + 1 detokenizer observed |
| each rank binds expected NPU device | PASS (empirical) — NPU 0 and NPU 1 held ~44 GiB per PID via `npu-smi`, other NPUs untouched |
| `attention_backend == npu_fia` | PASS — recorded in `ServerArgs` log line |
| `use_pynccl == false` | PASS — `use_pynccl=False` in `ServerArgs` log line |
| `cuda_graph_max_bs == 0` | PASS — `cuda_graph_max_bs=0` in `ServerArgs`; `CUDA graph is disabled.` emitted once |
| `distributed_addr` behaviour observed | PASS — server bound the port-relative TCP rendezvous automatically; no `MINISGL_DISTRIBUTED_ADDR` was set; HCCL + Gloo handshakes both succeeded |
| no CUDA / NCCL / pynccl fallback | PASS — zero `nccl`/`pynccl`/`cuda` matches in log (excluding the deliberate CUDA-graph-disabled notice) |
| server reaches ready / listening state | PASS — `API server is ready to serve on 0.0.0.0:1919` + `Uvicorn running`, `netstat` `LISTEN` on port 1919 |
| server is terminated cleanly | PASS — Uvicorn `Application shutdown complete.` + `Finished server process [40902]` |
| no residual minisgl server processes | PASS — final `ps` scan reported `NO_RESIDUAL_FINAL`, port released |

**Overall verdict: PASS.**

## 6. What this gate does NOT establish

* No HTTP request was issued. `/generate`, `/v1/chat/completions`
  (both `stream=false` and `stream=true`), and `/v1/models` were
  **not** curled — that is Phase 6B.4 (or equivalent).
* Per-rank device-pinning source is inferred from `npu-smi`; the
  code path that produces it was not traced in this gate.
* The `prompt_toolkit` install was outside the recipe. A later
  documentation gate should either fold it into the recipe or
  add it to `pyproject.toml`.
* No performance claim of any kind is made. Bring-up latency and
  memory usage are anecdotal, not benchmarked.
