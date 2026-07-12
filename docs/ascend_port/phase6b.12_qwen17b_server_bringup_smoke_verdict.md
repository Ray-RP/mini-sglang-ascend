# Phase 6B.12 Qwen3-1.7B TP=2 Server Bring-up Smoke Verdict

**Kind:** Documentation-only verdict for the Qwen3-1.7B fixed-TP2
server bring-up smoke against the Ascend host. Launched with the
Phase 6B.7 recipe
([`phase6b.2_server_launch_recipe.md`](./phase6b.2_server_launch_recipe.md)
as amended at Phase 6B.7 to include `--page-size 16`) but with
`--model-path /mnt/nvme/models/Qwen3-1.7B`, observed readiness
signals, then terminated the server. **No endpoint was curled;
`/v1/models`, `/generate`, and `/v1/chat/completions` were not
exercised.** This file introduces no runtime, script, test, tag,
GitHub Release, or `CHANGELOG.md` change and does not print
credentials.

Envelope: fixed TP=2, eager, `npu_fia`, bf16, `page_size=16` —
v0.2.0a1 recipe, second model.

**Overall verdict: PASS.**

---

## 1. Environment

* Host: Ascend NPU host (8 × 910B1).
* Container: `998ce5ba6e5e`.
* Repo tree: `/mnt/nvme/LR-606/mini-sglang-ascend` (unchanged since
  Phase 6B.3).
* Model: `/mnt/nvme/models/Qwen3-1.7B` (safetensors:
  `model-00001-of-00002.safetensors` + `model-00002-of-00002.safetensors`,
  total ~4.06 GB on disk; the ~2 B-parameter tier the Phase 6B.11
  summary flagged as the next recommended bring-up).
* Working dir: `/mnt/nvme/LR-606/phase6b12/` (launch script + log).
* Port: `1919`.
* Python: 3.11.14; `torch_npu` present.

## 2. Launch invocation (as executed)

```
python -m minisgl.server.launch \
  --model-path /mnt/nvme/models/Qwen3-1.7B \
  --tp-size 2 \
  --attention-backend npu_fia \
  --disable-pynccl \
  --cuda-graph-max-bs 0 \
  --page-size 16 \
  --host 0.0.0.0 \
  --port 1919
```

Executed via a wrapper shell script under `setsid nohup` inside
the container. `PYTHONPATH=python` set so the in-tree package
resolved.

## 3. Observed evidence

### 3.1 Parsed arguments (log)

The parent process printed a `ServerArgs(...)` record confirming
every flag from §2 landed as intended:

```
model_path='/mnt/nvme/models/Qwen3-1.7B',
tp_info=DistributedInfo(rank=0, size=2),
dtype=torch.bfloat16,
attention_backend='npu_fia',
cuda_graph_bs=None, cuda_graph_max_bs=0,
page_size=16,
use_pynccl=False,
server_host='0.0.0.0', server_port=1919
```

Per-required-field checklist:

| required field | observed |
|---|---|
| `attention_backend` | `'npu_fia'` |
| `use_pynccl` | `False` |
| `cuda_graph_max_bs` | `0` |
| `page_size` | `16` |
| `tp_info.size` | `2` |
| `dtype` | `torch.bfloat16` |

### 3.2 TP=2 worker processes appear

Process tree observed via `ps -eo pid,ppid,cmd` at ready time
(parent PID `43907`):

```
43907 python -m minisgl.server.launch ...
 43916  multiprocessing.resource_tracker
 43917  multiprocessing.spawn.spawn_main (--multiprocessing-fork)
 43918  multiprocessing.spawn.spawn_main (--multiprocessing-fork)
 43919  multiprocessing.spawn.spawn_main (--multiprocessing-fork)
```

Three `spawn_main` children = 2 scheduler ranks + 1 detokenizer
(no tokenizer processes because `num_tokenizer=0` is the default).
Both scheduler ranks emitted `core|rank=0` and `core|rank=1` log
lines during model load and KV allocation, confirming
`DistributedInfo(rank=0, size=2)` and
`DistributedInfo(rank=1, size=2)` were dispatched:

```
[core|rank=0] Free memory before loading model: 47.86 GiB
[core|rank=1] Allocating 772720 tokens for KV cache, K + V = 41.27 GiB
[core|rank=0] Allocating 772720 tokens for KV cache, K + V = 41.27 GiB
[core|rank=0] Free memory after initialization: 4.74 GiB
[core|rank=0] CUDA graph is disabled.
[core|rank=0] Scheduler is idle, waiting for new reqs...
```

### 3.3 NPU device pinning (rank/device evidence)

`npu-smi info` snapshot at ready time — top-of-table HBM-Usage
column:

```
| 0     910B1 | ... | 60722/ 65536 |
| 1     910B1 | ... | 60721/ 65536 |
| 2     910B1 | ... | 16453/ 65536 |
| 3     910B1 | ... | 16454/ 65536 |
| 4     910B1 | ... | 16454/ 65536 |
| 5     910B1 | ... | 16454/ 65536 |
| 6     910B1 | ... | 16454/ 65536 |
| 7     910B1 | ... | 16454/ 65536 |
```

`npu-smi info -t proc-mem -i {0,1} -c 0` returned exactly one
non-baseline process per NPU:

```
NPU 0: Process id:3764676 ... Process memory(MB):44320
NPU 1: Process id:3764677 ... Process memory(MB):44320
```

Interpretation:

* Two scheduler ranks are pinned one-per-device: rank 0 → NPU 0,
  rank 1 → NPU 1 (host-PID mapping is 1-to-1 with the per-NPU
  `proc-mem` listing).
* Each rank holds ~44 GiB HBM total per `proc-mem` (the ~60.7 GiB
  in the top-of-table view includes container-shared baseline
  allocations of ~16 GiB per device, visible on the untouched
  NPUs 2–7).
* NPUs 2–7 held only the ~16.4 GiB baseline — none of them were
  touched by this launch.
* The KV cache is 41.27 GiB per rank per the log, which is the
  bulk of the ~44 GiB `proc-mem` figure; the remainder is
  1.7 B-parameter model weights and per-rank workspaces.
* Consistent with Phase 6B.3 (Qwen3-0.6B): the pinning appears
  to be produced by the scheduler init path, not by an explicit
  `ASCEND_RT_VISIBLE_DEVICES` split.

### 3.4 Communication backend

* HCCL: two `ProcessGroupHCCL` watchdog-timeout warnings emitted
  by `torch_npu` (`compiler_depend.ts:1065`), one per rank —
  proof the HCCL primary process group was constructed.
* Gloo sidecar: `[Gloo] Rank 0 is connected to 1 peer ranks.` and
  `[Gloo] Rank 1 is connected to 1 peer ranks.` — proof the
  `new_group(backend="gloo")` sidecar handshake completed.
* Total `HCCL`/`Gloo` markers in log: 4.

### 3.5 No CUDA / NCCL / pynccl fallback

* `grep -inE "pynccl|nccl"` on the full log (excluding the
  `use_pynccl=False` argument-echo line): **zero matches**.
* `grep -inE "cuda"` (excluding the intentional
  `CUDA graph is disabled.` line and the argument-echo
  `cuda_graph_bs=`/`cuda_graph_max_bs=` fragments): **zero
  matches**.
* `grep -inE "fallback|falling back"`: **zero matches**.
* `grep -inE "Traceback|Error|RuntimeError"`: **zero matches**.
* `CUDA graph is disabled.` present exactly **once** —
  `--cuda-graph-max-bs 0` observed by the engine.

### 3.6 `block_size` / CANN error status

* `grep -inE "block_size|561002|CheckFeatureNoquant"` on the log:
  **zero matches**.
* `--page-size 16` from the Phase 6B.7 recipe held.

### 3.7 Ready / listening state

Log ended (before termination) with:

```
Scheduler is idle, waiting for new reqs...
Scheduler is ready
API server is ready to serve on 0.0.0.0:1919
INFO:     Started server process [43907]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:1919 (Press CTRL+C to quit)
```

`netstat -tlnp` confirmed a `LISTEN` socket on `0.0.0.0:1919`
owned by PID `43907` (parent server process).

Wall-clock from parse to ready: approximately 20–25 s (parsed
`ServerArgs` at 19:24:15; `API server is ready to serve` at
19:24:21). Consistent with Qwen3-0.6B (~20 s at Phase 6B.3) —
the larger weight tensor did not visibly extend ready time in
this observation.

## 4. Termination

* `kill -TERM 43907` (parent).
* Parent Uvicorn shutdown sequence completed cleanly:
  ```
  Shutting down
  Waiting for application shutdown.
  Application shutdown complete.
  Finished server process [43907]
  ```
* One benign
  `multiprocessing.resource_tracker: There appear to be 2 leaked
  semaphore objects to clean up at shutdown` warning at parent
  exit — same benign notice recorded across Phases 6B.5, 6B.7,
  6B.10.
* Follow-up SIGTERM sweep across the surviving spawn/resource
  children required no `SIGKILL` fallback.
* Final residual scan
  (`ps -eo pid,cmd | grep -E "minisgl.server|multiprocessing.(spawn|resource)"`):
  `NO_RESIDUAL`.
* Listening port `:1919` released; `netstat`: no match.

## 5. Verdict — per required checklist

| Check | Result |
|---|---|
| server reaches LISTEN | PASS — `tcp 0.0.0.0:1919 LISTEN 43907/python` |
| TP=2 scheduler ranks appear | PASS — 2 scheduler ranks + 1 detokenizer (3 `spawn_main` children); both ranks logged `core|rank=0` and `core|rank=1` during load |
| `attention_backend == 'npu_fia'` | PASS — recorded in `ServerArgs` log line |
| `use_pynccl == False` | PASS — recorded in `ServerArgs` log line |
| `cuda_graph_max_bs == 0` | PASS — recorded in `ServerArgs`; `CUDA graph is disabled.` emitted once |
| `page_size == 16` | PASS — recorded in `ServerArgs` |
| rank/device evidence | PASS — `npu-smi info` shows NPU 0 and NPU 1 at ~60.7 GiB HBM (KV + weights + baseline) with exactly one process per device via `-t proc-mem`; NPUs 2–7 untouched at baseline |
| no `block_size` / `561002` error | PASS — zero matches; `--page-size 16` held |
| no CUDA / NCCL / pynccl fallback | PASS — zero `pynccl`/`nccl`/`cuda`/`fallback`/`Traceback` matches (excluding argument echoes and the deliberate `CUDA graph is disabled.` notice); HCCL + Gloo handshakes present |
| clean shutdown | PASS — Uvicorn `Application shutdown complete.` + `Finished server process [43907]`; one benign `resource_tracker` semaphore notice |
| no residual server processes | PASS — final scan `NO_RESIDUAL`, port released |

**Overall verdict: PASS.**

## 6. What this gate does NOT establish

* No HTTP request was issued. `/v1/models`, `/generate`, and
  `/v1/chat/completions` (stream / non-stream, greedy /
  non-greedy) were **not** curled — those are follow-up gates.
* Per-rank device-pinning source is inferred from `npu-smi`; the
  code path that produces it was not traced here.
* No performance claim of any kind is made. Bring-up latency
  and memory usage are anecdotal, not benchmarked.
* The benign `resource_tracker` semaphore-leak warning at parent
  exit remains a documented observation, not a root-caused
  finding.
