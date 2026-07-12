# Phase 6B.18 Qwen3-4B Server Bring-Up Smoke Verdict

**Kind:** Documentation-only verdict for the Qwen3-4B fixed-TP2
server bring-up smoke against the Ascend host. Launched with the
Phase 6B.7 recipe
([`phase6b.2_server_launch_recipe.md`](./phase6b.2_server_launch_recipe.md)
as amended at Phase 6B.7 to include `--page-size 16`) but with
`--model-path /mnt/nvme/models/Qwen3-4B`, waited for
`API server is ready to serve`, captured the ready-state
evidence, then terminated. This file introduces no runtime,
script, test, tag, GitHub Release, or `CHANGELOG.md` change; it
issues no `curl`, hits no endpoint, and does not print
credentials. Its sole purpose is to prove the same converged
launch recipe brings a third model (Qwen3-4B) up cleanly on the
Ascend host.

Envelope: fixed TP=2, eager, `npu_fia`, bf16, `page_size=16` —
v0.2.0a1 recipe, third model.

**Overall verdict: PASS.**

Route: the Phase 6B.7 launch recipe brought Qwen3-4B up on the
Ascend host with no code change beyond `--model-path`. Both
scheduler ranks reached idle, the FastAPI frontend bound
`0.0.0.0:1919`, and HBM headroom on the two active NPUs held at
~4.70 GiB free per rank — the same headroom envelope observed on
Qwen3-0.6B and Qwen3-1.7B, i.e. the 4B tier fit inside the
64 GiB HBM budget without a recipe amendment. Uvicorn shut down
cleanly on SIGTERM; the final residual scan reported
`NO_RESIDUAL` and port `:1919` released.

---

## 1. Environment

* Host: Ascend NPU host (8 × 910B1, 64 GiB HBM per NPU).
* Container: `998ce5ba6e5e`.
* Repo tree: `/mnt/nvme/LR-606/mini-sglang-ascend` (unchanged since
  Phase 6B.3).
* Model: `/mnt/nvme/models/Qwen3-4B`.
* Working dir: `/mnt/nvme/LR-606/phase6b18/` (launch script + log).
* Port: `1919`.

## 2. Launch invocation (as executed)

```
python -m minisgl.server.launch \
  --model-path /mnt/nvme/models/Qwen3-4B \
  --tp-size 2 \
  --attention-backend npu_fia \
  --disable-pynccl \
  --cuda-graph-max-bs 0 \
  --page-size 16 \
  --host 0.0.0.0 \
  --port 1919
```

Same wrapper skeleton used at Phase 6B.12 → 6B.16 (`setsid nohup`
inside the container; `PYTHONPATH=python`), only the phase-specific
working directory changed to `/mnt/nvme/LR-606/phase6b18/` and the
model path swapped from Qwen3-1.7B to Qwen3-4B.

Parsed `ServerArgs` log line confirmed every intended flag landed:

```
ServerArgs(model_path='/mnt/nvme/models/Qwen3-4B',
           tp_info=DistributedInfo(rank=0, size=2),
           dtype=torch.bfloat16,
           max_running_req=256,
           attention_backend='npu_fia',
           moe_backend='auto',
           cuda_graph_bs=None, cuda_graph_max_bs=0,
           page_size=16,
           memory_ratio=0.9,
           distributed_timeout=60.0,
           use_dummy_weight=False,
           use_pynccl=False,
           max_seq_len_override=None,
           num_page_override=None,
           max_extend_tokens=8192,
           cache_type='radix',
           offline_mode=False,
           _unique_suffix='.pid=45873',
           server_host='0.0.0.0', server_port=1919,
           num_tokenizer=0, silent_output=False)
```

## 3. Server ready state

Ready log excerpt (from `/mnt/nvme/LR-606/phase6b18/server.log`):

```
[Gloo] Rank 0 is connected to 1 peer ranks. Expected number of connected peer ranks is : 1
[Gloo] Rank 1 is connected to 1 peer ranks. Expected number of connected peer ranks is : 1
[core|rank=0] Free memory before loading model: 47.86 GiB
Loading weights:   0%|          | 0/3 [00:00<?, ?it/s]
Loading weights: 100%|██████████| 3/3 [00:02<00:00,  1.38it/s]
[core|rank=0] Allocating 566928 tokens for KV cache, K + V = 38.93 GiB
[core|rank=1] Allocating 566928 tokens for KV cache, K + V = 38.93 GiB
[core|rank=0] Free memory after initialization: 4.74 GiB
[core|rank=0] CUDA graph is disabled.
[core|rank=0] Scheduler is idle, waiting for new reqs...
Scheduler is ready
API server is ready to serve on 0.0.0.0:1919
INFO:     Uvicorn running on http://0.0.0.0:1919 (Press CTRL+C to quit)
```

Timeline:

* `20:20:32` — parse of `ServerArgs` (parent enters main).
* `20:20:46` — tokenize server ready.
* `20:20:47` — HCCL watchdog warnings + `[Gloo] Rank {0,1} is
  connected to 1 peer ranks.` — the same HCCL + Gloo handshake
  signature as Phase 6B.12–6B.16.
* `20:20:47` — `Free memory before loading model: 47.86 GiB`
  (per rank).
* `20:20:48–20:20:50` — weight load in 3 shards, ~2 s wall-clock
  (matches the 1.09 s/it → 1.06 s/it → 1.38 it/s progress bar).
* `20:20:50` — KV cache allocation on both ranks (see §5).
* `20:20:51` — `Free memory after initialization: 4.74 GiB`
  (per rank).
* `20:20:52` — scheduler idle + API server ready + Uvicorn
  running. Ready wall-clock ~20 s from parse, same as
  Qwen3-1.7B / Qwen3-0.6B.

Process tree (captured immediately after ready):

```
PID     PPID    CMD
45873   45871   python -m minisgl.server.launch --model-path /mnt/nvme/models/Qwen3-4B ...
45882   45873     python -c from multiprocessing.resource_tracker import main;main(22)
45883   45873     python -c from multiprocessing.spawn import spawn_main; ... pipe_handle=25 --multiprocessing-fork
45884   45873     python -c from multiprocessing.spawn import spawn_main; ... pipe_handle=27 --multiprocessing-fork
45885   45873     python -c from multiprocessing.spawn import spawn_main; ... pipe_handle=29 --multiprocessing-fork
```

* Parent PID `45873`.
* Resource-tracker `45882`.
* Three `multiprocessing.spawn` children: `45883`, `45884`, `45885`
  = 2 scheduler ranks + 1 detokenizer.

`netstat -tnlp` confirmed listener:

```
tcp   0   0   0.0.0.0:1919   0.0.0.0:*   LISTEN   45873/python
```

## 4. Rank / device evidence

Per-NPU HBM (via `npu-smi info`, HBM-Usage column, MB used / MB
total):

| NPU | HBM-Usage (MB) | Interpretation |
|---|---|---|
| 0 | `60723 / 65536` | Rank 0 pinned here (~59.3 GiB used, ~4.70 GiB free) |
| 1 | `60722 / 65536` | Rank 1 pinned here (~59.3 GiB used, ~4.70 GiB free) |
| 2 | `16453 / 65536` | Container baseline only |
| 3 | `16454 / 65536` | Container baseline only |
| 4 | `16453 / 65536` | Container baseline only |
| 5 | `16454 / 65536` | Container baseline only |
| 6 | `16455 / 65536` | Container baseline only |
| 7 | `16454 / 65536` | Container baseline only |

Per-NPU proc-mem confirms the 1-rank-to-1-NPU pinning:

```
npu-smi info -t proc-mem -i 0 -c 0
  Process id:749805  Process memory(MB):44320
  Process id:1067985 Process memory(MB):13090

npu-smi info -t proc-mem -i 1 -c 0
  Process id:1067986 Process memory(MB):13090
  Process id:749806  Process memory(MB):44320
```

* NPU 0: one scheduler-rank process (~44.3 GiB `proc-mem`) plus
  one HCCL sidecar process (~13.1 GiB).
* NPU 1: mirror-image — one scheduler-rank process (~44.3 GiB)
  plus one HCCL sidecar.
* NPUs 2–7: at the container baseline (~16.1 GiB, unchanged from
  Phase 6B.12–6B.16).

TP=2 fan-out therefore held: two ranks, two NPUs, no cross-device
leakage.

## 5. HBM headroom (watch item from Phase 6B.17 §6)

| metric | Qwen3-0.6B / 1.7B | Qwen3-4B (this gate) |
|---|---|---|
| Free memory before loading model, per rank | 47.86 GiB | **47.86 GiB** (unchanged — container baseline) |
| KV cache tokens allocated | 772,720 | **566,928** |
| KV cache size per rank | 41.27 GiB | **38.93 GiB** |
| `proc-mem` per rank (npu-smi) | ~44 GiB | **~44.3 GiB** |
| Free memory after initialization, per rank | 4.74 GiB | **4.74 GiB** |
| HBM-Usage per active NPU (npu-smi top table) | ~60,724 MB / 65,536 MB | **~60,723 MB / 65,536 MB** |

Interpretation:

* The 4B weight tensor consumed the extra HBM that would
  otherwise have gone to the KV cache. The heuristic that
  bounds `Free memory after initialization` to ~4.74 GiB
  (driven by `memory_ratio=0.9`, see the `ServerArgs` block in
  §2) held: the scheduler simply allocated fewer KV-cache tokens
  (566,928 vs. 772,720 on the 1.7B tier) to keep the same headroom
  ceiling.
* Total HBM-Usage per active NPU is ~60.3 GiB of 64 GiB — the same
  ~92% pressure observed on the 1.7B tier. The
  Phase 6B.17 §6 concern that "64 GiB HBM per NPU may become a
  live constraint" at the 4B tier did **not** materialize as a
  live failure; the scheduler's KV-cache autosizer absorbed the
  weight-tensor delta.
* No `--memory-ratio` amendment or KV-cache-cap flag was needed;
  the Phase 6B.7 recipe survived the 4B tier as-is.

## 6. `block_size` / CANN error status

* `grep -inE "block_size|561002|CheckFeatureNoquant"` on the log:
  **zero matches**.
* `--page-size 16` from the Phase 6B.7 recipe held. The BF16
  no-quant `aclnnFusedInferAttentionScoreV3` kernel accepted the
  16-aligned paged-KV blocks during KV allocation.

## 7. No CUDA / NCCL / pynccl fallback

Grep over the full server log:

* `pynccl` / `nccl` external matches (excluding the intentional
  `use_pynccl=False` argument-echo line in the `ServerArgs`
  block): **zero**.
* `cuda` external matches (excluding `cuda_graph_bs=None`,
  `cuda_graph_max_bs=0` in the `ServerArgs` block, and the
  deliberate `CUDA graph is disabled.` notice): **zero**.
* `flashinfer` matches: **zero** (expected — bring-up alone does
  not enter the sampler).
* HCCL + Gloo evidence present: 2× `ProcessGroupHCCL` watchdog
  warnings + `[Gloo] Rank {0,1} is connected to 1 peer ranks.`.

The bring-up path ran on HCCL for collectives, Gloo as the
sidecar; no CUDA-only comm backend was loaded and no CUDA-only
sampling code was touched.

## 8. Scheduler rank health after ready

* All three `spawn_main` children (`45883`, `45884`, `45885`) and
  the resource-tracker `45882` alive at ready.
* Post-ready log line:
  ```
  [2026-07-12|20:20:52|core|rank=0] Scheduler is idle, waiting for new reqs...
  ```
  Rank 0 sat idle waiting for requests — no forward pass was
  triggered in this gate (no `curl`, no endpoint hit) but the
  scheduler loop was demonstrably running.
* Zero `Traceback` / `ModuleNotFoundError` / `RuntimeError`
  matches in the server log.

## 9. Termination and residual check

* `kill -TERM 45873` (parent). Uvicorn shutdown sequence
  completed:
  ```
  INFO:     Shutting down
  INFO:     Waiting for application shutdown.
  INFO:     Application shutdown complete.
  INFO:     Finished server process [45873]
  Terminated
  ```
* One benign
  `multiprocessing.resource_tracker: There appear to be 2 leaked
  semaphore objects to clean up at shutdown` warning at parent
  exit — same benign notice recorded across Phases 6B.5, 6B.7,
  6B.9, 6B.10, 6B.12, 6B.13, 6B.14, 6B.15, 6B.16.
* Follow-up SIGTERM sweep over surviving
  `multiprocessing.spawn` / `resource_tracker` children (`45882`,
  `45883`, `45884`, `45885`) exited on the signal path (no
  `SIGKILL` fallback).
* Final `ps -eo pid,cmd | grep -E "minisgl.server|multiprocessing.(spawn|resource)"`:
  `NO_RESIDUAL`.
* Listening port `:1919` released; `netstat`: no match.

## 10. Verdict — per required checklist

| Check | Result |
|---|---|
| server LISTEN on `0.0.0.0:1919` | PASS — `tcp 0.0.0.0:1919 LISTEN 45873/python`; `API server is ready to serve on 0.0.0.0:1919`; `Uvicorn running on http://0.0.0.0:1919` |
| TP=2 rank fan-out | PASS — `ServerArgs.tp_info=DistributedInfo(rank=0, size=2)`, 3 `multiprocessing.spawn` children (2 scheduler ranks + 1 detokenizer), `[Gloo] Rank {0,1} is connected to 1 peer ranks.` |
| `ServerArgs` reflects recipe | PASS — every flag confirmed (`model_path='/mnt/nvme/models/Qwen3-4B'`, `attention_backend='npu_fia'`, `use_pynccl=False`, `cuda_graph_max_bs=0`, `page_size=16`, `dtype=torch.bfloat16`) |
| rank / device pinning | PASS — NPU 0 rank 0 (~44.3 GiB `proc-mem`), NPU 1 rank 1 (~44.3 GiB `proc-mem`); NPUs 2–7 at container baseline (~16.1 GiB) |
| HBM headroom | PASS — 4.74 GiB free per rank after init (same envelope as Qwen3-0.6B / 1.7B); KV cache autosized to 38.93 GiB per rank; no `--memory-ratio` amendment needed |
| no `block_size` / `561002` error | PASS — zero matches; `--page-size 16` held for BF16 no-quant path |
| no CUDA / NCCL / pynccl fallback | PASS — zero `pynccl`/`nccl`/`cuda`/`flashinfer` non-argument matches; HCCL + Gloo handshakes present |
| scheduler idle after init | PASS — `[core|rank=0] Scheduler is idle, waiting for new reqs...` before ready |
| clean shutdown | PASS — Uvicorn `Application shutdown complete.` + `Finished server process [45873]`; one benign `resource_tracker` semaphore cleanup notice |
| no residual processes | PASS — final scan `NO_RESIDUAL`, port released |

**Overall verdict: PASS.**

## 11. What this gate does NOT establish

* No endpoint was exercised. `/v1/models`, `/generate`, and
  `/v1/chat/completions` (stream and non-stream) against
  Qwen3-4B are follow-up gates.
* No inference-path validation: no forward pass was triggered on
  the launched server, so this gate does not exercise Qwen3-4B's
  sampler, tokenizer, detokenizer, or SSE frontend. The
  Phase 6B.11 §3 recipe-level constraints (model field must equal
  absolute path, `page_size=16`, `temperature=0` to avoid
  `flashinfer.sampling`, `usage` counters unpopulated) are
  expected to carry over to Qwen3-4B on the same grounds as
  Qwen3-1.7B, but they have not been experimentally re-verified
  on this model.
* No non-greedy sampling was exercised. The Phase 6B.8
  `flashinfer.sampling` blocker is device-level, not
  model-specific, and is expected to reproduce on Qwen3-4B if
  the request body drops `temperature=0`.
* No performance claim of any kind is made. The ~20 s
  parse-to-ready wall-clock is anecdotal and unchanged from the
  Qwen3-1.7B tier despite the larger weight load (~2 s of the
  20 s window; the balance is init, HCCL handshake, and KV
  allocation, none of which scale with model size at TP=2 on
  this envelope).
* No multi-request, batch, or long-context behaviour was tested.
* The benign `resource_tracker` semaphore-leak warning at parent
  exit remains a documented observation, not a root-caused
  finding.
* Whether the 4B tier's ~92% HBM pressure leaves enough headroom
  for a live long-context request (KV cache would grow beyond
  its pre-allocated pool for a single very long sequence) is
  not covered by this gate; the pre-allocated 566,928-token pool
  is the operational bound.
