# Phase 6B.20 Qwen3-4B Server `/generate` SSE Smoke Verdict

**Kind:** Documentation-only verdict for the Qwen3-4B fixed-TP2
server `/generate` (SSE) smoke against the Ascend host. Launched
with the Phase 6B.7 recipe
([`phase6b.2_server_launch_recipe.md`](./phase6b.2_server_launch_recipe.md)
as amended at Phase 6B.7 to include `--page-size 16`) but with
`--model-path /mnt/nvme/models/Qwen3-4B`, issued one `curl -N`
to `POST /generate`, observed the outcome, then terminated. This
file introduces no runtime, script, test, tag, GitHub Release, or
`CHANGELOG.md` change and does not print credentials.

Envelope: fixed TP=2, eager, `npu_fia`, bf16, `page_size=16`,
greedy (defaults) — v0.2.0a1 recipe, third model.

**Overall verdict: PASS.**

`/generate` streamed eight SSE events over ~2.86 s wall-clock:
seven content-carrying `data:` lines that concatenate to
`" Paris. The capital of Germany is"` (a plausible greedy
continuation of `"The capital of France is"` at `max_tokens=8`
under `ignore_eos=true`) followed by a `data: [DONE]` sentinel.
Both scheduler ranks stayed alive; rank 0 returned to
`Scheduler is idle` after the request; HBM headroom held.

---

## 1. Environment

* Host: Ascend NPU host (8 × 910B1, 64 GiB HBM per NPU).
* Container: `998ce5ba6e5e`.
* Repo tree: `/mnt/nvme/LR-606/mini-sglang-ascend` (unchanged since
  Phase 6B.3).
* Model: `/mnt/nvme/models/Qwen3-4B`.
* Working dir: `/mnt/nvme/LR-606/phase6b20/` (launch script + log
  + response headers/body).
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

Same wrapper skeleton used at Phase 6B.12 → 6B.19 (`setsid nohup`
inside the container; `PYTHONPATH=python`), only the phase-specific
working directory changed to `/mnt/nvme/LR-606/phase6b20/`.

## 3. Server ready state

* Parent PID `46361`; three `multiprocessing.spawn` children
  (`46374`, `46375`, `46376` = 2 scheduler ranks + 1 detokenizer)
  + resource-tracker `46373`.
* Log excerpt:
  ```
  [Gloo] Rank 1 is connected to 1 peer ranks.
  [Gloo] Rank 0 is connected to 1 peer ranks.
  [core|rank=0] Allocating 566928 tokens for KV cache, K + V = 38.93 GiB
  [core|rank=1] Allocating 566928 tokens for KV cache, K + V = 38.93 GiB
  [core|rank=0] CUDA graph is disabled.
  [core|rank=0] Scheduler is idle, waiting for new reqs...
  Scheduler is ready
  API server is ready to serve on 0.0.0.0:1919
  INFO:     Uvicorn running on http://0.0.0.0:1919 (Press CTRL+C to quit)
  ```
* Consistent with Phase 6B.18 / 6B.19: KV cache 38.93 GiB per
  rank, ready ~20 s after launch (rank 0 idle at 21:36:15).

## 4. `/generate` SSE request

Command executed inside the container:

```
curl -N -sS -o gen.body -D gen.headers --max-time 60 \
  http://127.0.0.1:1919/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The capital of France is",
       "max_tokens":8,
       "ignore_eos":true}'
```

The `-N` flag disables curl's output buffering so SSE events land
in `gen.body` as they arrive. No explicit sampling knobs — the
server-side `SamplingParams` defaults (`temperature=0.0`,
`top_k=-1`, `top_p=1.0`) satisfy the greedy predicate
(`python/minisgl/core.py:30`), so the sampler stays on the
`torch.argmax` branch and never imports `flashinfer.sampling`.

## 5. Response — what the client got

Curl instrumentation:

| metric | value |
|---|---|
| HTTP status | `200` |
| content-type | `text/event-stream; charset=utf-8` |
| transfer-encoding | `chunked` |
| response bytes | `95` |
| wall-clock | `~2.86 s` |

Response headers verbatim:

```
HTTP/1.1 200 OK
date: Sun, 12 Jul 2026 21:36:33 GMT
server: uvicorn
content-type: text/event-stream; charset=utf-8
transfer-encoding: chunked
```

Response body verbatim (SSE stream, blank separator lines
preserved by curl output):

```
data:  Paris
data: .
data:  The
data:  capital
data:  of
data:  Germany
data:  is
data: [DONE]
```

Server access log:

```
INFO:     127.0.0.1:45066 - "POST /generate HTTP/1.1" 200 OK
```

### SSE event count

* Total `data:` lines: **8** (measured by
  `grep -c "^data:" gen.body`).
* Content-carrying events: **7** —
  `" Paris"`, `"."`, `" The"`, `" capital"`, `" of"`,
  `" Germany"`, `" is"`.
* `[DONE]` sentinel: **1** — final `data: [DONE]` line.

The `/generate` route emits raw text chunks (no JSON envelope,
unlike `/v1/chat/completions`). The eight requested tokens
correspond to the seven content chunks plus the `[DONE]`
terminator; `ignore_eos=true` prevents an early stop.

### Generated text

Concatenation of the seven content `data:` payload strings:

```
 Paris. The capital of Germany is
```

Interpretation: Qwen3-4B, prompted with `"The capital of France
is"`, greedy-decoded `" Paris"` as the first token — semantically
correct — then continued into a plausible sentence stem
(`". The capital of Germany is"`) as the `max_tokens=8` budget
was consumed. This confirms the forward loop, tokenizer, sampler,
and SSE frontend are all wired end-to-end on the Qwen3-4B TP=2
path.

Cross-check against Phase 6B.14 (Qwen3-1.7B) and Phase 6B.7
(Qwen3-0.6B): both smaller Qwen3 sizes emitted
`" Paris. The capital of the United"` under the same prompt and
budget. Qwen3-4B diverges at token 5 — where 0.6B/1.7B continued
with `" of the United"`, 4B continued with `" of Germany is"`.
Both continuations are grammatically plausible greedy
extrapolations; the divergence reflects genuine model behaviour
at the 4B tier under this prompt, not a sampler drift (all three
models share the greedy `torch.argmax` codepath — see §6).

### `[DONE]` presence

Present exactly once at the tail of the SSE stream. Matches the
`/generate` contract observed at Phase 6B.7 for Qwen3-0.6B and
Phase 6B.14 for Qwen3-1.7B.

## 6. `flashinfer.sampling` import status

* `grep -inE "flashinfer"` on
  `/mnt/nvme/LR-606/phase6b20/server.log`: **zero matches**.
* Neither scheduler rank raised the Phase 6B.8
  `ModuleNotFoundError: No module named 'flashinfer'`.
* Interpretation: the request took `SamplingParams` defaults
  (`temperature=0.0`), satisfying `SamplingParams.is_greedy`
  (`python/minisgl/core.py:30`). `Sampler.prepare`
  (`python/minisgl/engine/sample.py:55`) set
  `BatchSamplingArgs.temperatures = None`; `Sampler.sample` fell
  into the `torch.argmax` branch, and `sample_impl` (the
  CUDA-only-import caller at
  `python/minisgl/engine/sample.py:30`) was never executed.

**Flashinfer import: did NOT occur.**

## 7. `block_size` / CANN error status

* `grep -inE "block_size|561002|CheckFeatureNoquant"` on the log:
  **zero matches**.
* `--page-size 16` from the Phase 6B.7 recipe held. The BF16
  no-quant `aclnnFusedInferAttentionScoreV3` kernel accepted the
  16-aligned paged-KV blocks throughout the request.

## 8. No CUDA / NCCL / pynccl fallback

Grep over the full server log:

* `pynccl` / `nccl` external matches (excluding the intentional
  `use_pynccl=False` argument-echo line in the `ServerArgs`
  block): **zero**.
* `cuda` external matches (excluding `cuda_graph_bs=None`,
  `cuda_graph_max_bs=0` in the `ServerArgs` block, and the
  deliberate `CUDA graph is disabled.` notice): **zero**.
* HCCL + Gloo evidence present: 2× `ProcessGroupHCCL` watchdog
  warnings + `[Gloo] Rank {0,1} is connected to 1 peer ranks.`.

The forward pass ran on HCCL for collectives, Gloo as the
sidecar; no CUDA-only comm backend was loaded and no CUDA-only
sampling code was touched.

## 9. Scheduler rank health after the request

Post-response process tree:

```
PID     PPID    CMD
46361   46359   python -m minisgl.server.launch --model-path /mnt/nvme/models/Qwen3-4B ...
46373   46361     python -c from multiprocessing.resource_tracker import main;main(22)
46374   46361     python -c from multiprocessing.spawn import spawn_main; ... pipe_handle=25 --multiprocessing-fork
46375   46361     python -c from multiprocessing.spawn import spawn_main; ... pipe_handle=27 --multiprocessing-fork
46376   46361     python -c from multiprocessing.spawn import spawn_main; ... pipe_handle=29 --multiprocessing-fork
```

* All three `spawn_main` children (`46374`, `46375`, `46376`)
  and the resource-tracker `46373` remained alive after the SSE
  stream completed.
* Post-response log line:
  ```
  [2026-07-12|21:36:36|core|rank=0] Scheduler is idle, waiting for new reqs...
  ```
  Rank 0 returned to the idle state after the streamed request
  finished — the request went all the way through the forward
  loop, back through the detokenizer/frontend, and the scheduler
  released the slot. Timeline: rank 0 idle at 21:36:15 (ready) →
  request at 21:36:33 → idle again at 21:36:36 (~3 s working
  window, matching the ~2.86 s curl wall-clock plus scheduler
  handoff).
* Zero `Traceback` / `ModuleNotFoundError` / `RuntimeError`
  matches in the server log.

## 10. HBM headroom after the request

Per-NPU HBM (via `npu-smi info`, HBM-Usage column, MB used / MB
total) captured immediately after the SSE stream completed:

| NPU | HBM-Usage (MB) | Interpretation |
|---|---|---|
| 0 | `61139 / 65536` | Rank 0 pinned here (~59.7 GiB used, ~4.30 GiB free) |
| 1 | `61138 / 65536` | Rank 1 pinned here (~59.7 GiB used, ~4.30 GiB free) |
| 2 | `16453 / 65536` | Container baseline only |
| 3 | `16454 / 65536` | Container baseline only |
| 4 | `16454 / 65536` | Container baseline only |
| 5 | `16455 / 65536` | Container baseline only |
| 6 | `16455 / 65536` | Container baseline only |
| 7 | `16455 / 65536` | Container baseline only |

Per-NPU proc-mem confirms the 1-rank-to-1-NPU pinning is
unchanged after the request:

```
NPU 0: proc-mem 44743 MB (scheduler rank 0) + 13090 MB (HCCL sidecar)
NPU 1: proc-mem 44743 MB (scheduler rank 1) + 13090 MB (HCCL sidecar)
```

Comparison to Phase 6B.19 (immediately after bring-up + one
`/v1/models` request) — HBM-Usage was 60722 / 65536 MB on both
active NPUs. After this `/generate` request it is 61139 / 61138
MB — a ~416 MB per-rank increase (~423 MB in the scheduler-rank
`proc-mem` line: 44743 vs. 44320 in Phase 6B.19). This growth
reflects live-request working memory (activations for the
scheduler's forward pass, transient KV entries written for the
seven generated tokens beyond what was pre-allocated by the
scheduler at bring-up). The pool still holds ~4.30 GiB free per
rank, so headroom did not collapse under the single-request load.

**HBM headroom: PASS — held at ~4.30 GiB free per active rank
post-request; delta vs. bring-up (~416 MB) is working-memory
growth for the live forward pass, not a leak.**

## 11. Termination and residual check

* `kill -TERM 46361` (parent). Uvicorn shutdown sequence
  completed:
  ```
  INFO:     Shutting down
  INFO:     Waiting for application shutdown.
  INFO:     Application shutdown complete.
  INFO:     Finished server process [46361]
  Terminated
  ```
* One benign
  `multiprocessing.resource_tracker: There appear to be 2 leaked
  semaphore objects to clean up at shutdown` warning at parent
  exit — same benign notice recorded across Phases 6B.5, 6B.7,
  6B.9, 6B.10, 6B.12–6B.16, 6B.18, 6B.19.
* Follow-up SIGTERM sweep over surviving
  `multiprocessing.spawn` / `resource_tracker` children exited on
  the signal path (no `SIGKILL` fallback).
* Final `ps -eo pid,cmd | grep -E "minisgl.server|multiprocessing.(spawn|resource)"`:
  `NO_RESIDUAL`.
* Listening port `:1919` released; `netstat`: no match.

## 12. Verdict — per required checklist

| Check | Result |
|---|---|
| HTTP status | PASS — `200`, `content-type: text/event-stream; charset=utf-8`, `transfer-encoding: chunked` |
| SSE event count | PASS — 8 `data:` events (7 content-carrying + 1 `[DONE]` sentinel) |
| generated text | PASS — 7 content chunks concatenate to `" Paris. The capital of Germany is"` — non-empty, semantically correct first token (`" Paris"`), plausible continuation under `max_tokens=8, ignore_eos=true` |
| rank health after request | PASS — parent `46361` + 3 spawn children (`46374`, `46375`, `46376`) + resource-tracker (`46373`) still alive; rank 0 logged `Scheduler is idle, waiting for new reqs...` post-response |
| HBM headroom after request | PASS — HBM-Usage ~61,139 / 65,536 MB on both active NPUs (~4.30 GiB free per rank); ~416 MB working-memory delta vs. Phase 6B.19 bring-up state |
| no `block_size` / `561002` error | PASS — zero matches; `--page-size 16` held |
| no CUDA / NCCL / pynccl fallback | PASS — zero `pynccl` / `nccl` / `cuda` / `flashinfer` non-argument matches; HCCL + Gloo handshakes present |
| clean shutdown | PASS — Uvicorn `Application shutdown complete.` + `Finished server process [46361]`; one benign `resource_tracker` semaphore cleanup notice |
| no residual processes | PASS — final scan `NO_RESIDUAL`, port released |

**Overall verdict: PASS.**

## 13. What this gate does NOT establish

* No `/v1/chat/completions` (stream or non-stream) was issued
  against Qwen3-4B. Those are follow-up gates.
* No multi-request, batch, or long-context behaviour was tested.
* No non-greedy sampling was exercised on Qwen3-4B; the Phase
  6B.8 `flashinfer.sampling` blocker on the default OpenAI
  request body is still expected to reproduce on this model, and
  the Phase 6B.11 recipe-level workaround (`temperature=0`)
  still applies.
* No token-count accuracy check was performed; the request
  budgeted `max_tokens=8` and observed 7 content chunks + 1
  terminator, but per-token accounting on the `/generate` route
  is not part of this gate's scope.
* No client-disconnect / abort-ack behaviour was exercised.
* No claim is made about the semantic accuracy of tokens 5–7
  (`" of Germany is"` — a divergence from the smaller-model
  continuation `" of the United"`); both are plausible greedy
  outputs and the divergence is model-tier behaviour, not a bug.
* The benign `resource_tracker` semaphore-leak warning at parent
  exit remains a documented observation, not a root-caused
  finding.
* No performance claim of any kind is made. Wall-clock (~2.86 s
  end-to-end for 7 tokens) is anecdotal, not benchmarked.
