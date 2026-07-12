# Phase 6B.14 Qwen3-1.7B Server `/generate` SSE Smoke Verdict

**Kind:** Documentation-only verdict for the Qwen3-1.7B fixed-TP2
server `/generate` (SSE) smoke against the Ascend host. Launched
with the Phase 6B.7 recipe
([`phase6b.2_server_launch_recipe.md`](./phase6b.2_server_launch_recipe.md)
as amended at Phase 6B.7 to include `--page-size 16`) but with
`--model-path /mnt/nvme/models/Qwen3-1.7B`, issued one `curl -N`
to `POST /generate`, observed the outcome, then terminated. This
file introduces no runtime, script, test, tag, GitHub Release, or
`CHANGELOG.md` change and does not print credentials.

Envelope: fixed TP=2, eager, `npu_fia`, bf16, `page_size=16`,
greedy (defaults) — v0.2.0a1 recipe, second model.

**Overall verdict: PASS.**

`/generate` streamed eight SSE events over ~2.65 s wall-clock:
seven content-carrying `data:` lines that concatenate to
`" Paris. The capital of the United"` (a plausible greedy
continuation of `"The capital of France is"` at `max_tokens=8`
under `ignore_eos=true`) followed by a `data: [DONE]` sentinel.
Both scheduler ranks stayed alive; rank 0 returned to
`Scheduler is idle` after the request.

---

## 1. Environment

* Host: Ascend NPU host (8 × 910B1).
* Container: `998ce5ba6e5e`.
* Repo tree: `/mnt/nvme/LR-606/mini-sglang-ascend` (unchanged since
  Phase 6B.3).
* Model: `/mnt/nvme/models/Qwen3-1.7B`.
* Working dir: `/mnt/nvme/LR-606/phase6b14/` (launch script + log
  + response headers/body).
* Port: `1919`.

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

Parsed `ServerArgs` log line confirmed every intended flag landed:

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

## 3. Server ready state

* Parent PID `44261`; three `multiprocessing.spawn` children
  (`44271`, `44272`, `44273` = 2 scheduler ranks + 1 detokenizer)
  + resource-tracker `44270`.
* Log excerpt:
  ```
  [Gloo] Rank 0 is connected to 1 peer ranks.
  [Gloo] Rank 1 is connected to 1 peer ranks.
  [core|rank=0] Allocating 772720 tokens for KV cache, K + V = 41.27 GiB
  [core|rank=1] Allocating 772720 tokens for KV cache, K + V = 41.27 GiB
  [core|rank=0] CUDA graph is disabled.
  [core|rank=0] Scheduler is idle, waiting for new reqs...
  Scheduler is ready
  API server is ready to serve on 0.0.0.0:1919
  INFO:     Uvicorn running on http://0.0.0.0:1919 (Press CTRL+C to quit)
  ```
* `netstat`: `tcp 0.0.0.0:1919 LISTEN 44261/python`.
* Consistent with Phase 6B.12 / 6B.13: KV cache 41.27 GiB per
  rank, ready ~20 s after launch (ready at 19:43:39).

## 4. `/generate` SSE request

Command executed inside the container:

```
curl -N -sS -o gen.body -D gen.headers \
  --max-time 60 \
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
| wall-clock | `~2.65 s` |

Response headers verbatim:

```
HTTP/1.1 200 OK
date: Sun, 12 Jul 2026 19:44:36 GMT
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
data:  the
data:  United
data: [DONE]
```

Server access log:

```
INFO:     127.0.0.1:46580 - "POST /generate HTTP/1.1" 200 OK
```

### SSE event count

* Total `data:` lines: **8** (measured by
  `grep -c "^data: " gen.body`).
* Content-carrying events: **7** —
  `" Paris"`, `"."`, `" The"`, `" capital"`, `" of"`, `" the"`,
  `" United"`.
* `[DONE]` sentinel: **1** — final `data: [DONE]` line.

The `/generate` route emits raw text chunks (no JSON envelope,
unlike `/v1/chat/completions` at Phase 6B.10). The eight
requested tokens correspond to the seven content chunks plus the
`[DONE]` terminator; `ignore_eos=true` prevents an early stop.

### Generated text

Concatenation of the seven content `data:` payload strings:

```
 Paris. The capital of the United
```

Interpretation: Qwen3-1.7B, prompted with `"The capital of France
is"`, greedy-decoded `" Paris"` (a leading-space token in the
tokenizer's vocabulary) as the first token, then continued with a
plausible sentence stem (`". The capital of the United"`) as the
`max_tokens=8` budget was consumed. This is a semantically
correct completion — the first token is the ground-truth answer
— and confirms the forward loop, tokenizer, sampler, and SSE
frontend are all wired end-to-end on the Qwen3-1.7B TP=2 path.

### `[DONE]` presence

Present exactly once at the tail of the SSE stream. Matches the
`/generate` contract observed at Phase 6B.7 for Qwen3-0.6B.

## 6. `flashinfer.sampling` import status

* `grep -inE "flashinfer"` on
  `/mnt/nvme/LR-606/phase6b14/server.log`: **zero matches**.
* Neither scheduler rank raised the Phase 6B.8
  `ModuleNotFoundError: No module named 'flashinfer'`.
* Interpretation: the request took `SamplingParams` defaults
  (`temperature=0.0`), which satisfies
  `SamplingParams.is_greedy` (`python/minisgl/core.py:30`).
  `Sampler.prepare` (`python/minisgl/engine/sample.py:55`) set
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
  `use_pynccl=False` argument-echo line): **zero**.
* `cuda` external matches (excluding `cuda_graph_bs=`,
  `cuda_graph_max_bs=`, and the deliberate
  `CUDA graph is disabled.` notice): **zero**.
* HCCL + Gloo evidence present: 2× `ProcessGroupHCCL` watchdog
  warnings + `[Gloo] Rank {0,1} is connected to 1 peer ranks.`.

The forward pass ran on HCCL for collectives, Gloo as the
sidecar; no CUDA-only comm backend was loaded and no CUDA-only
sampling code was touched.

## 9. Scheduler rank health after the request

* All three `spawn_main` children (`44271`, `44272`, `44273`)
  and the resource-tracker `44270` remained alive after the SSE
  stream completed.
* Post-response log line:
  ```
  [2026-07-12|19:44:39|core|rank=0] Scheduler is idle, waiting for new reqs...
  ```
  This is proof rank 0 returned to the idle state after the
  streamed request finished — the request went all the way
  through the forward loop, back through the detokenizer/frontend,
  and the scheduler released the slot.
* Zero `Traceback` / `ModuleNotFoundError` / `RuntimeError`
  matches in the server log.

## 10. Termination and residual check

* `kill -TERM 44261` (parent). Uvicorn shutdown sequence
  completed:
  ```
  INFO:     Shutting down
  INFO:     Waiting for application shutdown.
  INFO:     Application shutdown complete.
  INFO:     Finished server process [44261]
  ```
* One benign
  `multiprocessing.resource_tracker: There appear to be 2 leaked
  semaphore objects to clean up at shutdown` warning at parent
  exit — same benign notice recorded across Phases 6B.5, 6B.7,
  6B.9, 6B.10, 6B.12, 6B.13.
* Follow-up SIGTERM sweep over surviving
  `multiprocessing.spawn` / `resource_tracker` children (`44270`,
  `44271`, `44272`, `44273`) exited on the signal path (no
  `SIGKILL` fallback).
* Final `ps -eo pid,cmd | grep -E "minisgl.server|multiprocessing.(spawn|resource)"`:
  `NO_RESIDUAL`.
* Listening port `:1919` released; `netstat`: no match.

## 11. Verdict — per required checklist

| Check | Result |
|---|---|
| HTTP status | PASS — `200`, `content-type: text/event-stream; charset=utf-8`, `transfer-encoding: chunked` |
| SSE event count | PASS — 8 `data:` events (7 content-carrying + 1 `[DONE]` sentinel) |
| generated text | PASS — 7 content chunks concatenate to `" Paris. The capital of the United"` — non-empty, semantically correct first token, plausible continuation under `max_tokens=8, ignore_eos=true` |
| rank health after request | PASS — parent `44261` + 3 spawn children (`44271`, `44272`, `44273`) + resource-tracker (`44270`) still alive; rank 0 logged `Scheduler is idle, waiting for new reqs...` post-response |
| no `block_size` / `561002` error | PASS — zero matches; `--page-size 16` held |
| no CUDA / NCCL / pynccl fallback | PASS — zero `pynccl`/`nccl`/`cuda`/`flashinfer` non-argument matches; HCCL + Gloo handshakes present |
| clean shutdown | PASS — Uvicorn `Application shutdown complete.` + `Finished server process [44261]`; one benign `resource_tracker` semaphore cleanup notice |
| no residual processes | PASS — final scan `NO_RESIDUAL`, port released |

**Overall verdict: PASS.**

## 12. What this gate does NOT establish

* No `/v1/chat/completions` (stream or non-stream) was issued.
  Those are follow-up gates (Phases 6B.15+).
* No multi-request, batch, or long-context behaviour was tested.
* No non-greedy sampling was exercised on Qwen3-1.7B; the
  Phase 6B.8 `flashinfer.sampling` blocker on the default OpenAI
  request body is still expected to reproduce on this model, and
  the Phase 6B.11 recipe-level workaround (`temperature=0`)
  still applies.
* No token-count accuracy check was performed; the request
  budgeted `max_tokens=8` and observed 7 content chunks + 1
  terminator, but per-token accounting on the `/generate` route
  is not part of this gate's scope.
* No client-disconnect / abort-ack behaviour was exercised.
* The benign `resource_tracker` semaphore-leak warning at parent
  exit remains a documented observation, not a root-caused
  finding.
* No performance claim of any kind is made. Wall-clock (~2.65 s
  end-to-end for 7 tokens) is anecdotal, not benchmarked.
