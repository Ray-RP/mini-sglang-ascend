# Phase 6B.22 Qwen3-4B Server `/v1/chat/completions` Stream Greedy Smoke Verdict

**Kind:** Documentation-only verdict for the Qwen3-4B fixed-TP2
server `/v1/chat/completions` (stream=true, greedy) smoke against
the Ascend host. Launched with the Phase 6B.7 recipe
([`phase6b.2_server_launch_recipe.md`](./phase6b.2_server_launch_recipe.md)
as amended at Phase 6B.7 to include `--page-size 16`) but with
`--model-path /mnt/nvme/models/Qwen3-4B`, issued one `curl -N` to
`POST /v1/chat/completions` with `temperature=0` and
`stream=true`, observed the outcome, then terminated. This file
introduces no runtime, script, test, tag, GitHub Release, or
`CHANGELOG.md` change and does not print credentials.

Envelope: fixed TP=2, eager, `npu_fia`, bf16, `page_size=16`,
greedy — v0.2.0a1 recipe, third model, streaming chat.

**Overall verdict: PASS.**

Route: with `temperature=0` in the request body (same greedy
contract validated at Phase 6B.10 / 6B.16 / 6B.21 for the
non-stream and stream chat paths on the 0.6B / 1.7B / 4B tiers),
the sampler short-circuits before importing the CUDA-only
`flashinfer.sampling` module. Both scheduler ranks survived,
`/v1/chat/completions` streamed nine SSE events (seven
content-carrying `text_completion.chunk` deltas, one final
`finish_reason="stop"` chunk with an empty delta, and one
`data: [DONE]` sentinel). The concatenated content is
`"<think>\nOkay, the user is"` — byte-for-byte identical to
Phase 6B.21 non-stream Qwen3-4B and to Phase 6B.15 / 6B.16 on
Qwen3-1.7B. The Uvicorn parent shut down cleanly with
`NO_RESIDUAL`.

---

## 1. Environment

* Host: Ascend NPU host (8 × 910B1, 64 GiB HBM per NPU).
* Container: `998ce5ba6e5e`.
* Repo tree: `/mnt/nvme/LR-606/mini-sglang-ascend` (unchanged since
  Phase 6B.3).
* Model: `/mnt/nvme/models/Qwen3-4B`.
* Working dir: `/mnt/nvme/LR-606/phase6b22/` (launch script + log
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

Same wrapper skeleton used at Phase 6B.12 → 6B.21 (`setsid nohup`
inside the container; `PYTHONPATH=python`), only the phase-specific
working directory changed to `/mnt/nvme/LR-606/phase6b22/`.

## 3. Server ready state

* Parent PID `47247`; three `multiprocessing.spawn` children
  (`47257`, `47258`, `47259` = 2 scheduler ranks + 1 detokenizer)
  + resource-tracker `47256`.
* Log excerpt:
  ```
  [Gloo] Rank 0 is connected to 1 peer ranks.
  [Gloo] Rank 1 is connected to 1 peer ranks.
  [core|rank=0] Allocating 566928 tokens for KV cache, K + V = 38.93 GiB
  [core|rank=1] Allocating 566928 tokens for KV cache, K + V = 38.93 GiB
  [core|rank=0] Free memory after initialization: 4.74 GiB
  [core|rank=0] CUDA graph is disabled.
  [core|rank=0] Scheduler is idle, waiting for new reqs...
  Scheduler is ready
  API server is ready to serve on 0.0.0.0:1919
  INFO:     Uvicorn running on http://0.0.0.0:1919 (Press CTRL+C to quit)
  ```
* Consistent with Phase 6B.18 → 6B.21: KV cache 38.93 GiB per rank,
  ready ~21 s after launch (parse 21:54:36 → ready 21:54:57).

## 4. `/v1/chat/completions` (stream=true) request

Command executed inside the container:

```
curl -N -sS -o chat_stream.body -D chat_stream.headers --max-time 60 \
  http://127.0.0.1:1919/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/mnt/nvme/models/Qwen3-4B",
    "messages": [{"role":"user","content":"What is the capital of France?"}],
    "max_tokens": 8,
    "temperature": 0,
    "stream": true
  }'
```

The `-N` flag disables curl's output buffering so SSE events land
in `chat_stream.body` as they arrive. The request body carries
`temperature=0` explicitly, satisfying `SamplingParams.is_greedy`
(`python/minisgl/core.py:30`): `(0.0 <= 0.0 or -1 == 1) and 1.0 == 1.0`
→ `True`. The sampler therefore takes the greedy `torch.argmax`
branch and never invokes `sample_impl` (the caller that imports
`flashinfer.sampling`).

## 5. Response — what the client got

Curl instrumentation:

| metric | value |
|---|---|
| HTTP status | `200` |
| content-type | `text/event-stream; charset=utf-8` |
| transfer-encoding | `chunked` |
| response bytes | `1130` |
| wall-clock | `~2.73 s` |

Response headers verbatim:

```
HTTP/1.1 200 OK
date: Sun, 12 Jul 2026 21:55:27 GMT
server: uvicorn
content-type: text/event-stream; charset=utf-8
transfer-encoding: chunked
```

Response body verbatim (SSE stream, blank separator lines
preserved by curl output):

```
data: {"id": "cmpl-0", "object": "text_completion.chunk", "choices": [{"delta": {"role": "assistant", "content": "<think>"}, "index": 0, "finish_reason": null}]}

data: {"id": "cmpl-0", "object": "text_completion.chunk", "choices": [{"delta": {"content": "\n"}, "index": 0, "finish_reason": null}]}

data: {"id": "cmpl-0", "object": "text_completion.chunk", "choices": [{"delta": {"content": "Okay"}, "index": 0, "finish_reason": null}]}

data: {"id": "cmpl-0", "object": "text_completion.chunk", "choices": [{"delta": {"content": ","}, "index": 0, "finish_reason": null}]}

data: {"id": "cmpl-0", "object": "text_completion.chunk", "choices": [{"delta": {"content": " the"}, "index": 0, "finish_reason": null}]}

data: {"id": "cmpl-0", "object": "text_completion.chunk", "choices": [{"delta": {"content": " user"}, "index": 0, "finish_reason": null}]}

data: {"id": "cmpl-0", "object": "text_completion.chunk", "choices": [{"delta": {"content": " is"}, "index": 0, "finish_reason": null}]}

data: {"id": "cmpl-0", "object": "text_completion.chunk", "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]}

data: [DONE]
```

Server access log:

```
INFO:     127.0.0.1:32832 - "POST /v1/chat/completions HTTP/1.1" 200 OK
```

### SSE event count

* Total `data:` lines: **9** (measured by
  `grep -c "^data:" chat_stream.body`).
* Content-carrying chunks: **7** — deltas
  `"<think>"`, `"\n"`, `"Okay"`, `","`, `" the"`, `" user"`,
  `" is"`.
* Terminal chunk with empty delta + `finish_reason="stop"`: **1**.
* `[DONE]` sentinel: **1** — final `data: [DONE]` line.

### Delta / content fragments

The seven content-carrying deltas concatenate to:

```
<think>
Okay, the user is
```

(byte-identical to `"<think>\nOkay, the user is"` — the same
prefix produced by the non-stream chat gate at Phase 6B.21 §5 on
the same Qwen3-4B model and to Phase 6B.9 / 6B.10 on Qwen3-0.6B
and Phase 6B.15 / 6B.16 on Qwen3-1.7B). The first delta carries
`"role": "assistant"` — the OpenAI streaming convention where the
role is emitted only on the initial chunk. Subsequent content
chunks omit `role`, and the terminal chunk omits both `role` and
`content` and instead carries `finish_reason: "stop"`.

### `[DONE]` presence

Present exactly once at the tail of the SSE stream. Matches the
`/v1/chat/completions` stream contract observed at Phase 6B.10
for Qwen3-0.6B and Phase 6B.16 for Qwen3-1.7B.

### Cross-check against Phase 6B.10 / 6B.16 / 6B.21

| gate | model | route | first-8-token content |
|---|---|---|---|
| Phase 6B.10 | Qwen3-0.6B | chat, stream | `"<think>\nOkay, the user is"` |
| Phase 6B.16 | Qwen3-1.7B | chat, stream | `"<think>\nOkay, the user is"` |
| Phase 6B.21 | Qwen3-4B | chat, non-stream | `"<think>\nOkay, the user is"` |
| Phase 6B.22 | Qwen3-4B | chat, stream | `"<think>\nOkay, the user is"` |

All four gates emit the byte-identical `<think>` prefix under
`max_tokens=8, temperature=0`. This confirms:

* the streaming and non-streaming greedy chat paths on Qwen3-4B
  share the same sampler pathway and produce the same tokens in
  the same order;
* the greedy `<think>` prefix is stable across the 0.6B → 1.7B →
  4B jumps under the Qwen3 chat template;
* no accidental sampling drift was introduced by streaming.

## 6. `flashinfer.sampling` import status

* `grep -inE "flashinfer"` on
  `/mnt/nvme/LR-606/phase6b22/server.log`: **zero matches**.
* Neither scheduler rank raised the Phase 6B.8
  `ModuleNotFoundError: No module named 'flashinfer'`.
* Interpretation: `Sampler.prepare`
  (`python/minisgl/engine/sample.py:55`) set
  `BatchSamplingArgs.temperatures = None` for the greedy request;
  `Sampler.sample` fell into the `torch.argmax` branch, so
  `sample_impl` — the caller that imports `flashinfer.sampling`
  at `python/minisgl/engine/sample.py:30` — was never executed.

**Flashinfer import: did NOT occur.**

## 7. `block_size` / CANN error status

* `grep -inE "block_size|561002|CheckFeatureNoquant"` on the log:
  **zero matches**.
* `--page-size 16` from the Phase 6B.7 recipe held. The BF16
  no-quant `aclnnFusedInferAttentionScoreV3` kernel accepted the
  16-aligned paged-KV blocks throughout the streamed request.

## 8. No CUDA / NCCL / pynccl fallback

Grep over the full server log:

* `pynccl` / `nccl` external matches (excluding the intentional
  `use_pynccl=False` argument-echo line in the `ServerArgs`
  block): **zero**.
* `cuda` external matches (excluding `cuda_graph_bs=None`,
  `cuda_graph_max_bs=0` in the `ServerArgs` block, and the
  deliberate `CUDA graph is disabled.` notice): **zero**.
* HCCL + Gloo evidence present: 2× `ProcessGroupHCCL` watchdog
  warnings + `[Gloo] Rank {0,1} is connected to 1 peer ranks.`
  (4 combined matches).

The greedy path never touched CUDA-only sampling code, and no
CUDA-only comm backend was loaded.

## 9. Scheduler rank health after the request

Post-response process tree:

```
PID     PPID    CMD
47247   47246   python -m minisgl.server.launch --model-path /mnt/nvme/models/Qwen3-4B ...
47256   47247     python -c from multiprocessing.resource_tracker import main;main(22)
47257   47247     python -c from multiprocessing.spawn import spawn_main; ... pipe_handle=25 --multiprocessing-fork
47258   47247     python -c from multiprocessing.spawn import spawn_main; ... pipe_handle=27 --multiprocessing-fork
47259   47247     python -c from multiprocessing.spawn import spawn_main; ... pipe_handle=29 --multiprocessing-fork
```

* All three `spawn_main` children (`47257`, `47258`, `47259`) and
  the resource-tracker `47256` remained alive after the streamed
  request completed.
* Post-response log line:
  ```
  [2026-07-12|21:55:30|core|rank=0] Scheduler is idle, waiting for new reqs...
  ```
  Rank 0 returned to the idle state after the streamed request
  finished — the request went all the way through the forward
  loop, back through the detokenizer/frontend, and the scheduler
  released the slot. Timeline: rank 0 idle at 21:54:57 (ready)
  → request completed at 21:55:27 (HTTP 200) → scheduler idle at
  21:55:30 (~3 s working window, matching the ~2.73 s curl
  wall-clock plus scheduler handoff).
* Zero `Traceback` / `ModuleNotFoundError` / `RuntimeError`
  matches in the server log.

## 10. HBM headroom after the request

Per-NPU HBM (via `npu-smi info`, HBM-Usage column, MB used / MB
total) captured immediately after the SSE stream completed:

| NPU | HBM-Usage (MB) | Interpretation |
|---|---|---|
| 0 | `61157 / 65536` | Rank 0 pinned here (~59.7 GiB used, ~4.28 GiB free) |
| 1 | `61157 / 65536` | Rank 1 pinned here (~59.7 GiB used, ~4.28 GiB free) |
| 2–7 | ~16,453–16,454 / 65,536 | Container baseline only |

Per-NPU proc-mem confirms the 1-rank-to-1-NPU pinning is
unchanged after the request:

```
NPU 0: proc-mem 44761 MB (scheduler rank 0) + 13090 MB (HCCL sidecar)
NPU 1: proc-mem 44761 MB (scheduler rank 1) + 13090 MB (HCCL sidecar)
```

Comparison to prior gates on Qwen3-4B:

| Gate | HBM-Usage NPU 0/1 (MB) | Scheduler proc-mem (MB) | Note |
|---|---|---|---|
| 6B.18 bring-up (no request) | 60723 / 60722 | 44320 | baseline |
| 6B.19 `/v1/models` (frontend-only) | 60722 / 60722 | 44320 | unchanged |
| 6B.20 `/generate` (7 gen tokens) | 61139 / 61138 | 44743 | +416 MB working set |
| 6B.21 `/v1/chat/completions` non-stream (8 gen tokens) | 61157 / 61157 | 44761 | +435 MB working set |
| 6B.22 `/v1/chat/completions` stream (this gate) | 61157 / 61157 | 44761 | +435 MB working set (unchanged from 6B.21) |

The ~435 MB per-rank delta vs. bring-up is live-request working
memory (activations for the forward pass, transient KV entries
written for the eight generated tokens). Identical to Phase
6B.21 — as expected, since the sampler / forward pass generated
the same eight tokens on the same model. Streaming vs.
non-streaming only affects how the frontend serialises the output
to the client; it does not change the working-memory footprint.
The pool still holds ~4.28 GiB free per rank; headroom did not
collapse under the streamed single-request load.

**HBM headroom: PASS — held at ~4.28 GiB free per active rank
post-request.**

## 11. Termination and residual check

* `kill -TERM 47247` (parent). Uvicorn shutdown sequence
  completed:
  ```
  INFO:     Shutting down
  INFO:     Waiting for application shutdown.
  INFO:     Application shutdown complete.
  INFO:     Finished server process [47247]
  Terminated
  ```
* Follow-up SIGTERM sweep over surviving
  `multiprocessing.spawn` / `resource_tracker` children exited on
  the signal path (no `SIGKILL` fallback).
* Final `ps -eo pid,cmd | grep -E "minisgl.server|multiprocessing.(spawn|resource)"`:
  `NO_RESIDUAL`.
* Listening port `:1919` released; `netstat`: no match.
* Same benign
  `multiprocessing.resource_tracker: There appear to be 2 leaked
  semaphore objects to clean up at shutdown` warning as
  Phases 6B.5, 6B.7, 6B.9, 6B.10, 6B.12–6B.16, 6B.18–6B.21.

## 12. Verdict — per required checklist

| Check | Result |
|---|---|
| HTTP status | PASS — `200`, `content-type: text/event-stream; charset=utf-8`, `transfer-encoding: chunked` |
| SSE event count | PASS — 9 `data:` events (7 content-carrying deltas + 1 terminal `finish_reason="stop"` chunk + 1 `[DONE]` sentinel) |
| delta / content fragments | PASS — deltas `"<think>"`, `"\n"`, `"Okay"`, `","`, `" the"`, `" user"`, `" is"` concatenate to `"<think>\nOkay, the user is"`; byte-identical to Phase 6B.21 Qwen3-4B non-stream and Phase 6B.10 / 6B.16 stream chat on 0.6B / 1.7B |
| `[DONE]` presence | PASS — present exactly once at tail |
| `flashinfer.sampling` import status | PASS — did NOT occur; zero `flashinfer` matches in log |
| rank health after request | PASS — parent `47247` + 3 spawn children (`47257`, `47258`, `47259`) + resource-tracker (`47256`) still alive; rank 0 logged `Scheduler is idle, waiting for new reqs...` post-response |
| HBM headroom after request | PASS — HBM-Usage 61,157 / 65,536 MB on both active NPUs (~4.28 GiB free per rank); ~435 MB working-memory delta vs. bring-up (identical to Phase 6B.21) |
| no `block_size` / `561002` error | PASS — zero matches; `--page-size 16` held |
| no CUDA / NCCL / pynccl fallback | PASS — zero `pynccl` / `nccl` / `cuda` / `flashinfer` non-argument matches; HCCL + Gloo handshakes present |
| clean shutdown | PASS — Uvicorn `Application shutdown complete.` + `Finished server process [47247]`; one benign `resource_tracker` semaphore cleanup notice; `NO_RESIDUAL`; port released |

**Overall verdict: PASS.**

## 13. What this gate does NOT establish

* No non-greedy request was attempted. The Phase 6B.8 root cause
  (default OpenAI sampling routes through CUDA-only
  `flashinfer.sampling`) is worked around here by request-body
  shape, not fixed at the sampler. The Phase 6B.11 §3 constraint
  ("`temperature=0` (or `top_k=1`) required to avoid the
  `flashinfer.sampling` path") continues to apply on Qwen3-4B
  under streaming as well.
* `usage` is entirely absent from the streaming chunk envelope,
  same as Phase 6B.10 / 6B.16 on the 0.6B / 1.7B tiers; the
  Phase 6B.11 §3 known-constraint that usage counters are
  currently unpopulated continues to apply.
* No multi-request or batch behaviour was tested.
* No client-disconnect / abort-ack behaviour was exercised.
* The benign `resource_tracker` semaphore-leak warning at parent
  exit remains a documented observation, not a root-caused
  finding.
* No performance claim of any kind is made. Wall-clock (~2.73 s
  end-to-end for the streamed 8-token response) is anecdotal, not
  benchmarked.
