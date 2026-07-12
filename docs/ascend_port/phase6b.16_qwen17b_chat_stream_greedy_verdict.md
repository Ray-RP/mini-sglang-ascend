# Phase 6B.16 Qwen3-1.7B Server `/v1/chat/completions` Stream Greedy Smoke Verdict

**Kind:** Documentation-only verdict for the Qwen3-1.7B fixed-TP2
server `/v1/chat/completions` (streaming, greedy) smoke against the
Ascend host. Launched with the Phase 6B.7 recipe
([`phase6b.2_server_launch_recipe.md`](./phase6b.2_server_launch_recipe.md)
as amended at Phase 6B.7 to include `--page-size 16`) but with
`--model-path /mnt/nvme/models/Qwen3-1.7B`, issued one `curl -N`
to `POST /v1/chat/completions` with `temperature=0` and
`stream=true`, observed the outcome, then terminated. This file
introduces no runtime, script, test, tag, GitHub Release, or
`CHANGELOG.md` change and does not print credentials.

Envelope: fixed TP=2, eager, `npu_fia`, bf16, `page_size=16`,
greedy, streaming — v0.2.0a1 recipe, second model.

**Overall verdict: PASS.**

Route: with `temperature=0` in the request body (same greedy
contract validated at Phase 6B.10 for Qwen3-0.6B and Phase 6B.15
for Qwen3-1.7B non-stream), the sampler short-circuits before
importing `flashinfer.sampling`. The `/v1/chat/completions` stream
returned nine SSE `data:` events — seven `assistant` content
deltas, one `finish_reason=stop` terminator, and a `[DONE]`
sentinel — before Uvicorn closed the chunked stream cleanly. Both
scheduler ranks stayed alive; rank 0 returned to idle after the
request.

---

## 1. Environment

* Host: Ascend NPU host (8 × 910B1).
* Container: `998ce5ba6e5e`.
* Repo tree: `/mnt/nvme/LR-606/mini-sglang-ascend` (unchanged since
  Phase 6B.3).
* Model: `/mnt/nvme/models/Qwen3-1.7B`.
* Working dir: `/mnt/nvme/LR-606/phase6b16/` (launch script + log
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

Same wrapper skeleton used at Phase 6B.12–6B.15 (`setsid nohup`
inside the container; `PYTHONPATH=python`), only the phase-specific
working directory changed to `/mnt/nvme/LR-606/phase6b16/`.

## 3. Server ready state

* Parent PID `45089`; three `multiprocessing.spawn` children
  (`45099`, `45100`, `45101` = 2 scheduler ranks + 1 detokenizer)
  + resource-tracker `45098`.
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
* `netstat`: `tcp 0.0.0.0:1919 LISTEN 45089/python`.
* Consistent with Phase 6B.12–6B.15: KV cache 41.27 GiB per rank,
  ready ~20 s after launch (rank 0 idle at 20:04:55).

## 4. `/v1/chat/completions` streaming request

Command executed inside the container:

```
curl -N -sS -o chat.body -D chat.headers \
  --max-time 60 \
  http://127.0.0.1:1919/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/mnt/nvme/models/Qwen3-1.7B",
    "messages": [{"role":"user","content":"What is the capital of France?"}],
    "max_tokens": 8,
    "temperature": 0,
    "stream": true
  }'
```

The `-N` flag disables curl's output buffering so the SSE events
land in `chat.body` as they arrive. `temperature=0` keeps the
sampler on the greedy branch (Phase 6B.9 §1); `stream=true`
selects the SSE path.

## 5. Response — what the client got

Curl instrumentation:

| metric | value |
|---|---|
| HTTP status | `200` |
| content-type | `text/event-stream; charset=utf-8` |
| transfer-encoding | `chunked` |
| response bytes | `1130` |
| wall-clock | `~2.64 s` |

Response headers verbatim:

```
HTTP/1.1 200 OK
date: Sun, 12 Jul 2026 20:05:20 GMT
server: uvicorn
content-type: text/event-stream; charset=utf-8
transfer-encoding: chunked
```

Response body verbatim (SSE stream, blank separator lines
preserved):

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
INFO:     127.0.0.1:43710 - "POST /v1/chat/completions HTTP/1.1" 200 OK
```

### SSE event count

* Total `data:` lines: **9** (measured by
  `grep -c "^data: " chat.body`).
* Content-carrying deltas (`delta.content` present): **7**
  — `"<think>"`, `"\n"`, `"Okay"`, `","`, `" the"`, `" user"`,
  `" is"`.
* Role-first delta: **1** — the very first event carries both
  `delta.role = "assistant"` and the initial `delta.content =
  "<think>"`. Counted inside the 7 content-carrying deltas above.
* Terminator delta (`delta` empty, `finish_reason = "stop"`):
  **1**.
* `[DONE]` sentinel: **1** — final `data: [DONE]` line, matched
  by `grep -c "\[DONE\]"`.

### Delta / content fragments (concatenated)

The concatenation of `delta.content` across the seven content
deltas is:

```
<think>
Okay, the user is
```

Under `max_tokens=8, temperature=0, greedy`, Qwen3-1.7B emitted
its `<think>` prefix followed by a chain-of-thought continuation,
then stopped after the eighth-token budget was reached. This
prefix is **byte-for-byte identical** to:

* the Phase 6B.10 Qwen3-0.6B stream greedy result, and
* the Phase 6B.15 Qwen3-1.7B non-stream greedy
  `choices[0].message.content`.

Confirming both:

* the streaming path produces the exact greedy-decoded prefix as
  the non-stream path on Qwen3-1.7B (cross-check against Phase
  6B.15), and
* Qwen3-0.6B and Qwen3-1.7B agree on the first eight greedy tokens
  under this prompt (shared tokenizer + shared greedy pathway).

### `[DONE]` presence

Present exactly once at the tail of the SSE stream, after the
`finish_reason = "stop"` event. Matches the OpenAI SSE contract
and the Phase 6B.10 Qwen3-0.6B observation.

## 6. `flashinfer.sampling` import status

* `grep -inE "flashinfer"` on
  `/mnt/nvme/LR-606/phase6b16/server.log`: **zero matches**.
* Neither scheduler rank raised the Phase 6B.8
  `ModuleNotFoundError: No module named 'flashinfer'`.
* Interpretation: `Sampler.prepare` at
  `python/minisgl/engine/sample.py:55` set
  `BatchSamplingArgs.temperatures = None` for the greedy request;
  `Sampler.sample` fell into the `torch.argmax` branch, and
  `sample_impl` (the CUDA-only-import caller at
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

The greedy streaming path never touched CUDA-only sampling code,
and no CUDA-only comm backend was loaded.

## 9. Scheduler rank health after the request

* All three `spawn_main` children (`45099`, `45100`, `45101`) and
  the resource-tracker `45098` remained alive after the SSE
  stream completed.
* Post-response log line:
  ```
  [2026-07-12|20:05:24|core|rank=0] Scheduler is idle, waiting for new reqs...
  ```
  Rank 0 returned to the idle state after the streamed request
  finished — the request went all the way through the forward
  loop, back through the detokenizer/frontend, and the scheduler
  released the slot.
* Zero `Traceback` / `ModuleNotFoundError` / `RuntimeError`
  matches in the server log.

## 10. Termination and residual check

* `kill -TERM 45089` (parent). Uvicorn shutdown sequence
  completed:
  ```
  INFO:     Shutting down
  INFO:     Waiting for application shutdown.
  INFO:     Application shutdown complete.
  INFO:     Finished server process [45089]
  ```
* One benign
  `multiprocessing.resource_tracker: There appear to be 2 leaked
  semaphore objects to clean up at shutdown` warning at parent
  exit — same benign notice recorded across Phases 6B.5, 6B.7,
  6B.9, 6B.10, 6B.12, 6B.13, 6B.14, 6B.15.
* Follow-up SIGTERM sweep over surviving
  `multiprocessing.spawn` / `resource_tracker` children (`45098`,
  `45099`, `45100`, `45101`) exited on the signal path (no
  `SIGKILL` fallback).
* Final `ps -eo pid,cmd | grep -E "minisgl.server|multiprocessing.(spawn|resource)"`:
  `NO_RESIDUAL`.
* Listening port `:1919` released; `netstat`: no match.

## 11. Verdict — per required checklist

| Check | Result |
|---|---|
| HTTP status | PASS — `200`, `content-type: text/event-stream; charset=utf-8`, `transfer-encoding: chunked` |
| content-type | PASS — `text/event-stream; charset=utf-8` |
| SSE event count | PASS — 9 `data:` events (7 content deltas + 1 `finish_reason=stop` terminator + 1 `[DONE]`) |
| delta / content fragments | PASS — 7 non-empty `delta.content` values concatenate to `"<think>\nOkay, the user is"`, matching the Phase 6B.15 Qwen3-1.7B non-stream body and the Phase 6B.10 Qwen3-0.6B stream body |
| `[DONE]` presence | PASS — exactly one `data: [DONE]` at end of stream |
| `flashinfer.sampling` import status | PASS — did NOT occur; zero `flashinfer` matches in log |
| rank health after request | PASS — parent `45089` + 3 spawn children (`45099`, `45100`, `45101`) + resource-tracker (`45098`) still alive; rank 0 logged `Scheduler is idle, waiting for new reqs...` post-response |
| no `block_size` / `561002` error | PASS — zero matches; `--page-size 16` held |
| no CUDA / NCCL / pynccl fallback | PASS — zero `pynccl`/`nccl`/`cuda`/`flashinfer` non-argument matches; HCCL + Gloo handshakes present |
| clean shutdown | PASS — Uvicorn `Application shutdown complete.` + `Finished server process [45089]`; one benign `resource_tracker` semaphore cleanup notice |

**Overall verdict: PASS.**

## 12. What this gate does NOT establish

* No non-greedy (`temperature > 0` or `top_k != 1`) streaming
  request was attempted. The Phase 6B.8 root cause
  (`flashinfer.sampling` import on the non-greedy path) is worked
  around here by request-body shape; a real fix is out of scope
  for this gate. The Phase 6B.11 §3 constraint
  ("`temperature=0` (or `top_k=1`) required to avoid the
  `flashinfer.sampling` path") continues to apply on Qwen3-1.7B.
* No `usage` accounting was tested; the streaming chunk envelope
  in this build does not carry a trailing `usage` object (same
  as Phase 6B.10).
* No client-disconnect / abort-ack behaviour was exercised
  (Gate 2.3f, referenced in the Phase 6B.2 route inventory,
  remains untested against the server path on either model).
* No multi-request or batch behaviour was tested.
* The benign `resource_tracker` semaphore-leak warning at parent
  exit remains a documented observation, not a root-caused
  finding.
* No performance claim of any kind is made. Wall-clock (~2.64 s
  end-to-end for 7 tokens streamed) is anecdotal, not
  benchmarked.
