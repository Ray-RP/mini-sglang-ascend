# Phase 6B.9 Qwen3-0.6B Server `/v1/chat/completions` Non-Stream Greedy Smoke Verdict

**Kind:** Documentation-only verdict for the Qwen3-0.6B fixed-TP2
server `/v1/chat/completions` (non-stream, greedy) smoke against
the Ascend host. Launched with the Phase 6B.7 recipe
([`phase6b.2_server_launch_recipe.md`](./phase6b.2_server_launch_recipe.md)
as amended at Phase 6B.7 to include `--page-size 16`), issued one
`curl` to `POST /v1/chat/completions` with `temperature=0` and
`stream=false`, observed the outcome, then terminated. This file
introduces no runtime, script, test, tag, GitHub Release, or
`CHANGELOG.md` change and does not print credentials.

Envelope: fixed TP=2, eager, `npu_fia`, bf16, `page_size=16`,
greedy — v0.2.0a1.

**Overall verdict: PASS.**

Route: with `temperature=0` in the request body, the sampler
short-circuits before ever importing the CUDA-only
`flashinfer.sampling` module. Both scheduler ranks survived,
`/v1/chat/completions` returned a complete OpenAI-compatible JSON
envelope with a non-empty `choices[0].message.content`, and the
Uvicorn parent shut down cleanly.

---

## 1. Greedy contract (as read from source)

`SamplingParams.is_greedy` at `python/minisgl/core.py:30`:

```python
@property
def is_greedy(self) -> bool:
    return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0
```

Consumers:

* `python/minisgl/engine/sample.py:55` — `Sampler.prepare` sets
  `BatchSamplingArgs.temperatures = None` iff every
  `SamplingParams.is_greedy` is `True`.
* `python/minisgl/engine/sample.py:59` — non-greedy path builds a
  per-request temperature vector.

The API surface (`python/minisgl/server/api_server.py:66-85`)
defaults `temperature=1.0`, `top_k=-1`, `top_p=1.0`. To land on
the greedy branch, the request body must override at least one of
`temperature≤0` or `top_k==1`, and must keep `top_p==1.0`. This
gate sends `temperature=0` while leaving the other two at
defaults, which satisfies the predicate:
`(0.0 <= 0.0 or -1 == 1) and 1.0 == 1.0` → `True`.

## 2. Environment

* Host: Ascend NPU host (8 × 910B1).
* Container: `998ce5ba6e5e`.
* Repo tree: `/mnt/nvme/LR-606/mini-sglang-ascend` (unchanged since
  Phase 6B.3).
* Model: `/mnt/nvme/models/Qwen3-0.6B`.
* Working dir: `/mnt/nvme/LR-606/phase6b9/` (launch script + log +
  response headers/body).
* Port: `1919`.

## 3. Launch invocation (as executed)

```
python -m minisgl.server.launch \
  --model-path /mnt/nvme/models/Qwen3-0.6B \
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
attention_backend='npu_fia',
cuda_graph_bs=None, cuda_graph_max_bs=0,
page_size=16,
use_pynccl=False,
tp_info=DistributedInfo(rank=0, size=2),
dtype=torch.bfloat16
```

## 4. Server ready state

* Parent PID `42749`; three `multiprocessing.spawn` children
  (`42759`, `42760`, `42761` = 2 scheduler ranks + 1 detokenizer)
  + resource-tracker `42758`.
* Log excerpt:
  ```
  [Gloo] Rank 0 is connected to 1 peer ranks.
  [Gloo] Rank 1 is connected to 1 peer ranks.
  CUDA graph is disabled.
  Scheduler is idle, waiting for new reqs...
  Scheduler is ready
  API server is ready to serve on 0.0.0.0:1919
  INFO:     Uvicorn running on http://0.0.0.0:1919 (Press CTRL+C to quit)
  ```
* `netstat`: `tcp 0.0.0.0:1919 LISTEN 42749/python`.

## 5. `/v1/chat/completions` request

Command executed inside the container:

```
curl -sS -o chat.body -D chat.headers \
  --max-time 60 \
  http://127.0.0.1:1919/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/mnt/nvme/models/Qwen3-0.6B",
    "messages": [{"role":"user","content":"What is the capital of France?"}],
    "max_tokens": 8,
    "temperature": 0,
    "stream": false
  }'
```

The request body carries `temperature=0` explicitly. All other
sampling knobs use `OpenAICompletionRequest` defaults (`top_k=-1`,
`top_p=1.0`), which — combined with `temperature=0` — satisfies
`SamplingParams.is_greedy` (§1).

## 6. Response — what the client got

Curl instrumentation:

| metric | value |
|---|---|
| HTTP status | `200` |
| content-type | `application/json` |
| content-length | `289` |
| response bytes | `289` |
| wall-clock | `~2.76 s` |

Response headers verbatim:

```
HTTP/1.1 200 OK
date: Sun, 12 Jul 2026 18:32:04 GMT
server: uvicorn
content-length: 289
content-type: application/json
```

Response body verbatim (JSON):

```json
{
  "id": "chatcmpl-0",
  "object": "chat.completion",
  "created": 1783881127,
  "model": "/mnt/nvme/models/Qwen3-0.6B",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "<think>\nOkay, the user is"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
}
```

Server access log:

```
INFO:     127.0.0.1:33604 - "POST /v1/chat/completions HTTP/1.1" 200 OK
```

### Per-field verdict

| field | observed | verdict |
|---|---|---|
| `id` | `"chatcmpl-0"` | PASS — populated, deterministic sequence handle |
| `object` | `"chat.completion"` | PASS — matches OpenAI shape |
| `created` | `1783881127` (Unix seconds) | PASS |
| `model` | `"/mnt/nvme/models/Qwen3-0.6B"` | PASS — matches the id from `/v1/models` (Phase 6B.5) |
| `choices` | list of length 1 | PASS |
| `choices[0].index` | `0` | PASS |
| `choices[0].message.role` | `"assistant"` | PASS |
| `choices[0].message.content` | `"<think>\nOkay, the user is"` | PASS — non-empty completion string, 8 whitespace-delimited pieces greedy-decoded from Qwen3-0.6B's thinking prefix under `max_tokens=8` |
| `choices[0].finish_reason` | `"stop"` | PASS — populated |
| `usage.prompt_tokens` | `0` | PARTIAL — usage counters are all zero. Observation only; not part of this gate's PASS criteria. |
| `usage.completion_tokens` | `0` | PARTIAL — same |
| `usage.total_tokens` | `0` | PARTIAL — same |

The `usage` block is present and correctly typed, but every
counter reports `0` — the server does not yet populate these
fields on the chat.completions non-stream path. This is recorded
as an observation, not a scope failure of Phase 6B.9.

## 7. `flashinfer.sampling` import status

* `grep -inE "flashinfer"` on
  `/mnt/nvme/LR-606/phase6b9/server.log`: **zero matches**.
* Neither scheduler rank raised the Phase 6B.8
  `ModuleNotFoundError: No module named 'flashinfer'`.
* Interpretation: `Sampler.prepare` at
  `python/minisgl/engine/sample.py:55` shortcut the request into
  the greedy branch (`args.temperatures is None`), so
  `Sampler.sample` at line 76-78 fell into the
  `torch.argmax` path. `sample_impl` — the caller that imports
  `flashinfer.sampling` at line 30 — was never executed.

**Flashinfer import: did NOT occur.**

## 8. `block_size` / CANN error status

* `grep -inE "block_size|561002|CheckFeatureNoquant"` on the log:
  **zero matches**.
* `--page-size 16` from the Phase 6B.7 recipe held.

## 9. No CUDA / NCCL / pynccl fallback

Grep over the full server log:

* `pynccl` / `nccl` external matches (excluding the intentional
  `use_pynccl=False` argument-echo line): **zero**.
* `cuda` external matches (excluding `cuda_graph_bs=`,
  `cuda_graph_max_bs=`, and the deliberate
  `CUDA graph is disabled.` notice): **zero**.
* HCCL + Gloo evidence present: 2× `ProcessGroupHCCL` watchdog
  warnings + `[Gloo] Rank {0,1} is connected to 1 peer ranks.`.

The greedy path never touched CUDA-only sampling code, and no
CUDA/NCCL/pynccl comm backend was loaded.

## 10. Scheduler rank health after the request

* All three `spawn_main` children (`42759`, `42760`, `42761`)
  remained alive after the response completed.
* Post-response log line:
  ```
  [2026-07-12|18:32:07|core|rank=0] Scheduler is idle, waiting for new reqs...
  ```
  This is proof rank 0 returned to the idle state — the request
  went all the way through the forward loop and back to the
  scheduler, not just up to the HTTP layer.
* Zero `Traceback` / `ModuleNotFoundError` / `RuntimeError`
  matches in the server log.

## 11. Termination and residual check

* `kill -TERM 42749` (parent). Uvicorn shutdown sequence
  completed:
  ```
  Shutting down
  Waiting for application shutdown.
  Application shutdown complete.
  Finished server process [42749]
  ```
* Follow-up SIGTERM sweep over surviving
  `multiprocessing.spawn` / `resource_tracker` children exited on
  the signal path (no `SIGKILL` fallback).
* Final `ps -eo pid,cmd | grep -E "minisgl.server|multiprocessing.(spawn|resource)"`:
  `NO_RESIDUAL`.
* Listening port `:1919` released; `netstat`: no match.

## 12. Verdict — per required checklist

| Check | Result |
|---|---|
| HTTP status | PASS — `200`, `content-type: application/json`, `content-length: 289` |
| response JSON shape | PASS — OpenAI `chat.completion` envelope with `id`, `object`, `created`, `model`, `choices`, `usage` |
| `choices[0].message.content` observed | PASS — `"<think>\nOkay, the user is"` (non-empty completion) |
| `finish_reason` observed | PASS — `"stop"` |
| `flashinfer.sampling` import status | PASS — did NOT occur; zero `flashinfer` matches in log |
| rank health after request | PASS — parent + 3 spawn children still alive; rank 0 logged `Scheduler is idle, waiting for new reqs...` post-response |
| no `block_size` / `561002` error | PASS — zero matches |
| no CUDA / NCCL / pynccl fallback | PASS — zero non-argument matches; HCCL + Gloo present |
| clean shutdown | PASS — Uvicorn `Application shutdown complete.` + `Finished server process [42749]` |
| no residual server processes | PASS — final scan `NO_RESIDUAL`, port released |

**Overall verdict: PASS.**

## 13. What this gate does NOT establish

* No `/v1/chat/completions` **with `stream=true`** was attempted.
* No multi-request or batch behaviour was tested.
* `usage.prompt_tokens` / `.completion_tokens` / `.total_tokens`
  all report `0`; the accuracy of usage accounting is not covered
  by this gate and is out of scope for the v0.2.0a1 envelope.
* The Phase 6B.8 root cause (non-greedy default routes through
  CUDA-only `flashinfer.sampling`) is worked around here by
  request-body shape, not fixed at the sampler. A follow-up
  gate must decide whether to lock this shape as a client
  contract or introduce an NPU-compatible sampling backend.
* No performance claim of any kind is made.
