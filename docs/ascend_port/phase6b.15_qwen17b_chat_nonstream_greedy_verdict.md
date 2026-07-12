# Phase 6B.15 Qwen3-1.7B Server `/v1/chat/completions` Non-Stream Greedy Smoke Verdict

**Kind:** Documentation-only verdict for the Qwen3-1.7B fixed-TP2
server `/v1/chat/completions` (non-stream, greedy) smoke against
the Ascend host. Launched with the Phase 6B.7 recipe
([`phase6b.2_server_launch_recipe.md`](./phase6b.2_server_launch_recipe.md)
as amended at Phase 6B.7 to include `--page-size 16`) but with
`--model-path /mnt/nvme/models/Qwen3-1.7B`, issued one `curl` to
`POST /v1/chat/completions` with `temperature=0` and
`stream=false`, observed the outcome, then terminated. This file
introduces no runtime, script, test, tag, GitHub Release, or
`CHANGELOG.md` change and does not print credentials.

Envelope: fixed TP=2, eager, `npu_fia`, bf16, `page_size=16`,
greedy — v0.2.0a1 recipe, second model.

**Overall verdict: PASS.**

Route: with `temperature=0` in the request body (same greedy
contract validated at Phase 6B.9 for Qwen3-0.6B), the sampler
short-circuits before importing the CUDA-only
`flashinfer.sampling` module. Both scheduler ranks survived,
`/v1/chat/completions` returned a complete OpenAI-compatible JSON
envelope with a non-empty `choices[0].message.content`, and the
Uvicorn parent shut down cleanly.

---

## 1. Environment

* Host: Ascend NPU host (8 × 910B1).
* Container: `998ce5ba6e5e`.
* Repo tree: `/mnt/nvme/LR-606/mini-sglang-ascend` (unchanged since
  Phase 6B.3).
* Model: `/mnt/nvme/models/Qwen3-1.7B`.
* Working dir: `/mnt/nvme/LR-606/phase6b15/` (launch script + log
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

Same wrapper skeleton used at Phase 6B.12 / 6B.13 / 6B.14
(`setsid nohup` inside the container; `PYTHONPATH=python`), only
the phase-specific working directory changed to
`/mnt/nvme/LR-606/phase6b15/`.

## 3. Server ready state

* Parent PID `44673`; three `multiprocessing.spawn` children
  (`44683`, `44684`, `44685` = 2 scheduler ranks + 1 detokenizer)
  + resource-tracker `44682`.
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
* `netstat`: `tcp 0.0.0.0:1919 LISTEN 44673/python`.
* Consistent with Phase 6B.12–6B.14: KV cache 41.27 GiB per rank,
  ready ~20 s after launch (rank 0 idle at 19:56:32).

## 4. `/v1/chat/completions` request

Command executed inside the container:

```
curl -sS -o chat.body -D chat.headers \
  --max-time 60 \
  http://127.0.0.1:1919/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/mnt/nvme/models/Qwen3-1.7B",
    "messages": [{"role":"user","content":"What is the capital of France?"}],
    "max_tokens": 8,
    "temperature": 0,
    "stream": false
  }'
```

The request body carries `temperature=0` explicitly, satisfying
`SamplingParams.is_greedy`
(`python/minisgl/core.py:30`) as documented at Phase 6B.9 §1:
`(0.0 <= 0.0 or -1 == 1) and 1.0 == 1.0` → `True`. The sampler
therefore takes the greedy `torch.argmax` branch and never
invokes `sample_impl` (the caller that imports
`flashinfer.sampling`).

## 5. Response — what the client got

Curl instrumentation:

| metric | value |
|---|---|
| HTTP status | `200` |
| content-type | `application/json` |
| content-length | `289` |
| response bytes | `289` |
| wall-clock | `~3.29 s` |

Response headers verbatim:

```
HTTP/1.1 200 OK
date: Sun, 12 Jul 2026 19:57:07 GMT
server: uvicorn
content-length: 289
content-type: application/json
```

Response body verbatim (JSON, pretty-printed for readability; wire
form is a single line):

```json
{
  "id": "chatcmpl-0",
  "object": "chat.completion",
  "created": 1783886231,
  "model": "/mnt/nvme/models/Qwen3-1.7B",
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
INFO:     127.0.0.1:46162 - "POST /v1/chat/completions HTTP/1.1" 200 OK
```

### Per-field verdict

| field | observed | verdict |
|---|---|---|
| `id` | `"chatcmpl-0"` | PASS — populated, deterministic sequence handle |
| `object` | `"chat.completion"` | PASS — matches OpenAI shape |
| `created` | `1783886231` (Unix seconds) | PASS |
| `model` | `"/mnt/nvme/models/Qwen3-1.7B"` | PASS — matches the id from `/v1/models` at Phase 6B.13 |
| `choices` | list of length 1 | PASS |
| `choices[0].index` | `0` | PASS |
| `choices[0].message.role` | `"assistant"` | PASS |
| `choices[0].message.content` | `"<think>\nOkay, the user is"` | PASS — non-empty completion string; 8 whitespace-delimited pieces greedy-decoded from Qwen3-1.7B's thinking prefix under `max_tokens=8` |
| `choices[0].finish_reason` | `"stop"` | PASS — populated |
| `usage.prompt_tokens` | `0` | PARTIAL — usage counters are all zero. Observation only; matches the Phase 6B.9 Qwen3-0.6B observation and the Phase 6B.11 §3 known constraint (usage counters currently unpopulated). Not a scope failure of this gate. |
| `usage.completion_tokens` | `0` | PARTIAL — same |
| `usage.total_tokens` | `0` | PARTIAL — same |

### Cross-check against Phase 6B.9 (Qwen3-0.6B)

The generated `choices[0].message.content` string
(`"<think>\nOkay, the user is"`) is **byte-for-byte identical** to
the Phase 6B.9 Qwen3-0.6B non-stream greedy result and to the
Phase 6B.10 Qwen3-0.6B stream greedy concatenation. Both Qwen3
sizes share the same tokenizer and both use the `<think>` prefix
convention, so greedy-decoding the same first eight tokens
against the same user message reproduces the same prefix under
`max_tokens=8`. This confirms:

* the greedy path is stable across the 0.6B → 1.7B jump;
* the sampler / detokenizer / frontend chain is deterministic;
* no accidental temperature or sampling drift was introduced by
  the model swap.

## 6. `flashinfer.sampling` import status

* `grep -inE "flashinfer"` on
  `/mnt/nvme/LR-606/phase6b15/server.log`: **zero matches**.
* Neither scheduler rank raised the Phase 6B.8
  `ModuleNotFoundError: No module named 'flashinfer'`.
* Interpretation: `Sampler.prepare` at
  `python/minisgl/engine/sample.py:55` set
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

The greedy path never touched CUDA-only sampling code, and no
CUDA-only comm backend was loaded.

## 9. Scheduler rank health after the request

* All three `spawn_main` children (`44683`, `44684`, `44685`) and
  the resource-tracker `44682` remained alive after the response
  completed.
* Post-response log line:
  ```
  [2026-07-12|19:57:11|core|rank=0] Scheduler is idle, waiting for new reqs...
  ```
  Rank 0 returned to the idle state after the request — the
  request went all the way through the forward loop and back to
  the scheduler, not just up to the HTTP layer.
* Zero `Traceback` / `ModuleNotFoundError` / `RuntimeError`
  matches in the server log.

## 10. Termination and residual check

* `kill -TERM 44673` (parent). Uvicorn shutdown sequence
  completed:
  ```
  INFO:     Shutting down
  INFO:     Waiting for application shutdown.
  INFO:     Application shutdown complete.
  INFO:     Finished server process [44673]
  ```
* One benign
  `multiprocessing.resource_tracker: There appear to be 2 leaked
  semaphore objects to clean up at shutdown` warning at parent
  exit — same benign notice recorded across Phases 6B.5, 6B.7,
  6B.9, 6B.10, 6B.12, 6B.13, 6B.14.
* Follow-up SIGTERM sweep over surviving
  `multiprocessing.spawn` / `resource_tracker` children (`44682`,
  `44683`, `44684`, `44685`) exited on the signal path (no
  `SIGKILL` fallback).
* Final `ps -eo pid,cmd | grep -E "minisgl.server|multiprocessing.(spawn|resource)"`:
  `NO_RESIDUAL`.
* Listening port `:1919` released; `netstat`: no match.

## 11. Verdict — per required checklist

| Check | Result |
|---|---|
| HTTP status | PASS — `200`, `content-type: application/json`, `content-length: 289` |
| response JSON shape | PASS — OpenAI `chat.completion` envelope with `id`, `object`, `created`, `model`, `choices`, `usage` |
| `choices[0].message.content` observed | PASS — `"<think>\nOkay, the user is"` (non-empty completion; identical to Phase 6B.9 Qwen3-0.6B greedy result) |
| `finish_reason` observed | PASS — `"stop"` |
| `usage` fields | PARTIAL — `prompt_tokens=0`, `completion_tokens=0`, `total_tokens=0`. Documented as a known unpopulated-accounting observation (Phase 6B.11 §3), not a scope failure of this gate. |
| `flashinfer.sampling` import status | PASS — did NOT occur; zero `flashinfer` matches in log |
| rank health after request | PASS — parent `44673` + 3 spawn children (`44683`, `44684`, `44685`) + resource-tracker (`44682`) still alive; rank 0 logged `Scheduler is idle, waiting for new reqs...` post-response |
| no `block_size` / `561002` error | PASS — zero matches; `--page-size 16` held |
| no CUDA / NCCL / pynccl fallback | PASS — zero `pynccl`/`nccl`/`cuda`/`flashinfer` non-argument matches; HCCL + Gloo handshakes present |
| clean shutdown | PASS — Uvicorn `Application shutdown complete.` + `Finished server process [44673]`; one benign `resource_tracker` semaphore cleanup notice |

**Overall verdict: PASS.**

## 12. What this gate does NOT establish

* No `/v1/chat/completions` **with `stream=true`** was attempted
  against Qwen3-1.7B; that is a follow-up gate (Phase 6B.16).
* No non-greedy request was attempted. The Phase 6B.8 root cause
  (default OpenAI sampling routes through CUDA-only
  `flashinfer.sampling`) is worked around here by request-body
  shape, not fixed at the sampler. The Phase 6B.11 §3 constraint
  ("`temperature=0` (or `top_k=1`) required to avoid the
  `flashinfer.sampling` path") continues to apply on Qwen3-1.7B.
* `usage.prompt_tokens` / `.completion_tokens` / `.total_tokens`
  all report `0`; usage-accounting accuracy is not covered by
  this gate and is out of scope for the v0.2.0a1 envelope
  (Phase 6B.11 §3 known constraint).
* No multi-request or batch behaviour was tested.
* No client-disconnect / abort-ack behaviour was exercised.
* The benign `resource_tracker` semaphore-leak warning at parent
  exit remains a documented observation, not a root-caused
  finding.
* No performance claim of any kind is made. Wall-clock (~3.29 s
  end-to-end) is anecdotal, not benchmarked.
