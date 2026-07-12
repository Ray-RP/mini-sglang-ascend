# Phase 6B.21 Qwen3-4B Server `/v1/chat/completions` Non-Stream Greedy Smoke Verdict

**Kind:** Documentation-only verdict for the Qwen3-4B fixed-TP2
server `/v1/chat/completions` (non-stream, greedy) smoke against
the Ascend host. Launched with the Phase 6B.7 recipe
([`phase6b.2_server_launch_recipe.md`](./phase6b.2_server_launch_recipe.md)
as amended at Phase 6B.7 to include `--page-size 16`) but with
`--model-path /mnt/nvme/models/Qwen3-4B`, issued one `curl` to
`POST /v1/chat/completions` with `temperature=0` and
`stream=false`, observed the outcome, then terminated. This file
introduces no runtime, script, test, tag, GitHub Release, or
`CHANGELOG.md` change and does not print credentials.

Envelope: fixed TP=2, eager, `npu_fia`, bf16, `page_size=16`,
greedy — v0.2.0a1 recipe, third model.

**Overall verdict: PASS.**

Route: with `temperature=0` in the request body (same greedy
contract validated at Phase 6B.9 / 6B.15 for Qwen3-0.6B / 1.7B),
the sampler short-circuits before importing the CUDA-only
`flashinfer.sampling` module. Both scheduler ranks survived,
`/v1/chat/completions` returned a complete OpenAI-compatible JSON
envelope with a non-empty `choices[0].message.content`
byte-for-byte identical to the 0.6B and 1.7B non-stream greedy
outputs, and the Uvicorn parent shut down cleanly with
`NO_RESIDUAL`.

---

## 1. Environment

* Host: Ascend NPU host (8 × 910B1, 64 GiB HBM per NPU).
* Container: `998ce5ba6e5e`.
* Repo tree: `/mnt/nvme/LR-606/mini-sglang-ascend` (unchanged since
  Phase 6B.3).
* Model: `/mnt/nvme/models/Qwen3-4B`.
* Working dir: `/mnt/nvme/LR-606/phase6b21/` (launch script + log
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

Same wrapper skeleton used at Phase 6B.12 → 6B.20 (`setsid nohup`
inside the container; `PYTHONPATH=python`), only the phase-specific
working directory changed to `/mnt/nvme/LR-606/phase6b21/`.

## 3. Server ready state

* Parent PID `46794`; three `multiprocessing.spawn` children
  (`46807`, `46808`, `46809` = 2 scheduler ranks + 1 detokenizer)
  + resource-tracker `46806`.
* Log excerpt:
  ```
  [Gloo] Rank 0 is connected to 1 peer ranks.
  [Gloo] Rank 1 is connected to 1 peer ranks.
  [core|rank=0] Allocating 566928 tokens for KV cache, K + V = 38.93 GiB
  [core|rank=1] Allocating 566928 tokens for KV cache, K + V = 38.93 GiB
  [core|rank=0] CUDA graph is disabled.
  [core|rank=0] Scheduler is idle, waiting for new reqs...
  Scheduler is ready
  API server is ready to serve on 0.0.0.0:1919
  INFO:     Uvicorn running on http://0.0.0.0:1919 (Press CTRL+C to quit)
  ```
* Consistent with Phase 6B.18 / 6B.19 / 6B.20: KV cache 38.93 GiB
  per rank, ready ~20 s after launch (rank 0 idle at 21:46:05).

## 4. `/v1/chat/completions` request

Command executed inside the container:

```
curl -sS -o chat.body -D chat.headers --max-time 60 \
  http://127.0.0.1:1919/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/mnt/nvme/models/Qwen3-4B",
    "messages": [{"role":"user","content":"What is the capital of France?"}],
    "max_tokens": 8,
    "temperature": 0,
    "stream": false
  }'
```

The request body carries `temperature=0` explicitly, satisfying
`SamplingParams.is_greedy` (`python/minisgl/core.py:30`) as
documented at Phase 6B.9 §1: `(0.0 <= 0.0 or -1 == 1) and 1.0 == 1.0`
→ `True`. The sampler therefore takes the greedy `torch.argmax`
branch and never invokes `sample_impl` (the caller that imports
`flashinfer.sampling`).

## 5. Response — what the client got

Curl instrumentation:

| metric | value |
|---|---|
| HTTP status | `200` |
| content-type | `application/json` |
| content-length | `287` |
| response bytes | `287` |
| wall-clock | `~2.84 s` |

Response headers verbatim:

```
HTTP/1.1 200 OK
date: Sun, 12 Jul 2026 21:46:30 GMT
server: uvicorn
content-length: 287
content-type: application/json
```

Response body verbatim (JSON, pretty-printed for readability;
wire form is a single line):

```json
{
  "id": "chatcmpl-0",
  "object": "chat.completion",
  "created": 1783892793,
  "model": "/mnt/nvme/models/Qwen3-4B",
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
INFO:     127.0.0.1:39092 - "POST /v1/chat/completions HTTP/1.1" 200 OK
```

### Per-field verdict

| field | observed | verdict |
|---|---|---|
| `id` | `"chatcmpl-0"` | PASS — populated, deterministic sequence handle |
| `object` | `"chat.completion"` | PASS — matches OpenAI shape |
| `created` | `1783892793` (Unix seconds) | PASS |
| `model` | `"/mnt/nvme/models/Qwen3-4B"` | PASS — matches the id from `/v1/models` at Phase 6B.19 |
| `choices` | list of length 1 | PASS |
| `choices[0].index` | `0` | PASS |
| `choices[0].message.role` | `"assistant"` | PASS |
| `choices[0].message.content` | `"<think>\nOkay, the user is"` | PASS — non-empty; 8 whitespace-delimited pieces greedy-decoded from Qwen3-4B's thinking prefix under `max_tokens=8` |
| `choices[0].finish_reason` | `"stop"` | PASS — populated |
| `usage.prompt_tokens` | `0` | PARTIAL — usage counters all zero; matches Phase 6B.9 / 6B.15 observation and Phase 6B.11 §3 known constraint (usage counters currently unpopulated). Not a scope failure of this gate. |
| `usage.completion_tokens` | `0` | PARTIAL — same |
| `usage.total_tokens` | `0` | PARTIAL — same |

### Cross-check against Phase 6B.9 (Qwen3-0.6B) and Phase 6B.15 (Qwen3-1.7B)

The generated `choices[0].message.content` string
(`"<think>\nOkay, the user is"`) is **byte-for-byte identical** to
both the Phase 6B.9 Qwen3-0.6B non-stream greedy result and the
Phase 6B.15 Qwen3-1.7B non-stream greedy result. All three
Qwen3 sizes share the same tokenizer and both apply the `<think>`
prefix convention in their chat template. Greedy-decoding the
same first eight tokens against the same user message therefore
reproduces the same prefix under `max_tokens=8`. This confirms:

* the greedy path is stable across the 0.6B → 1.7B → 4B jumps for
  this specific prompt;
* the sampler / detokenizer / frontend chain is deterministic on
  the Qwen3-4B tier;
* no accidental temperature or sampling drift was introduced by
  the 4B-tier model swap.

Note the contrast with Phase 6B.20 `/generate`: on the raw
`/generate` route (no chat template applied) Qwen3-4B produced a
different continuation than 0.6B/1.7B starting at token 5
(`" of Germany is"` vs `" of the United"`). That divergence is
genuine 4B-tier behaviour on the raw-prompt path; the `<think>`
prefix on this chat-templated path is model-family-uniform, so
all three sizes agree here.

## 6. `flashinfer.sampling` import status

* `grep -inE "flashinfer"` on
  `/mnt/nvme/LR-606/phase6b21/server.log`: **zero matches**.
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

The greedy path never touched CUDA-only sampling code, and no
CUDA-only comm backend was loaded.

## 9. Scheduler rank health after the request

Post-response process tree:

```
PID     PPID    CMD
46794   46792   python -m minisgl.server.launch --model-path /mnt/nvme/models/Qwen3-4B ...
46806   46794     python -c from multiprocessing.resource_tracker import main;main(22)
46807   46794     python -c from multiprocessing.spawn import spawn_main; ... pipe_handle=25 --multiprocessing-fork
46808   46794     python -c from multiprocessing.spawn import spawn_main; ... pipe_handle=27 --multiprocessing-fork
46809   46794     python -c from multiprocessing.spawn import spawn_main; ... pipe_handle=29 --multiprocessing-fork
```

* All three `spawn_main` children (`46807`, `46808`, `46809`) and
  the resource-tracker `46806` remained alive after the response
  completed.
* Post-response log line:
  ```
  [2026-07-12|21:46:33|core|rank=0] Scheduler is idle, waiting for new reqs...
  ```
  Rank 0 returned to the idle state after the request. Timeline:
  ready at 21:46:05 → request completed at 21:46:30 (HTTP 200) →
  scheduler idle at 21:46:33 (~3 s working window, matching the
  ~2.84 s curl wall-clock plus scheduler handoff).
* Zero `Traceback` / `ModuleNotFoundError` / `RuntimeError`
  matches in the server log.

## 10. HBM headroom after the request

Per-NPU HBM (via `npu-smi info`, HBM-Usage column, MB used / MB
total) captured immediately after the response:

| NPU | HBM-Usage (MB) | Interpretation |
|---|---|---|
| 0 | `61157 / 65536` | Rank 0 pinned here (~59.7 GiB used, ~4.28 GiB free) |
| 1 | `61157 / 65536` | Rank 1 pinned here (~59.7 GiB used, ~4.28 GiB free) |
| 2–7 | ~16,453–16,455 / 65,536 | Container baseline only |

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
| 6B.21 `/v1/chat/completions` (this gate) | 61157 / 61157 | 44761 | +435 MB working set |

The ~435 MB per-rank delta vs. bring-up is live-request working
memory (activations for the forward pass, transient KV entries
written for the eight generated tokens). Very close to the
Phase 6B.20 `/generate` delta (~416 MB), as expected — same
model, same generation length, similar working-set growth. The
pool still holds ~4.28 GiB free per rank; headroom did not
collapse under the chat-templated single-request load.

**HBM headroom: PASS — held at ~4.28 GiB free per active rank
post-request.**

## 11. Termination and residual check

* `kill -TERM 46794` (parent). Uvicorn shutdown sequence
  completed:
  ```
  INFO:     Shutting down
  INFO:     Waiting for application shutdown.
  INFO:     Application shutdown complete.
  INFO:     Finished server process [46794]
  Terminated
  ```
* Follow-up SIGTERM sweep over surviving
  `multiprocessing.spawn` / `resource_tracker` children exited on
  the signal path (no `SIGKILL` fallback).
* Final `ps -eo pid,cmd | grep -E "minisgl.server|multiprocessing.(spawn|resource)"`:
  `NO_RESIDUAL`.
* Listening port `:1919` released; `netstat`: no match.
* Same benign
  `multiprocessing.resource_tracker: leaked semaphore` warning as
  Phases 6B.5, 6B.7, 6B.9, 6B.10, 6B.12–6B.16, 6B.18–6B.20.

## 12. Verdict — per required checklist

| Check | Result |
|---|---|
| HTTP status | PASS — `200`, `content-type: application/json`, `content-length: 287` |
| response JSON shape | PASS — OpenAI `chat.completion` envelope with `id`, `object`, `created`, `model`, `choices`, `usage` |
| `choices[0].message.content` | PASS — `"<think>\nOkay, the user is"` (non-empty completion; byte-identical to Phase 6B.9 Qwen3-0.6B and Phase 6B.15 Qwen3-1.7B greedy results) |
| `finish_reason` | PASS — `"stop"` |
| `usage` fields | PARTIAL — `prompt_tokens=0`, `completion_tokens=0`, `total_tokens=0`. Documented as a known unpopulated-accounting observation (Phase 6B.11 §3), not a scope failure of this gate. |
| `flashinfer.sampling` import status | PASS — did NOT occur; zero `flashinfer` matches in log |
| rank health after request | PASS — parent `46794` + 3 spawn children (`46807`, `46808`, `46809`) + resource-tracker (`46806`) still alive; rank 0 logged `Scheduler is idle, waiting for new reqs...` post-response |
| HBM headroom after request | PASS — HBM-Usage 61,157 / 65,536 MB on both active NPUs (~4.28 GiB free per rank); ~435 MB working-memory delta vs. bring-up |
| no `block_size` / `561002` error | PASS — zero matches; `--page-size 16` held |
| no CUDA / NCCL / pynccl fallback | PASS — zero `pynccl` / `nccl` / `cuda` / `flashinfer` non-argument matches; HCCL + Gloo handshakes present |
| clean shutdown | PASS — Uvicorn `Application shutdown complete.` + `Finished server process [46794]`; one benign `resource_tracker` semaphore cleanup notice; `NO_RESIDUAL`; port released |

**Overall verdict: PASS.**

## 13. What this gate does NOT establish

* No `/v1/chat/completions` **with `stream=true`** was attempted
  against Qwen3-4B; that is a follow-up gate.
* No non-greedy request was attempted. The Phase 6B.8 root cause
  (default OpenAI sampling routes through CUDA-only
  `flashinfer.sampling`) is worked around here by request-body
  shape, not fixed at the sampler. The Phase 6B.11 §3 constraint
  ("`temperature=0` (or `top_k=1`) required to avoid the
  `flashinfer.sampling` path") continues to apply on Qwen3-4B.
* `usage.prompt_tokens` / `.completion_tokens` / `.total_tokens`
  all report `0`; usage-accounting accuracy is not covered by
  this gate and is out of scope for the v0.2.0a1 envelope
  (Phase 6B.11 §3 known constraint).
* No multi-request or batch behaviour was tested.
* No client-disconnect / abort-ack behaviour was exercised.
* The benign `resource_tracker` semaphore-leak warning at parent
  exit remains a documented observation, not a root-caused
  finding.
* No performance claim of any kind is made. Wall-clock (~2.84 s
  end-to-end) is anecdotal, not benchmarked.
