# Phase 6B.11 Qwen3-0.6B Server Path Capability Summary

**Kind:** Documentation-only summary of the Ascend fixed-TP2
Qwen3-0.6B server-path bring-up work from Phase 6B.3 through
Phase 6B.10. This file introduces no runtime, script, test, tag,
GitHub Release, or `CHANGELOG.md` change; it launches no server,
calls no endpoint, and prints no credentials. Its sole purpose is
to consolidate what the eight preceding gates established, so any
follow-up model bring-up (e.g. Qwen3-1.7B) starts from an
unambiguous baseline.

Envelope: fixed TP=2, eager, `npu_fia`, bf16, `page_size=16`,
greedy — v0.2.0a1.

**Overall verdict: PARTIAL.** Every readiness route (metadata,
`/generate` SSE, and both non-stream/stream `/v1/chat/completions`)
passes under the greedy contract. One non-greedy sampling path
remains blocked by a missing CUDA-only dependency (`flashinfer`).
Two accuracy observations (`usage` counters, model-id alias) are
not yet closed but do not block the envelope.

---

## 1. Launch envelope (locked)

Frozen at Phase 6B.2
([`phase6b.2_server_launch_recipe.md`](./phase6b.2_server_launch_recipe.md)),
amended once at Phase 6B.7 to add `--page-size 16`. Required
flags:

```
--tp-size 2
--attention-backend npu_fia
--disable-pynccl
--cuda-graph-max-bs 0
--page-size 16
```

Preflight: `prompt_toolkit` must be importable by the server
process. Already declared as a base runtime dependency in
`pyproject.toml` (Phase 6B.4). Environments provisioned before
that declaration was merged must install it explicitly
(`pip install prompt_toolkit`).

Launch skeleton (redacted, `<PORT>` chosen at bring-up time):

```
python -m minisgl.server.launch \
  --model-path /mnt/nvme/models/Qwen3-0.6B \
  --tp-size 2 \
  --attention-backend npu_fia \
  --disable-pynccl \
  --cuda-graph-max-bs 0 \
  --page-size 16 \
  --host 0.0.0.0 \
  --port <PORT>
```

Rationale for each flag lives in Phase 6B.2 §3 (recipe) and Phase
6B.7 §2 (page-size amendment). Nothing new is decided here.

## 2. Endpoint matrix

| Route | Request shape | Verdict | Evidence gate |
|---|---|---|---|
| Server bring-up (HCCL + Gloo, `Uvicorn running`, `netstat` LISTEN) | — | **PASS** | [`phase6b.3_qwen06b_server_bringup_smoke_verdict.md`](./phase6b.3_qwen06b_server_bringup_smoke_verdict.md) |
| `GET /v1/models` | — | **PASS** | [`phase6b.5_qwen06b_v1_models_smoke_verdict.md`](./phase6b.5_qwen06b_v1_models_smoke_verdict.md) |
| `POST /generate` (SSE) | `prompt`, `max_tokens=8`, `ignore_eos=true`, greedy defaults | **PASS** (after `--page-size 16` fix) | [`phase6b.6_qwen06b_generate_sse_smoke_verdict.md`](./phase6b.6_qwen06b_generate_sse_smoke_verdict.md) (BLOCKED) → [`phase6b.7_qwen06b_generate_sse_page16_verdict.md`](./phase6b.7_qwen06b_generate_sse_page16_verdict.md) (PASS) |
| `POST /v1/chat/completions` `stream=false`, `temperature=0` | greedy override | **PASS** | [`phase6b.9_qwen06b_chat_nonstream_greedy_verdict.md`](./phase6b.9_qwen06b_chat_nonstream_greedy_verdict.md) |
| `POST /v1/chat/completions` `stream=true`, `temperature=0` | greedy override | **PASS** | [`phase6b.10_qwen06b_chat_stream_greedy_verdict.md`](./phase6b.10_qwen06b_chat_stream_greedy_verdict.md) |
| `POST /v1/chat/completions` `stream=false`, default sampling | `temperature=1.0` (OpenAI default) | **BLOCKED** | [`phase6b.8_qwen06b_chat_nonstream_smoke_verdict.md`](./phase6b.8_qwen06b_chat_nonstream_smoke_verdict.md) |

Cross-gate cross-check: the streaming and non-streaming greedy
chat paths produced the exact same generated prefix
(`"<think>\nOkay, the user is"` under
`max_tokens=8, temperature=0`), confirming both routes share the
same greedy sampler pathway.

## 3. Known constraints (must be honoured by any client of the
   server path)

1. **`model` field must equal the `/v1/models` returned id.**
   The server reports its model by absolute filesystem path
   (Phase 6B.5 §4: `"id": "/mnt/nvme/models/Qwen3-0.6B"`). Short
   aliases are not resolved. Any `/v1/chat/completions` request
   must send that exact string.
2. **`--page-size 16` is required for `npu_fia`.** Confirmed at
   Phase 6B.7. The default `page_size=1` is rejected by the CANN
   `aclnnFusedInferAttentionScoreV3` no-quant BF16 kernel
   (`CheckFeatureNoquantBlockSize` at
   `fused_infer_attention_score_tiling_check_feature.cpp:159`)
   with error `561002`
   (`In NO_QUANT situation, block_size should aligned to 16, but got 1`).
3. **`temperature=0` (or `top_k=1`) is required to avoid the
   `flashinfer.sampling` path.** Confirmed at Phase 6B.9 §1.
   `SamplingParams.is_greedy`
   (`python/minisgl/core.py:30`) evaluates
   `(temperature <= 0.0 or top_k == 1) and top_p == 1.0`.
   Requests that fail this predicate route through
   `sample_impl` (`python/minisgl/engine/sample.py:30`), which
   imports the CUDA-only `flashinfer.sampling` module and
   crashes on NPU.
4. **`usage` token counters are currently unpopulated.** Phase
   6B.9 §6 recorded `prompt_tokens=0`, `completion_tokens=0`,
   `total_tokens=0` on the non-stream chat response. The
   `usage` block is present and correctly typed but the counters
   are not accounted. Not fixed in this envelope.
5. **Stream chunk envelope is used as-observed.** The streaming
   `/v1/chat/completions` chunk shape is
   `{"id":"cmpl-0","object":"text_completion.chunk","choices":[{"delta":...,"index":0,"finish_reason":...}]}`
   with a trailing `data: [DONE]` (Phase 6B.10 §5). No `usage`
   block is emitted at the tail. Client code must accept this
   envelope verbatim; do not assume additional OpenAI fields.

## 4. Known blocked path

**`POST /v1/chat/completions` with default OpenAI sampling
(`temperature=1.0`, `top_k=-1`, `top_p=1.0`)** — BLOCKED as of
Phase 6B.8. Both scheduler ranks crash with
`ModuleNotFoundError: No module named 'flashinfer'` inside
`sample_impl` at `python/minisgl/engine/sample.py:30`. Root cause
is that non-greedy sampling on this build routes through a
CUDA-only sampling library; `flashinfer` is not installed on the
NPU container by design.

Resolution options (recorded here for completeness, all out of
scope for the current envelope):

* **Recipe-level constraint.** Require `temperature=0` (or
  `top_k=1`) in every request body — the workaround already used
  at Phase 6B.9 and Phase 6B.10.
* **NPU-compatible sampling backend.** Introduce an NPU-native
  path so `sample_impl` no longer needs `flashinfer`.
* **Optional-import guard.** Gate the `flashinfer.sampling`
  import behind a device check and fall back to a CUDA-free
  reference implementation.

Only the recipe-level constraint has been exercised; the other
two require runtime code changes and were explicitly excluded
from the v0.2.0a1 envelope.

## 5. Cleanup / invariant summary

Across Phases 6B.3, 6B.5, 6B.6, 6B.7, 6B.8, 6B.9, and 6B.10:

* Every gate confirmed HCCL + Gloo (`ProcessGroupHCCL` watchdog
  warnings + `[Gloo] Rank {0,1} is connected to 1 peer ranks.`)
  as the sole communication backend.
* Every gate reported zero external `pynccl` / `nccl` matches,
  and zero external `cuda` matches outside the deliberate
  `CUDA graph is disabled.` notice.
* Every PASS gate ended with `NO_RESIDUAL` after SIGTERM and a
  released port; the two BLOCKED gates (6B.6, 6B.8) also
  reached `NO_RESIDUAL` on shutdown despite mid-run scheduler
  crashes.
* A benign `multiprocessing.resource_tracker` leaked-semaphore
  warning at parent exit is documented across gates and remains
  a known observation, not a root-caused finding.

## 6. Next recommended gate

**Qwen3-1.7B server bring-up with the same recipe.**

Concretely: repeat the Phase 6B.3 → 6B.10 arc against
`/mnt/nvme/models/Qwen3-1.7B` (or the actual model directory), with
no changes to launch flags. Expected outcomes:

* `/v1/models` reports the new model's absolute path as its `id`.
* `/generate` SSE returns a non-empty completion under `page_size=16`.
* Both `/v1/chat/completions` greedy paths (non-stream and
  stream) succeed with `temperature=0`.
* The non-greedy `/v1/chat/completions` path remains BLOCKED for
  the same reason as Phase 6B.8; that is out of scope until the
  sampler is patched.

If Qwen3-1.7B lands cleanly, the same envelope is a viable
candidate for the next model up (Qwen3-4B or the multi-billion
tier), gated by memory headroom on the 8 × 910B1 host.

## 7. What this summary does NOT establish

* No new experiment was run for this gate — every claim above is
  reference-only, cited to the source verdict file.
* No client-disconnect / abort-ack behaviour (Gate 2.3f) has
  been exercised against the server path.
* No multi-request, batch, or long-context behaviour has been
  tested.
* No performance claim of any kind is made.
