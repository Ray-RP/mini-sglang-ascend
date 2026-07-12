# Phase 6B.23 Qwen3-4B Server Path Capability Summary

**Kind:** Documentation-only summary of the Ascend fixed-TP2
Qwen3-4B server-path bring-up work from Phase 6B.18 through
Phase 6B.22. This file introduces no runtime, script, test, tag,
GitHub Release, or `CHANGELOG.md` change; it launches no server,
calls no endpoint, and prints no credentials. Its sole purpose is
to consolidate what the five preceding gates established for the
4B tier so any follow-up work (a three-model roll-up or a
next-model bring-up) starts from an unambiguous baseline.

Envelope: fixed TP=2, eager, `npu_fia`, bf16, `page_size=16`,
greedy — v0.2.0a1.

**Overall verdict: PASS.** Every readiness route (bring-up,
metadata, `/generate` SSE, and both non-stream/stream
`/v1/chat/completions`) passes under the greedy contract on
Qwen3-4B. One non-greedy sampling path remains expected-blocked by
the same CUDA-only `flashinfer.sampling` dependency documented at
Phase 6B.8 / Phase 6B.11 §3–4. One accuracy observation (`usage`
token counters) is not closed but does not block the envelope.

The Qwen3-4B arc is byte-for-byte consistent with the Qwen3-0.6B
arc (Phase 6B.11) and the Qwen3-1.7B arc (Phase 6B.17) on the
chat-templated greedy path: same greedy-decoded prefix
(`"<think>\nOkay, the user is"`) on both stream and non-stream
chat, same envelope shapes, same rank behaviour, same cleanup
signature. The recipe generalizes across the
0.6B → 1.7B → 4B jump without any code change.

---

## 1. Launch envelope (locked)

Frozen at Phase 6B.2
([`phase6b.2_server_launch_recipe.md`](./phase6b.2_server_launch_recipe.md)),
amended once at Phase 6B.7 to add `--page-size 16`. Required
flags for Qwen3-4B on the Ascend host are identical to the
Qwen3-0.6B / 1.7B envelopes; only the model path changes:

```
--tp-size 2
--attention-backend npu_fia
--disable-pynccl
--cuda-graph-max-bs 0
--page-size 16
```

Launch skeleton (redacted, `<PORT>` chosen at bring-up time):

```
python -m minisgl.server.launch \
  --model-path /mnt/nvme/models/Qwen3-4B \
  --tp-size 2 \
  --attention-backend npu_fia \
  --disable-pynccl \
  --cuda-graph-max-bs 0 \
  --page-size 16 \
  --host 0.0.0.0 \
  --port <PORT>
```

Preflight: same `prompt_toolkit` runtime-dependency preflight as
Phase 6B.11 §1 / Phase 6B.17 §1 (declared in `pyproject.toml` at
Phase 6B.4; environments provisioned before that merge must
`pip install prompt_toolkit`). No new preflight introduced by the
4B tier.

Rationale for each flag lives in Phase 6B.2 §3 (recipe) and Phase
6B.7 §2 (page-size amendment). Nothing new is decided here.

Observed resource footprint (recorded at Phase 6B.18 §5 and
reproduced across 6B.19–6B.22):

* NPU 0 + NPU 1 only — TP=2 pins ranks 1-to-1 (rank 0 → NPU 0,
  rank 1 → NPU 1); NPUs 2–7 remain at container baseline
  (~16.4 GiB HBM).
* KV cache 38.93 GiB per rank (566,928 tokens) — smaller than the
  0.6B / 1.7B tiers' 41.27 GiB (772,720 tokens) because the KV
  cache autosizer absorbs the larger 4B weight-tensor delta while
  holding `Free memory after initialization` at ~4.74 GiB per
  rank (driven by `memory_ratio=0.9`).
* ~44.3 GiB HBM `proc-mem` per rank at bring-up (scheduler rank),
  +13.09 GiB HCCL sidecar, ~4.74 GiB free per rank post-init.
* HBM-Usage per active NPU ~60.7 GiB / 64 GiB at bring-up
  (~92% pressure) — same envelope as Qwen3-1.7B.
* Parse-to-ready wall-clock ~20 s, same as Qwen3-0.6B / 1.7B
  despite the larger weight tensor (~2 s weight-load window
  inside the 20 s total; init, HCCL handshake, and KV allocation
  dominate).

## 2. Endpoint matrix

| Route | Request shape | Verdict | Evidence gate |
|---|---|---|---|
| Server bring-up (HCCL + Gloo, `Uvicorn running`, `netstat` LISTEN, per-rank NPU pinning, HBM headroom watch item from Phase 6B.17 §6) | — | **PASS** | [`phase6b.18_qwen4b_server_bringup_smoke_verdict.md`](./phase6b.18_qwen4b_server_bringup_smoke_verdict.md) |
| `GET /v1/models` | — | **PASS** | [`phase6b.19_qwen4b_v1_models_smoke_verdict.md`](./phase6b.19_qwen4b_v1_models_smoke_verdict.md) |
| `POST /generate` (SSE) | `prompt`, `max_tokens=8`, `ignore_eos=true`, greedy defaults | **PASS** | [`phase6b.20_qwen4b_generate_sse_smoke_verdict.md`](./phase6b.20_qwen4b_generate_sse_smoke_verdict.md) |
| `POST /v1/chat/completions` `stream=false`, `temperature=0` | greedy override | **PASS** | [`phase6b.21_qwen4b_chat_nonstream_greedy_verdict.md`](./phase6b.21_qwen4b_chat_nonstream_greedy_verdict.md) |
| `POST /v1/chat/completions` `stream=true`, `temperature=0` | greedy override | **PASS** | [`phase6b.22_qwen4b_chat_stream_greedy_verdict.md`](./phase6b.22_qwen4b_chat_stream_greedy_verdict.md) |

Cross-gate cross-check:

* The streaming and non-streaming greedy chat paths produced the
  exact same generated prefix (`"<think>\nOkay, the user is"`
  under `max_tokens=8, temperature=0`), confirming both routes
  share the same greedy sampler pathway on Qwen3-4B.
* That same prefix matches Phase 6B.9 / 6B.10 (Qwen3-0.6B
  non-stream / stream chat) and Phase 6B.15 / 6B.16 (Qwen3-1.7B
  non-stream / stream chat) byte-for-byte, confirming the greedy
  sampler is deterministic and stable across the 0.6B → 1.7B → 4B
  jump on this chat-templated prompt.
* `/generate` on Qwen3-4B returned
  `" Paris. The capital of Germany is"` under `max_tokens=8,
  ignore_eos=true` (Phase 6B.20 §5) — semantically correct first
  token (`" Paris"`) then a plausible sentence stem. This diverges
  from the 0.6B / 1.7B `/generate` output (`" Paris. The capital
  of the United"`) starting at token 5 — genuine 4B-tier behaviour
  on the raw (non-chat-templated) path, not a sampler drift; the
  greedy `torch.argmax` codepath is shared across all three model
  sizes (see §3 constraint 3).
* Every 6B.18–6B.22 gate ended with `NO_RESIDUAL`, port released,
  and the same benign `resource_tracker` semaphore-leak notice at
  parent exit — the cleanup signature is identical to the
  Qwen3-0.6B / 1.7B arcs summarized at Phase 6B.11 §5 / Phase
  6B.17 §5.
* HBM working-set growth across the arc:
  bring-up 60723/60722 MB → `/v1/models` 60722/60722 MB (no change,
  frontend-only route) → `/generate` 61139/61138 MB (+416 MB per
  rank) → `/v1/chat/completions` non-stream 61157/61157 MB
  (+435 MB) → `/v1/chat/completions` stream 61157/61157 MB
  (+435 MB, identical to non-stream — streaming affects only the
  frontend serialisation, not the working-memory footprint). The
  pool held ~4.28 GiB free per rank under live-request load; no
  leak.

## 3. Known constraints (must be honoured by any client of the
   server path on Qwen3-4B)

Same four constraints as the Qwen3-0.6B / 1.7B envelopes
(Phase 6B.11 §3 / Phase 6B.17 §3), re-confirmed against Qwen3-4B
evidence:

1. **`model` field must equal
   `/mnt/nvme/models/Qwen3-4B`.** The server reports its model by
   absolute filesystem path (Phase 6B.19 §5:
   `"id": "/mnt/nvme/models/Qwen3-4B"`). Short aliases are not
   resolved. Any `/v1/chat/completions` request must send that
   exact string.
2. **`--page-size 16` is required for `npu_fia`.** Same CANN
   constraint documented at Phase 6B.7. The default `page_size=1`
   would be rejected by the BF16 no-quant kernel with error
   `561002`. Every Qwen3-4B gate confirmed zero `block_size` /
   `561002` / `CheckFeatureNoquant` matches under `--page-size 16`.
3. **`temperature=0` (or `top_k=1`) is required to avoid the
   `flashinfer.sampling` path.** `SamplingParams.is_greedy`
   (`python/minisgl/core.py:30`) evaluates
   `(temperature <= 0.0 or top_k == 1) and top_p == 1.0`.
   Requests that fail this predicate route through `sample_impl`
   (`python/minisgl/engine/sample.py:30`), which imports the
   CUDA-only `flashinfer.sampling` module and would crash both
   scheduler ranks (Phase 6B.8 root cause). Phase 6B.21 §6 and
   Phase 6B.22 §6 confirmed zero `flashinfer` matches when
   `temperature=0` is honoured on Qwen3-4B (non-stream and stream
   chat). Phase 6B.20 §6 confirmed the `/generate` default
   `SamplingParams` (`temperature=0.0`, `top_k=-1`, `top_p=1.0`)
   also satisfy the greedy predicate.
4. **`usage` token counters are currently unpopulated.** Phase
   6B.21 §5 recorded `prompt_tokens=0`, `completion_tokens=0`,
   `total_tokens=0` on the non-stream chat response for
   Qwen3-4B, matching the Phase 6B.9 (0.6B) / Phase 6B.15 (1.7B)
   observations. The `usage` block is present and correctly typed
   in the non-stream envelope but the counters are not accounted.
   The streaming chunk envelope (Phase 6B.22 §5) omits `usage`
   entirely, same as Phase 6B.10 (0.6B) / Phase 6B.16 (1.7B). Not
   fixed in this envelope.

The stream chunk envelope shape (Phase 6B.11 §3 item 5 / Phase
6B.17 §3) also carries over unchanged — the Phase 6B.22 body used
the same
`{"id":"cmpl-0","object":"text_completion.chunk","choices":[{"delta":...,"index":0,"finish_reason":...}]}`
+ trailing `data: [DONE]` shape as Phase 6B.10 / Phase 6B.16, with
no `usage` tail.

## 4. Known blocked path

**`POST /v1/chat/completions` with default OpenAI sampling
(`temperature=1.0`, `top_k=-1`, `top_p=1.0`) on Qwen3-4B** —
**expected BLOCKED**. Not exercised in the Phase 6B.18–6B.22 arc,
but the root cause is device-level (CUDA-only `flashinfer.sampling`
import in `sample_impl` at `python/minisgl/engine/sample.py:30`),
not model-specific; the failure documented at Phase 6B.8 for
Qwen3-0.6B is expected to reproduce identically on Qwen3-4B if
the request body drops the `temperature=0` override.

Resolution options (unchanged from Phase 6B.11 §4 / Phase 6B.17
§4, all out of scope for the v0.2.0a1 envelope):

* **Recipe-level constraint.** Require `temperature=0` (or
  `top_k=1`) in every request body — the workaround already used
  at Phase 6B.21 and Phase 6B.22.
* **NPU-compatible sampling backend.** Introduce an NPU-native
  path so `sample_impl` no longer needs `flashinfer`.
* **Optional-import guard.** Gate the `flashinfer.sampling`
  import behind a device check and fall back to a CUDA-free
  reference implementation.

Only the recipe-level constraint has been exercised on Qwen3-4B;
the other two require runtime code changes and remain explicitly
excluded from the v0.2.0a1 envelope.

## 5. Cleanup / invariant summary

Across Phases 6B.18, 6B.19, 6B.20, 6B.21, and 6B.22:

* Every gate confirmed HCCL + Gloo (`ProcessGroupHCCL` watchdog
  warnings + `[Gloo] Rank {0,1} is connected to 1 peer ranks.`)
  as the sole communication backend.
* Every gate reported zero external `pynccl` / `nccl` matches,
  and zero external `cuda` matches outside the deliberate
  `CUDA graph is disabled.` notice.
* Every gate that exercised the sampler (6B.20, 6B.21, 6B.22)
  reported zero external `flashinfer` matches — the greedy
  contract held on both `/generate` (default greedy
  `SamplingParams`) and `/v1/chat/completions` (`temperature=0`
  override, non-stream and stream).
* Every gate reported zero `block_size` / `561002` /
  `CheckFeatureNoquant` matches — `--page-size 16` held for the
  BF16 no-quant path.
* Every gate ended with `NO_RESIDUAL` after SIGTERM and a
  released port; no `SIGKILL` fallback was needed on any gate.
* A benign `multiprocessing.resource_tracker` leaked-semaphore
  warning at parent exit remains the same documented observation
  as in the Qwen3-0.6B / 1.7B arcs (Phase 6B.11 §5 / Phase 6B.17
  §5), not a root-caused finding.

## 6. Next recommended gate

**Three-model server-path capability summary.**

Concretely: a documentation-only roll-up that consolidates the
Qwen3-0.6B (Phase 6B.11), Qwen3-1.7B (Phase 6B.17), and Qwen3-4B
(this file) server-path summaries into a single cross-model
capability table for the v0.2.0a1 technical preview. Expected
shape:

* Same launch envelope holds across all three tiers, with model
  path being the only per-model variable (no `--memory-ratio` or
  KV-cache-cap amendment needed).
* Same endpoint matrix (bring-up, `/v1/models`, `/generate` SSE,
  `/v1/chat/completions` non-stream + stream) PASSes on all three
  tiers under the greedy contract.
* Same four constraints (absolute model path, `--page-size 16`,
  `temperature=0`, unpopulated `usage`) apply uniformly.
* Same expected-BLOCKED non-greedy chat path applies uniformly.

Watch items specifically for the roll-up:

* The three tiers share the byte-identical chat-templated greedy
  prefix (`"<think>\nOkay, the user is"`) on the chat routes — a
  strong invariant to lead the roll-up with.
* On the raw `/generate` path the 0.6B / 1.7B agree
  (`" Paris. The capital of the United"`) while 4B diverges
  (`" Paris. The capital of Germany is"`) — an expected model-tier
  behaviour, not a bug; the roll-up should call it out
  explicitly so readers do not confuse it with sampler drift.
* HBM headroom pressure grows with model size: 0.6B / 1.7B run
  with ~5 GiB free per rank at bring-up while 4B runs at ~4.74 GiB
  free per rank (~92% HBM-Usage pressure). The 4B tier is the
  ceiling on this envelope for KV cache autosizing; the next
  larger model would need a recipe amendment to fit.

If the roll-up lands cleanly, the next-model bring-up (a tier
above 4B) becomes the natural follow-up — but that is gated by
memory headroom on the 8 × 910B1 host and would land as a
**recipe amendment** (e.g. `--memory-ratio` adjustment or a
smaller KV-cache-token cap), not as a routine bring-up.

## 7. What this summary does NOT establish

* No new experiment was run for this gate — every claim above is
  reference-only, cited to the source verdict file.
* No client-disconnect / abort-ack behaviour has been exercised
  against the Qwen3-4B server path.
* No multi-request, batch, or long-context behaviour has been
  tested on Qwen3-4B.
* No non-greedy sampling has been exercised on Qwen3-4B; §4 is an
  inference from the device-level Phase 6B.8 root cause, not an
  experimentally-observed failure on this model.
* No `usage` accounting fix has been validated; the constraint
  from §3 item 4 remains open.
* Whether the 4B tier's ~92% HBM pressure leaves enough headroom
  for a live long-context request (KV cache would grow beyond its
  pre-allocated 566,928-token pool for a single very long
  sequence) is not covered; the pre-allocated pool is the
  operational bound.
* No performance claim of any kind is made.
