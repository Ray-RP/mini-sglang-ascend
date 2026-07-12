# Phase 6B.17 Qwen3-1.7B Server Path Capability Summary

**Kind:** Documentation-only summary of the Ascend fixed-TP2
Qwen3-1.7B server-path bring-up work from Phase 6B.12 through
Phase 6B.16. This file introduces no runtime, script, test, tag,
GitHub Release, or `CHANGELOG.md` change; it launches no server,
calls no endpoint, and prints no credentials. Its sole purpose is
to consolidate what the five preceding gates established for the
1.7B tier so any follow-up model bring-up (e.g. Qwen3-4B) starts
from an unambiguous baseline.

Envelope: fixed TP=2, eager, `npu_fia`, bf16, `page_size=16`,
greedy — v0.2.0a1.

**Overall verdict: PASS.** Every readiness route (metadata,
`/generate` SSE, and both non-stream/stream `/v1/chat/completions`)
passes under the greedy contract on Qwen3-1.7B. One non-greedy
sampling path remains expected-blocked by the same CUDA-only
`flashinfer.sampling` dependency documented at Phase 6B.8 /
Phase 6B.11 §3–4. One accuracy observation (`usage` token
counters) is not closed but does not block the envelope.

The Qwen3-1.7B arc is byte-for-byte consistent with the Qwen3-0.6B
arc summarized at Phase 6B.11: same greedy-decoded prefix
(`"<think>\nOkay, the user is"`) on both stream and non-stream
chat, same envelope shapes, same rank behaviour, same cleanup
signature. The recipe generalizes across the 0.6B → 1.7B jump
without any code change.

---

## 1. Launch envelope (locked)

Frozen at Phase 6B.2
([`phase6b.2_server_launch_recipe.md`](./phase6b.2_server_launch_recipe.md)),
amended once at Phase 6B.7 to add `--page-size 16`. Required
flags for Qwen3-1.7B on the Ascend host are identical to the
Qwen3-0.6B envelope; only the model path changes:

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
  --model-path /mnt/nvme/models/Qwen3-1.7B \
  --tp-size 2 \
  --attention-backend npu_fia \
  --disable-pynccl \
  --cuda-graph-max-bs 0 \
  --page-size 16 \
  --host 0.0.0.0 \
  --port <PORT>
```

Preflight: same `prompt_toolkit` runtime-dependency preflight as
Phase 6B.11 §1 (declared in `pyproject.toml` at Phase 6B.4;
environments provisioned before that merge must
`pip install prompt_toolkit`). No new preflight introduced by the
1.7B tier.

Rationale for each flag lives in Phase 6B.2 §3 (recipe) and Phase
6B.7 §2 (page-size amendment). Nothing new is decided here.

Observed resource footprint (recorded at Phase 6B.12 §3 and
reproduced across 6B.13–6B.16):

* NPU 0 + NPU 1 only — TP=2 pins ranks 1-to-1 (rank 0 → NPU 0,
  rank 1 → NPU 1); NPUs 2–7 remain at container baseline
  (~16.4 GiB HBM).
* KV cache 41.27 GiB per rank (772,720 tokens), ~44 GiB HBM
  `proc-mem` per rank, ~4.74 GiB free per rank post-init.
* Parse-to-ready wall-clock ~20 s, same as Qwen3-0.6B.

## 2. Endpoint matrix

| Route | Request shape | Verdict | Evidence gate |
|---|---|---|---|
| Server bring-up (HCCL + Gloo, `Uvicorn running`, `netstat` LISTEN, per-rank NPU pinning) | — | **PASS** | [`phase6b.12_qwen17b_server_bringup_smoke_verdict.md`](./phase6b.12_qwen17b_server_bringup_smoke_verdict.md) |
| `GET /v1/models` | — | **PASS** | [`phase6b.13_qwen17b_v1_models_smoke_verdict.md`](./phase6b.13_qwen17b_v1_models_smoke_verdict.md) |
| `POST /generate` (SSE) | `prompt`, `max_tokens=8`, `ignore_eos=true`, greedy defaults | **PASS** | [`phase6b.14_qwen17b_generate_sse_smoke_verdict.md`](./phase6b.14_qwen17b_generate_sse_smoke_verdict.md) |
| `POST /v1/chat/completions` `stream=false`, `temperature=0` | greedy override | **PASS** | [`phase6b.15_qwen17b_chat_nonstream_greedy_verdict.md`](./phase6b.15_qwen17b_chat_nonstream_greedy_verdict.md) |
| `POST /v1/chat/completions` `stream=true`, `temperature=0` | greedy override | **PASS** | [`phase6b.16_qwen17b_chat_stream_greedy_verdict.md`](./phase6b.16_qwen17b_chat_stream_greedy_verdict.md) |

Cross-gate cross-check:

* The streaming and non-streaming greedy chat paths produced the
  exact same generated prefix (`"<think>\nOkay, the user is"`
  under `max_tokens=8, temperature=0`), confirming both routes
  share the same greedy sampler pathway on Qwen3-1.7B.
* That same prefix matches Phase 6B.9 (Qwen3-0.6B non-stream) and
  Phase 6B.10 (Qwen3-0.6B stream) byte-for-byte, confirming the
  greedy sampler is deterministic and stable across the
  0.6B → 1.7B jump on this prompt.
* `/generate` on Qwen3-1.7B returned
  `" Paris. The capital of the United"` under `max_tokens=8,
  ignore_eos=true` (Phase 6B.14 §5), a semantically correct first
  token to the prompt `"The capital of France is"` — evidence
  the model weights are being consumed correctly, not just the
  frontend/scheduler chain.
* Every 6B.12–6B.16 gate ended with `NO_RESIDUAL`, port
  released, and the same benign
  `resource_tracker` semaphore-leak notice at parent exit — the
  cleanup signature is identical to the Qwen3-0.6B arc summarized
  at Phase 6B.11 §5.

## 3. Known constraints (must be honoured by any client of the
   server path on Qwen3-1.7B)

Same four constraints as the Qwen3-0.6B envelope (Phase 6B.11 §3),
re-confirmed against Qwen3-1.7B evidence:

1. **`model` field must equal
   `/mnt/nvme/models/Qwen3-1.7B`.** The server reports its model
   by absolute filesystem path (Phase 6B.13 §5:
   `"id": "/mnt/nvme/models/Qwen3-1.7B"`). Short aliases are not
   resolved. Any `/v1/chat/completions` request must send that
   exact string.
2. **`--page-size 16` is required for `npu_fia`.** Same CANN
   constraint documented at Phase 6B.7. The default `page_size=1`
   would be rejected by the BF16 no-quant kernel with error
   `561002`. Every Qwen3-1.7B gate confirmed zero
   `block_size` / `561002` / `CheckFeatureNoquant` matches under
   `--page-size 16`.
3. **`temperature=0` (or `top_k=1`) is required to avoid the
   `flashinfer.sampling` path.** `SamplingParams.is_greedy`
   (`python/minisgl/core.py:30`) evaluates
   `(temperature <= 0.0 or top_k == 1) and top_p == 1.0`.
   Requests that fail this predicate route through
   `sample_impl` (`python/minisgl/engine/sample.py:30`), which
   imports the CUDA-only `flashinfer.sampling` module and would
   crash both scheduler ranks (Phase 6B.8 root cause). Phase 6B.15
   §6 and Phase 6B.16 §6 confirmed zero `flashinfer` matches when
   `temperature=0` is honoured on Qwen3-1.7B.
4. **`usage` token counters are currently unpopulated.** Phase
   6B.15 §5 recorded `prompt_tokens=0`, `completion_tokens=0`,
   `total_tokens=0` on the non-stream chat response for
   Qwen3-1.7B, matching the Phase 6B.9 Qwen3-0.6B observation.
   The `usage` block is present and correctly typed but the
   counters are not accounted. The streaming chunk envelope
   (Phase 6B.16 §5) omits `usage` entirely, same as Phase 6B.10.
   Not fixed in this envelope.

The stream chunk envelope shape (Phase 6B.11 §3 item 5) also
carries over unchanged — the Phase 6B.16 body used the same
`{"id":"cmpl-0","object":"text_completion.chunk","choices":[{"delta":...,"index":0,"finish_reason":...}]}`
+ trailing `data: [DONE]` shape as Phase 6B.10, with no `usage`
tail.

## 4. Known blocked path

**`POST /v1/chat/completions` with default OpenAI sampling
(`temperature=1.0`, `top_k=-1`, `top_p=1.0`) on Qwen3-1.7B** —
**expected BLOCKED**. Not exercised in the Phase 6B.12–6B.16 arc,
but the root cause is device-level (CUDA-only `flashinfer.sampling`
import in `sample_impl` at `python/minisgl/engine/sample.py:30`),
not model-specific; the failure documented at Phase 6B.8 for
Qwen3-0.6B is expected to reproduce identically on Qwen3-1.7B if
the request body drops the `temperature=0` override.

Resolution options (unchanged from Phase 6B.11 §4, all out of
scope for the v0.2.0a1 envelope):

* **Recipe-level constraint.** Require `temperature=0` (or
  `top_k=1`) in every request body — the workaround already used
  at Phase 6B.15 and Phase 6B.16.
* **NPU-compatible sampling backend.** Introduce an NPU-native
  path so `sample_impl` no longer needs `flashinfer`.
* **Optional-import guard.** Gate the `flashinfer.sampling`
  import behind a device check and fall back to a CUDA-free
  reference implementation.

Only the recipe-level constraint has been exercised on Qwen3-1.7B;
the other two require runtime code changes and remain explicitly
excluded from the v0.2.0a1 envelope.

## 5. Cleanup / invariant summary

Across Phases 6B.12, 6B.13, 6B.14, 6B.15, and 6B.16:

* Every gate confirmed HCCL + Gloo (`ProcessGroupHCCL` watchdog
  warnings + `[Gloo] Rank {0,1} is connected to 1 peer ranks.`)
  as the sole communication backend.
* Every gate reported zero external `pynccl` / `nccl` matches,
  and zero external `cuda` matches outside the deliberate
  `CUDA graph is disabled.` notice.
* Every gate reported zero external `flashinfer` matches — the
  greedy contract held on both `/generate` (default greedy
  `SamplingParams`) and `/v1/chat/completions`
  (`temperature=0` override).
* Every gate reported zero `block_size` / `561002` /
  `CheckFeatureNoquant` matches — `--page-size 16` held for the
  BF16 no-quant path.
* Every gate ended with `NO_RESIDUAL` after SIGTERM and a
  released port; no `SIGKILL` fallback was needed on any gate.
* A benign `multiprocessing.resource_tracker` leaked-semaphore
  warning at parent exit remains the same documented observation
  as in the Qwen3-0.6B arc (Phase 6B.11 §5), not a root-caused
  finding.

## 6. Next recommended gate

**Qwen3-4B server bring-up with the same recipe.**

Concretely: repeat the Phase 6B.12 → 6B.16 arc against
`/mnt/nvme/models/Qwen3-4B` (or the actual model directory), with
no changes to launch flags. Expected outcomes:

* `/v1/models` reports the new model's absolute path as its `id`.
* `/generate` SSE returns a non-empty completion under
  `page_size=16`.
* Both `/v1/chat/completions` greedy paths (non-stream and
  stream) succeed with `temperature=0`.
* The non-greedy `/v1/chat/completions` path remains expected
  BLOCKED for the same reason as Phase 6B.8; that is out of scope
  until the sampler is patched.

Watch items specifically new at the 4B tier:

* Memory headroom: KV cache at 41.27 GiB per rank plus the 4B
  weight tensor per rank is the first gate where the ~64 GiB HBM
  per NPU may become a live constraint (0.6B and 1.7B both fit
  with ~5 GiB free per rank).
* Ready wall-clock: 20 s for 0.6B / 1.7B; the 4B weight tensor
  may extend load time.
* If either watch item forces a change to `--memory-ratio` or a
  smaller KV-cache footprint, that is a **recipe amendment**, not
  a routine bring-up, and would land as a Phase 6B.7-style recipe
  update rather than as part of the 4B arc itself.

If Qwen3-4B lands cleanly, the same envelope is a viable
candidate for the next model up, gated by memory headroom on the
8 × 910B1 host.

## 7. What this summary does NOT establish

* No new experiment was run for this gate — every claim above is
  reference-only, cited to the source verdict file.
* No client-disconnect / abort-ack behaviour (Gate 2.3f) has
  been exercised against the Qwen3-1.7B server path.
* No multi-request, batch, or long-context behaviour has been
  tested on Qwen3-1.7B.
* No non-greedy sampling has been exercised on Qwen3-1.7B; §4 is
  an inference from the device-level Phase 6B.8 root cause, not
  an experimentally-observed failure on this model.
* No `usage` accounting fix has been validated; the constraint
  from §3 remains open.
* No performance claim of any kind is made.
