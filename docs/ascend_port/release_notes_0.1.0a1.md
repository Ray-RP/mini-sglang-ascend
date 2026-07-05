# v0.1.0a1 — Ascend Technical Preview

**Release name:** `v0.1.0a1 — Ascend Technical Preview`
**Status:** Technical preview — **not production-ready**.

This is the first public alpha of the Ascend NPU port of
[`sgl-project/mini-sglang`](https://github.com/sgl-project/mini-sglang).
Every capability listed here is backed by a signed Gate verdict under
[`docs/ascend_port/`](.). The scope of what has been proven is narrow
and deliberately conservative; anything not listed under "Key
capabilities" is not covered by this release.

---

## Verified platform

| Dimension | Verified value |
| :-- | :-- |
| Hardware | Ascend 910B1 |
| Model | Qwen3-0.6B |
| Parallelism | TP=1 |
| Execution | eager mode |
| Attention backend | `npu_fia` |

Anything outside this envelope (other 910 SKUs, other models, TP>1,
graph capture, alternative attention backends) is **not** attested by
this release.

---

## Included milestones

Each milestone is frozen at a specific commit with a signed verdict.

| Gate | Scope | Verdict |
| :-- | :-- | :-- |
| Gate 1 | Single-request eager on Ascend 910B1 | [`gate1_verdict.md`](./gate1_verdict.md) |
| Gate 2.1 | Single-request multistep generation, per-request stop tokens | [`gate2_1_multistep_verdict.md`](./gate2_1_multistep_verdict.md) |
| Gate 2.2 | Multi-request batching (equal-length, ragged, mixed-KV decode, dynamic admission) | [`gate2_2_multirequest_verdict.md`](./gate2_2_multirequest_verdict.md) |
| Gate 2.3 | Request lifecycle & cancel protocol (rollback, atomicity, drain, overlap abort, abort-ack) | [`gate2_3_request_lifecycle_verdict.md`](./gate2_3_request_lifecycle_verdict.md) |

---

## Key capabilities

- Real model inference on Ascend 910B1 with Qwen3-0.6B in float16.
- Multi-step generation with per-request stop tokens.
- Paged KV cache with radix prefix reuse.
- Equal-length continuous batching.
- Ragged continuous batching (heterogeneous input lengths in one batch).
- Mixed-KV-length decode (per-request cached-length differences within a
  single decode batch).
- Dynamic admission / grow / shrink of the running set across ticks.
- Allocation rollback: transactional `_prepare_batch` returns KV pages
  on failure with allocator invariants preserved.
- Sampler commit atomicity: sampler-stage failure leaves the request
  uncommitted; no partial-token state escapes.
- Shutdown drain: `Scheduler.shutdown()` completes cleanly on NPU
  including with TP=1 (no distributed group required).
- Overlap-safe abort: overlap-loop respects an abort fence between the
  compute and post-processing stages.
- AbortAck: end-to-end abort acknowledgement via `AbortAckMsg` /
  `AbortAckReply`.

---

## Explicit limitations

- **No TP>1 guarantee.** Gate 1 asserts HCCL init only; there is no
  attested cross-rank forward or decode on NPU.
- **Only Qwen3-0.6B has completed end-to-end validation.** Other Qwen
  sizes, the Llama family, and MoE variants are out of scope for this
  release.
- **No full HTTP + ZMQ cross-process freeze.** All lifecycle
  guarantees were proven with the in-process offline driver plus the
  hermetic test suite; the multi-process API server path is not
  frozen.
- **No long-duration soak.** There is no rolling-allocator or
  thousands-of-tick stability run behind this release.
- **No performance-leadership claim.** No benchmark numbers vs.
  SGLang / vLLM / any other framework on NPU are published in this
  release.
- **Not upstream-merged.** This fork is **not** merged into
  `sgl-project/mini-sglang`; it is a downstream Ascend port.

---

## Links

- Gate 1 verdict — [`gate1_verdict.md`](./gate1_verdict.md)
- Gate 2.1 verdict — [`gate2_1_multistep_verdict.md`](./gate2_1_multistep_verdict.md)
- Gate 2.2 verdict — [`gate2_2_multirequest_verdict.md`](./gate2_2_multirequest_verdict.md)
- Gate 2.3 verdict — [`gate2_3_request_lifecycle_verdict.md`](./gate2_3_request_lifecycle_verdict.md)
- Upstream project — [`sgl-project/mini-sglang`](https://github.com/sgl-project/mini-sglang)
- This fork — [`Ray-RP/mini-sglang-ascend`](https://github.com/Ray-RP/mini-sglang-ascend)
