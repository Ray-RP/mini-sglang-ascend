# Changelog

## v0.2.0a1 - Ascend fixed-TP2 technical preview

- Adds documented fixed-TP2 Ascend adaptation coverage for Qwen3-0.6B, Qwen3-1.7B, and Qwen3-4B.
- Adds evidence-backed functional matrix for B=1, B=2 equal-length, B=2 ragged prefill, mixed-KV decode, and dynamic admission B: 1→2→1.
- Adds release-readiness audit, RC notes, credential-rotation checklist, and explicit release-owner tag permission records.
- Scope remains fixed TP=2, eager npu_fia bf16 greedy.
- Not TP elasticity, not runtime TP switching, not a benchmark, not a cross-stack comparison, not a performance-superiority claim.

## v0.1.0a1 — Ascend Technical Preview

See `docs/ascend_port/release_notes_0.1.0a1.md`.
