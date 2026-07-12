# Fixed-TP2 Ascend Adaptation Milestone

**Status:** Documentation-only summary of Gate 4.1–4.14 and
Gate 5.1–5.7 evidence. No new experiments were performed for
this document. No runtime source was modified. This page
consolidates the fixed-TP2 adaptation record for reviewers who
want a single entry point.

---

## Scope

Mini-SGLang Ascend adaptation, **fixed TP=2**.

> This is fixed-TP2 Ascend adaptation.
> **It is not runtime TP elasticity.**
> **It is not TP switching.**
> **It is not a benchmark.**
> **It is not a cross-stack comparison.**
> **It is not a performance superiority claim.**

The fixed-TP2 envelope is the same TP-count for the entire
process lifetime (`torchrun --nproc_per_node=2`, two HCCL ranks
lockstep). Runtime TP resizing / elasticity, cross-TP-count
switching, Graph Re-Linker, Tensor-Remap-Kernel, and any TP > 2
configuration are all **out of scope** for this milestone.

## Envelope

| Knob | Value |
|---|---|
| Hardware | 2 × Ascend 910B1 (64 GiB HBM each) |
| Parallelism | TP = 2 (torchrun `--nproc_per_node=2`) |
| Execution mode | eager |
| Attention backend | `npu_fia` |
| dtype | bf16 |
| Sampling | greedy (temperature=0.0, top_k=1, top_p=1.0, `ignore_eos=True`) |
| Distributed collectives | HCCL primary, gloo sidecar; `use_pynccl=False` |
| Rendezvous | `MINISGL_DISTRIBUTED_ADDR=env://` (reuses torchrun's TCPStore) |
| CUDAGraph | disabled (`cuda_graph_bs=[]`, torch_npu has no CUDAGraph) |
| KV | paged, `page_size=16`, radix prefix reuse |

Software: torch 2.4.0, torch_npu 2.9.0.post1, CANN 8.5.1,
Python 3.11.14.

## Models

All three models are frozen at these paths on the validation host:

| Model | Path on validation host |
|---|---|
| Qwen3-0.6B | `/mnt/nvme/models/Qwen3-0.6B` |
| Qwen3-1.7B | `/mnt/nvme/models/Qwen3-1.7B` |
| Qwen3-4B   | `/mnt/nvme/models/Qwen3-4B` |

All three are dense Qwen3 architectures, bf16 weights.

## Covered capabilities

Each capability was proven separately per model per gate; the
fixed-TP2 capability matrix at Gate 4.14 re-ran the entire
functional set on Qwen3-0.6B and Qwen3-1.7B in one driver, and
Gate 5.7 unified all three models (Qwen3-0.6B / 1.7B / 4B) under
the same six-case functional matrix in a single driver.

| Capability | Qwen3-0.6B evidence | Qwen3-1.7B evidence | Qwen3-4B evidence |
|---|---|---|---|
| TP=2 init + weight-shard load | Gate 4.1 | Gate 4.8 | Gate 5.1 |
| B=1 single request | Gate 4.1 | Gate 4.8 | Gate 5.1 |
| B=1 `max_new_tokens=16` | Gate 4.13 §4 (via 4.14 matrix) | Gate 4.13 §4 | Gate 5.6 §5 / Gate 5.7 §5 |
| B=2 equal-length | Gate 4.2 | Gate 4.9 | Gate 5.2 |
| B=2 ragged prefill (unequal lengths) | Gate 4.3 | Gate 4.10 | Gate 5.3 |
| B=2 mixed-KV decode | Gate 4.4 | Gate 4.11 | Gate 5.4 |
| Dynamic admission B: 1 → 2 → 1 | Gate 4.5 | Gate 4.12 | Gate 5.5 |
| Dynamic grow / shrink B: 1 → 2 → 3 → 2 → 1 | Gate 4.6 | *(out of milestone scope)* | *(out of milestone scope)* |
| Timing snapshots (not benchmarks) | Gate 4.7 | Gate 4.13 | Gate 5.6 |
| Two-model functional capability matrix | Gate 4.14 | Gate 4.14 | — |
| Three-model unified functional capability matrix | Gate 5.7 | Gate 5.7 | Gate 5.7 |

Gate 5.7 confirms that **Qwen3-0.6B, Qwen3-1.7B, and Qwen3-4B all
pass the fixed-TP2 functional matrix**:

* **A.** B=1 single request, `max_new_tokens=8`
* **B.** B=1 single request, `max_new_tokens=16`
* **C.** B=2 equal-length prefill, `max_new_tokens=8` each
* **D.** B=2 ragged prefill (unequal lengths), `max_new_tokens=8` each
* **E.** B=2 mixed-KV decode (unequal per-request KV extents on every joint decode step)
* **F.** Dynamic admission B: 1 → 2 → 1 (staggered reveal of the second request)

Every capability record above verifies the following invariants
on **both** TP ranks:

* `output_texts[i]` and `output_token_ids[i]` byte-identical
  across rank 0 and rank 1 for every uid
* `available_tokens_after_case == baseline_available_tokens`
* `deferred_abort_uids == 0`
* `cache_integrity_ok == true`

Case-specific extra evidence (mixed-KV `kv_lengths` inequality,
dynamic admission ordered `[1, 2, 1]` timeline, etc.) is listed
in each individual gate verdict.

## Evidence source

The single source of truth is the frozen Gate verdict set under
[`docs/ascend_port/`](./). Every commit SHA, launch command,
JSONL trace prefix, and allocator baseline referenced in this
milestone is anchored in one of these files:

* Gate 4.1 — [`gate4.1_tp2_single_request_verdict.md`](./gate4.1_tp2_single_request_verdict.md)
* Gate 4.2 — [`gate4.2_tp2_b2_equal_length_verdict.md`](./gate4.2_tp2_b2_equal_length_verdict.md)
* Gate 4.3 — [`gate4.3_tp2_b2_ragged_prefill_verdict.md`](./gate4.3_tp2_b2_ragged_prefill_verdict.md)
* Gate 4.4 — [`gate4.4_tp2_b2_mixed_kv_decode_verdict.md`](./gate4.4_tp2_b2_mixed_kv_decode_verdict.md)
* Gate 4.5 — [`gate4.5_tp2_dynamic_admission_b1_b2_b1_verdict.md`](./gate4.5_tp2_dynamic_admission_b1_b2_b1_verdict.md)
* Gate 4.6 — [`gate4.6_tp2_dynamic_grow_shrink_b1_b2_b3_b2_b1_verdict.md`](./gate4.6_tp2_dynamic_grow_shrink_b1_b2_b3_b2_b1_verdict.md)
* Gate 4.7 — [`gate4.7_tp2_timing_baseline_verdict.md`](./gate4.7_tp2_timing_baseline_verdict.md)
* Gate 4.8 — [`gate4.8_qwen1.7b_tp2_single_request_verdict.md`](./gate4.8_qwen1.7b_tp2_single_request_verdict.md)
* Gate 4.9 — [`gate4.9_qwen1.7b_tp2_b2_equal_length_verdict.md`](./gate4.9_qwen1.7b_tp2_b2_equal_length_verdict.md)
* Gate 4.10 — [`gate4.10_qwen1.7b_tp2_b2_ragged_prefill_verdict.md`](./gate4.10_qwen1.7b_tp2_b2_ragged_prefill_verdict.md)
* Gate 4.11 — [`gate4.11_qwen1.7b_tp2_b2_mixed_kv_decode_verdict.md`](./gate4.11_qwen1.7b_tp2_b2_mixed_kv_decode_verdict.md)
* Gate 4.12 — [`gate4.12_qwen1.7b_tp2_dynamic_admission_b1_b2_b1_verdict.md`](./gate4.12_qwen1.7b_tp2_dynamic_admission_b1_b2_b1_verdict.md)
* Gate 4.13 — [`gate4.13_qwen1.7b_tp2_timing_baseline_verdict.md`](./gate4.13_qwen1.7b_tp2_timing_baseline_verdict.md)
* Gate 4.14 — [`gate4.14_fixed_tp2_capability_matrix_verdict.md`](./gate4.14_fixed_tp2_capability_matrix_verdict.md)
* Gate 5.1 — [`gate5.1_qwen4b_tp2_single_request_verdict.md`](./gate5.1_qwen4b_tp2_single_request_verdict.md)
* Gate 5.2 — [`gate5.2_qwen4b_tp2_b2_equal_length_verdict.md`](./gate5.2_qwen4b_tp2_b2_equal_length_verdict.md)
* Gate 5.3 — [`gate5.3_qwen4b_tp2_b2_ragged_prefill_verdict.md`](./gate5.3_qwen4b_tp2_b2_ragged_prefill_verdict.md)
* Gate 5.4 — [`gate5.4_qwen4b_tp2_b2_mixed_kv_decode_verdict.md`](./gate5.4_qwen4b_tp2_b2_mixed_kv_decode_verdict.md)
* Gate 5.5 — [`gate5.5_qwen4b_tp2_dynamic_admission_b1_b2_b1_verdict.md`](./gate5.5_qwen4b_tp2_dynamic_admission_b1_b2_b1_verdict.md)
* Gate 5.6 — [`gate5.6_qwen4b_tp2_timing_baseline_verdict.md`](./gate5.6_qwen4b_tp2_timing_baseline_verdict.md)
* Gate 5.7 — [`gate5.7_fixed_tp2_qwen3_three_model_matrix_verdict.md`](./gate5.7_fixed_tp2_qwen3_three_model_matrix_verdict.md)

Public-hygiene redaction of the above corpus was performed at
Gate 4.13a (targeted, Gate 4.13 only) and Gate 4.13b (repo-wide
tracked HEAD sweep). Details are in
[`gate4.13a_redact_public_hygiene_verdict.md`](./gate4.13a_redact_public_hygiene_verdict.md)
and
[`gate4.13b_repo_head_redaction_verdict.md`](./gate4.13b_repo_head_redaction_verdict.md).

## Non-goals (explicit)

The following are **not** claimed by this milestone and are
**not** covered by any of the Gate 4.1–4.14 or Gate 5.1–5.7
verdicts:

* Runtime TP elasticity or runtime TP switching
* Graph Re-Linker
* Tensor-Remap-Kernel
* TP > 2 (TP=4 / TP=8 / …)
* Non-Qwen3 architectures
* Qwen3-14B / 32B / quantized / MoE variants (Qwen3-Next, Qwen3-Coder-Next, Qwen3-ASR-*, etc.)
* Cross-stack comparison against SGLang, vLLM, TGI, TensorRT-LLM
* Any benchmark, performance ranking, or throughput claim
* Long-duration soak / rolling-allocator stability run
* HTTP / server / cross-process request path (only the offline
  driver plus committed hermetic tests are attested)
* Upstream merge acceptance into `sgl-project/mini-sglang`

## Closure

The Gate 4.1–4.14 and Gate 5.1–5.7 verdict set jointly closes
the fixed-TP2 Ascend adaptation milestone for Qwen3-0.6B,
Qwen3-1.7B, and Qwen3-4B under the documented envelope. All
further work (TP > 2, larger dense models, quantized / MoE
variants, benchmark harnesses, upstream merge) is out of scope
for this milestone and must be introduced under a new gate.
