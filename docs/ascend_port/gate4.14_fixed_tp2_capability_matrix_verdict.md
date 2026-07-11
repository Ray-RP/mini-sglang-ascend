# Gate 4.14 Verdict — Fixed-TP2 Adaptation Capability Matrix

**Gate ID:** 4.14 (Fixed-TP2 adaptation capability matrix, Qwen3-0.6B + Qwen3-1.7B on Ascend 910B1)
**Verdict:** PASS
**Branch:** `gate4.14-fixed-tp2-capability-matrix`
**Base commit:** `7f5f639` (tip of `ascend-port`, Gate 4.13b merge)
**Freeze commit:** `e9e1212`
**Date:** 2026-07-12
**Kind:** Functional capability matrix — two ranks × two models
(Qwen3-0.6B + Qwen3-1.7B) × six cases (A–F) × one pass each,
greedy, `use_pynccl=False`. Records structured JSONL per rank per
case (prompts, timeline, decode metadata, allocator invariants,
output token ids / texts) plus a per-model
`GATE4.14_MATRIX_RESULT=PASS` footer. Neither runtime nor tests
are touched.

> **This is a fixed-TP2 Ascend adaptation capability matrix.**
> **It is not TP elasticity.** **It is not a benchmark.**
> **It is not a cross-stack comparison.** No timing statistics
> are collected; per-case wall-clocks are intentionally out of
> scope. TP > 2, TP runtime elasticity, runtime TP switching,
> Graph Re-Linker, Tensor-Remap-Kernel, non-Qwen3 architectures,
> and Qwen3-4B / 14B / 32B / quantized / MoE variants are all
> out of scope.

This verdict does not modify Gate 1 / 2.1 / 2.2 / 2.3 / 2.4 /
2.5 / 3.1 / 3.2 / 3.3 / 3.4 / 4.1 / 4.2 / 4.3 / 4.4 / 4.5 / 4.6
/ 4.7 / 4.8 / 4.9 / 4.10 / 4.11 / 4.12 / 4.13 / 4.13a / 4.13b,
does not mutate release tag `v0.1.0a1`, does not touch the
GitHub Release / CHANGELOG / release notes, and does not extend
the Ascend port beyond the fixed-TP2 envelope. The only new
artefacts at this gate are one bring-up script and this verdict;
no runtime source under `python/minisgl/` is modified; no test
file is modified.

---

## 1. Verdict summary

**PASS on all 12 model×case records (2 models × 6 cases) across
both TP ranks.**

Per-model matrix — each cell reports rank 0 / rank 1 status:

| Case | Description | Qwen3-0.6B | Qwen3-1.7B |
|---|---|:---:|:---:|
| A | B=1 single request, `max_new_tokens=8` | **PASS** / **PASS** | **PASS** / **PASS** |
| B | B=1 single request, `max_new_tokens=16` | **PASS** / **PASS** | **PASS** / **PASS** |
| C | B=2 equal-length, `max_new_tokens=8` | **PASS** / **PASS** | **PASS** / **PASS** |
| D | B=2 ragged prefill (unequal), `max_new_tokens=8` | **PASS** / **PASS** | **PASS** / **PASS** |
| E | B=2 mixed-KV decode evidence, `max_new_tokens=8` | **PASS** / **PASS** | **PASS** / **PASS** |
| F | dynamic admission B: 1 → 2 → 1 | **PASS** / **PASS** | **PASS** / **PASS** |

Every record on both ranks reported:

* `available_tokens_after_case == baseline_available_tokens`
* `deferred_abort_uids == 0`
* `cache_integrity_ok == true`
* `output_texts[i]` and `output_token_ids[i]` byte-identical
  across rank 0 and rank 1 for every uid

Per-model footers:

```
GATE4.14_MATRIX_RESULT=PASS   # Qwen3-0.6B run
GATE4.14_MATRIX_RESULT=PASS   # Qwen3-1.7B run
```

## 2. Envelope (locked)

| Knob | Value |
|---|---|
| Hardware | 2 × Ascend 910B1 (64 GiB HBM each) |
| Container | `<CONTAINER>` on `<HOST>:<PORT>` |
| torch | 2.4.0 |
| torch_npu | 2.9.0.post1 |
| CANN | 8.5.1 |
| Python | 3.11.14 |
| Models | Qwen3-0.6B (`/mnt/nvme/models/Qwen3-0.6B`), Qwen3-1.7B (`/mnt/nvme/models/Qwen3-1.7B`) |
| dtype | bf16 |
| TP | 2 (torchrun `--nproc_per_node=2`) |
| Attention backend | `npu_fia` |
| Rendezvous | `MINISGL_DISTRIBUTED_ADDR=env://` (reuses torchrun's TCPStore) |
| Distributed collectives | HCCL primary, gloo sidecar; `use_pynccl=False` |
| CUDAGraph batch sizes | `cuda_graph_bs=[]` (torch_npu has no CUDAGraph) |
| Execution mode | eager |
| Sampling | greedy (temperature=0.0, top_k=1, top_p=1.0, `ignore_eos=True`) |
| `memory_ratio` | 0.85 |
| `page_size` | 16 |
| `max_running_req` | 4 |
| Per-case repeats | 1 (functional, no warmup, no measured repeats) |
| Metadata hook | script-local monkey-patch of `AscendFIABackend.prepare_metadata`; runtime unchanged |

## 3. Launch command

Same outer-shell retry pattern as Gates 4.10 / 4.11 / 4.12 /
4.13: sleep 8 s between attempts, bump `--master_port`, thread
`--cold-start-attempt-id` and `--memory-sync-retry-note` into
the driver. Attempt 1 (port 29610) succeeded cleanly on both
models — no retry needed.

> **Public hygiene note:** remote host, username, password, and
> container identifiers are redacted in this document.

```bash
ssh -p <PORT> <USER>@<HOST> \
  "docker exec <CONTAINER> bash -c '
    set +e
    cd <REMOTE_PATH>/mini-sglang-ascend-gate4.14 &&
    mkdir -p logs &&
    for MODEL_ID in Qwen3-0.6B Qwen3-1.7B; do
      PORT_BASE=29600
      NOTE=\"\"
      for ATTEMPT in 1 2 3; do
        PORT=\$((PORT_BASE + ATTEMPT * 10))
        LOG=logs/gate4.14_\${MODEL_ID}_attempt\${ATTEMPT}.log
        PYTHONPATH=./python:\$PYTHONPATH torchrun --nproc_per_node=2 --master_port=\$PORT \\
          scripts/gate4_14_fixed_tp2_capability_matrix.py \\
          --model-path /mnt/nvme/models/\$MODEL_ID \\
          --model-name \$MODEL_ID \\
          --cold-start-attempt-id \$ATTEMPT \\
          --memory-sync-retry-note \"\$NOTE\" > \$LOG 2>&1
        if grep -q GATE4.14_MATRIX_RESULT= \$LOG && ! grep -q \"Memory across TP ranks are imbalanced\" \$LOG; then
          cp \$LOG logs/gate4.14_\${MODEL_ID}.log
          break
        fi
        IMB=\$(grep \"Memory across TP ranks are imbalanced\" \$LOG | head -1)
        NOTE=\"\$NOTE | attempt \$ATTEMPT port \$PORT: \$IMB\"
        sleep 8
      done
    done
  '"
```

## 4. Prompts

Prompts (Qwen3 tokenizer counts, identical across both model
tokenizers because both are Qwen3 family):

| Case | Prompts | `prompt_token_lengths` | `requested_max_new_tokens` |
|---|---|---|---|
| A | `"Paris is"` | `[2]` | `[8]` |
| B | `"Paris is"` | `[2]` | `[16]` |
| C | `"The capital of France is"`, `"The capital of Italy is"` | `[5, 5]` | `[8, 8]` |
| D | `"Paris is"`, `"The largest planet in our solar system by mass and volume is"` | `[2, 12]` | `[8, 8]` |
| E | same as D | `[2, 12]` | `[8, 8]` |
| F | same as D (dynamic admission) | `[2, 12]` | `[8, 8]` |

## 5. Per-model output evidence

### 5.1 Qwen3-0.6B (rank 0 / rank 1 byte-identical per uid)

| Case | `actual_output_tokens_per_request` | `output_texts` |
|---|---|---|
| A | `[8]` | `[" a city in the southern part of France"]` |
| B | `[16]` | `[" a city in the southern part of France, and it is the capital of the"]` |
| C | `[8, 8]` | `[" Paris. The capital of Italy is Rome", " Rome. The capital of France is Paris"]` |
| D | `[8, 8]` | `[" a city in the southern part of France", "...? A. Mercury B. Venus"]` |
| E | `[8, 8]` | same as D |
| F | `[8, 8]` | same as D |

### 5.2 Qwen3-1.7B (rank 0 / rank 1 byte-identical per uid)

| Case | `actual_output_tokens_per_request` | `output_texts` |
|---|---|---|
| A | `[8]` | `[" the capital of France. Paris is the"]` |
| B | `[16]` | `[" the capital of France. Paris is the capital of France. Paris is the capital"]` |
| C | `[8, 8]` | `[" Paris. The capital of the United States", " Rome. The capital of France is Paris"]` |
| D | `[8, 8]` | `[" the capital of France. Paris is the", " the planet Jupiter. It is the second"]` |
| E | `[8, 8]` | same as D |
| F | `[8, 8]` | same as D |

Rank-0 vs rank-1 output equality was verified by exact byte
match on `output_texts[i]` and every element of
`output_token_ids[i]` for every uid across all 12 records.

## 6. Per-model allocator evidence

`baseline_available_tokens` and `total_pages` differ across
models because the KV budget is model-specific (Qwen3-0.6B has
smaller per-layer KV state than Qwen3-1.7B), but each model's
baseline is bit-identical across ranks and every case returned
to it exactly.

### 6.1 Qwen3-0.6B

| Field | value | 
|---|---|
| `baseline_available_tokens` (both ranks) | 945376 |
| `baseline_free_pages` (both ranks) | 59086 |
| `available_tokens_after_case` on **every** case | 945376 |
| `deferred_abort_uids` on every case | 0 |
| `cache_integrity_ok` on every case | true |
| `free_pages_before_after` per case | A `[59086, 59086]`, B `[59086, 59085]`, C `[59086, 59085]`, D `[59086, 59084]`, E `[59086, 59084]`, F `[59086, 59084]` |

### 6.2 Qwen3-1.7B

| Field | value | 
|---|---|
| `baseline_available_tokens` (both ranks) | 923152 |
| `baseline_free_pages` (both ranks) | 57697 |
| `available_tokens_after_case` on **every** case | 923152 |
| `deferred_abort_uids` on every case | 0 |
| `cache_integrity_ok` on every case | true |
| `free_pages_before_after` per case | A `[57697, 57697]`, B `[57697, 57696]`, C `[57697, 57696]`, D `[57697, 57695]`, E `[57697, 57695]`, F `[57697, 57695]` |

The 0–2 page reduction in `free_pages` after C–F is retained
evictable radix-cache prefixes for the previously-seen prompt
strings — the same benign pattern documented at Gate 4.5 / 4.10
/ 4.11 / 4.12 / 4.13, absorbed by `available_size = free_slots +
evictable_prefix_pages`. `check_integrity()` confirms radix-cache
linkage on every record; not a page leak.

## 7. Mixed-KV decode evidence (Case E)

Both models produced a decode metadata trace where every pure
decode step (`query_lengths == [1, 1]`) had `kv_lengths[0] !=
kv_lengths[1]`:

| Model | `decode_step_count` | `mixed_kv_decode_step_count` |
|---|---:|---:|
| Qwen3-0.6B | 7 | **7** |
| Qwen3-1.7B | 7 | **7** |

Gate 4.14 acceptance for E requires `mixed_kv_decode_step_count
>= 1`. Both models satisfied it on **7 / 7** decode steps, on
both ranks (rank 0 and rank 1 metadata traces are byte-identical
because the FIA scheduler plans identical shapes across TP
ranks). The KV delta per decode step is exactly
`long_prompt_len - short_prompt_len == 10` on every step,
consistent with Gate 4.4 (0.6B) and Gate 4.11 (1.7B) evidence.

## 8. Dynamic admission evidence (Case F)

Both models produced the exact same `batch_timeline` on Case F,
on both ranks:

```
[1, 1, 1, 2, 2, 2, 2, 2, 2, 1]
```

The ordered subsequence `1 → 2 → 1` is present on every trace
(`contains_ordered_1_2_1 == true`). This matches the Gate 4.5
(0.6B) and Gate 4.12 (1.7B) reference timelines. The staggered
`LLM` subclass reveals request B on the tick after the metadata
hook observes `batch_size == 1, query_lengths == [1]` (A alone
decoding), and B completes one step after A because it was
admitted two prefill/decode steps behind.

`actual_output_tokens_per_request` was `[8, 8]` on every F pass,
so the total output-tokens work is 16 tokens across two
requests. `available_tokens_after_case` returned to baseline on
both ranks.

## 9. Cold-start retry note

**Attempt 1 succeeded cleanly for both models.** Neither model's
run tripped the pre/post-load `_sync_get_memory()` imbalance
check on the first attempt. Every JSONL record reports
`cold_start_attempt_id == 1` and `memory_sync_retry_note == ""`.

The outer shell loop was authorised (per Gate 4.14 spec §3) to
sleep 8 s and re-launch with a fresh `--master_port` up to 2 more
times per model on cold-start imbalance. It was not exercised
on either model in this run.

The pre-existing 2 GiB tolerance in `Engine._sync_get_memory()`
at `python/minisgl/engine/engine.py:246` is unchanged. Gate 4.14
did not modify `python/minisgl/`.

## 10. Regression evidence

Optional pytest per §8 was skipped — Gate 4.14 modifies zero
runtime and zero test files. The Gate 4.13 regression measurement
of `51 / 51 PASS` per-file pytest on `tests/misc/` is unchanged
by construction.

## 11. Known limitations

* **Functional-only capability probe.** No timing statistics, no
  repeats, no warmup. The metadata-snapshot hook adds unbudgeted
  per-step overhead, present equally on every record, which is
  fine for functional PASS but disqualifies these traces as
  timing evidence.
* **Two Qwen3 models only.** Qwen3-4B / 14B / 32B, quantized
  weights, and MoE variants are out of scope.
* **TP=2 only.** TP=4 / TP=8, TP runtime elasticity, runtime TP
  switching, Graph Re-Linker, Tensor-Remap-Kernel are all out of
  scope.
* **`use_pynccl=False` is mandatory on NPU.** All numbers reflect
  the HCCL + gloo sidecar collective path.
* **Radix-cache retention between cases** (0–2 pages) is benign;
  `available_size` normalises back to baseline exactly on every
  record.
* **Cold-start `_sync_get_memory()` variability** documented at
  Gate 4.9 remains outside `python/minisgl/`. The Gate 4.14
  outer shell loop authorises up to 2 retries with fresh
  `master_port` and 8 s sleep per model. On this run only
  attempt 1 was required.
* **Not a benchmark.** No throughput, latency, or cross-stack
  comparisons are made or implied.

## 12. Decision matrix

| Question | Answer |
|---|---|
| Does Qwen3-0.6B `LLM.__init__` return on both ranks under TP=2? | Yes (attempt 1) |
| Does Qwen3-1.7B `LLM.__init__` return on both ranks under TP=2? | Yes (attempt 1) |
| Do both models complete cases A–F on both ranks? | Yes |
| Is `actual_output_tokens_per_request` correct on A/B for both models? | Yes (`[8]`, `[16]`) |
| Are prompt lengths equal on C and unequal on D/E/F for both models? | Yes (`[5,5]` and `[2,12]`) |
| Is `actual_output_tokens_per_request == [8, 8]` on C/D/E/F for both models? | Yes |
| Does Case E have `>=1` decode snapshot with `qlens==[1,1]` and unequal `kv_lengths` for both models? | Yes (7 / 7 for both) |
| Does Case F `batch_timeline` contain ordered `[1, 2, 1]` for both models? | Yes |
| Is rank 0 `output_texts[i]` byte-identical to rank 1 for every uid? | Yes on all 12 records |
| Is rank 0 `output_token_ids[i]` byte-identical to rank 1 for every uid? | Yes on all 12 records |
| Does `available_tokens_after_case == baseline` on every record? | Yes on all 12 records for both models |
| Are `deferred_abort_uids == 0` on every record? | Yes |
| Does `check_integrity()` pass on every record? | Yes |
| Did the driver touch `python/minisgl/`? | No |
| Did the driver touch tests? | No |
| Did the driver claim performance superiority? | No |
| Did the driver compare against SGLang / vLLM / TGI? | No |
| Is `use_pynccl=False`? | Yes |
| Are the models limited to Qwen3-0.6B and Qwen3-1.7B? | Yes |
| Is TP fixed at 2? | Yes |

**Verdict: PASS.**

## 13. Freeze boundary

The frozen artefacts for Gate 4.14 are:

* `scripts/gate4_14_fixed_tp2_capability_matrix.py`
* `docs/ascend_port/gate4.14_fixed_tp2_capability_matrix_verdict.md`

No files under `python/minisgl/` were modified at this gate. No
tests were modified at this gate. The freeze commit SHA is
recorded in this document header once the driver + verdict pair
is committed, and it is recorded on the `ascend-port` tip once
the branch is merged with `--no-ff`.
