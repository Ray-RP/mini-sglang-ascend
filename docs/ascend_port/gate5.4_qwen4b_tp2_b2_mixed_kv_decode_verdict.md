# Gate 5.4 Verdict — Qwen3-4B Fixed-TP2 B=2 Mixed-KV Decode Explicit Evidence

**Gate ID:** 5.4 (Qwen3-4B fixed-TP2 B=2 mixed-KV decode explicit evidence on Ascend 910B1)
**Verdict:** PASS
**Branch:** `gate5.4-qwen4b-tp2-b2-mixed-kv-decode`
**Base commit:** `e1b157b` (tip of `ascend-port`, Gate 5.3 merge)
**Freeze commit:** `4c1c7e0`
**Original pre-amend commit:** `d0ceaac` (superseded by header-fix
amend; retained as provenance only — the final frozen SHA on the
`gate5.4-qwen4b-tp2-b2-mixed-kv-decode` branch tip is `4c1c7e0`)
**Date:** 2026-07-12
**Kind:** Functional B=2 mixed-KV-decode bring-up with explicit
per-step FIA metadata evidence — Qwen3-4B on 2 × Ascend 910B1 under
fixed TP=2, eager, `npu_fia`, bf16, greedy. Records structured
JSONL per rank for Cases A/B/C/D plus a `GATE5.4_RESULT=PASS`
footer. No timing statistics, no repeats, no warmup. Neither
runtime nor tests are touched.

> **This is fixed-TP2 Ascend adaptation with explicit mixed-KV
> decode evidence.**
> **It is not TP elasticity.**
> **It is not TP switching.**
> **It is not a benchmark.**
> No timing statistics are collected; the wall-clock in the JSONL
> `generate_ms` field is diagnostic only. Dynamic admission, B > 2,
> TP > 2, TP elasticity, runtime TP switching, Graph Re-Linker,
> Tensor-Remap-Kernel, non-Qwen3 architectures, and Qwen3-14B / 32B
> / quantized / MoE variants are all out of scope. The Gate 2.2f
> documented "ragged + non-zero `cached_len` + `extend_len > 1`"
> unsupported FIA branch is intentionally avoided.

This verdict does not modify Gate 1 / 2.1 / 2.2 / 2.3 / 2.4 / 2.5 /
3.1 / 3.2 / 3.3 / 3.4 / 4.1 / 4.2 / 4.3 / 4.4 / 4.5 / 4.6 / 4.7 /
4.8 / 4.9 / 4.10 / 4.11 / 4.12 / 4.13 / 4.13a / 4.13b / 4.14 / 4.15
/ 4.15a / 5.1 / 5.2 / 5.3, does not mutate release tag `v0.1.0a1`,
does not touch the GitHub Release / CHANGELOG / release notes /
README, and does not extend the Ascend port beyond the fixed-TP2
B=2 mixed-KV-decode Qwen3-4B envelope. The only new artefacts at
this gate are one bring-up script and this verdict; no runtime
source under `python/minisgl/` is modified; no test file is
modified.

---

## 1. Verdict summary

**PASS on all four cases (A / B / C / D) across both TP ranks.**

| Case | Description | Rank 0 | Rank 1 |
|---|---|:---:|:---:|
| A | TP=2 init (`LLM.__init__` returns) | **PASS** | **PASS** |
| B | Qwen3-4B TP=2 weight load + post-load `check_integrity()` | **PASS** | **PASS** |
| C | B=2 unequal-length prefill + decode, `max_new_tokens=8` | **PASS** | **PASS** |
| D | Mixed-KV decode metadata evidence (≥1 step with `q=[1,1]` and `kv[0]!=kv[1]`) | **PASS** | **PASS** |

Every record on both ranks reported:

* `prompt_token_lengths == [2, 12]` (unequal)
* `actual_output_tokens_per_request == [8, 8]`
* `mixed_kv_decode_step_count == 7`
* `available_tokens_after_case == baseline_available_tokens`
* `deferred_abort_uids == 0`
* `cache_integrity_ok == true`
* `output_texts[i]` and `output_token_ids[i]` byte-identical across
  rank 0 and rank 1 for every uid

Footer:

```
GATE5.4_RESULT=PASS
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
| Model | Qwen3-4B (`/mnt/nvme/models/Qwen3-4B`) |
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
| Batch size | 2 (unequal-length, `cached_len == 0`) |
| Repeats | 1 (functional, no warmup, no measured repeats) |

## 3. Model path check

`/mnt/nvme/models/Qwen3-4B` present on the validation host
container. Both ranks logged `model_exists == true`. The on-disk
config (`Qwen3ForCausalLM`, `hidden_size=2560`,
`num_hidden_layers=36`, `num_attention_heads=32`,
`num_key_value_heads=8` (GQA 32/8), `head_dim=128`,
`intermediate_size=9728`, `vocab_size=151936`,
`tie_word_embeddings=true`, `bf16`) was already documented at
Gate 5.1 §4 and is unchanged for this gate.

## 4. Launch command

Same outer-shell retry pattern as Gates 4.10 / 4.11 / 4.12 / 4.13 /
4.14 / 5.1 / 5.2 / 5.3: sleep 8 s between attempts, bump
`--master_port` (PORT_BASE=29800 + ATTEMPT*10), thread
`--cold-start-attempt-id` and `--memory-sync-retry-note` into the
driver. **Attempt 1 (port 29810) succeeded cleanly — no retry
needed.**

> **Public hygiene note:** remote host, username, password, and
> container identifiers are redacted in this document.

```bash
ssh -p <PORT> <USER>@<HOST> \
  "docker exec <CONTAINER> bash -c '
    set +e
    cd <REMOTE_PATH>/mini-sglang-ascend-gate5.4 &&
    mkdir -p logs &&
    PORT_BASE=29800
    NOTE=""
    for ATTEMPT in 1 2 3; do
      PORT=$((PORT_BASE + ATTEMPT * 10))
      LOG=logs/gate5.4_Qwen3-4B_attempt${ATTEMPT}.log
      PYTHONPATH=./python:$PYTHONPATH torchrun --nproc_per_node=2 --master_port=$PORT \
        scripts/gate5_4_qwen4b_tp2_b2_mixed_kv_decode.py \
        --model-path /mnt/nvme/models/Qwen3-4B \
        --model-name Qwen3-4B \
        --cold-start-attempt-id $ATTEMPT \
        --memory-sync-retry-note "$NOTE" > $LOG 2>&1
      if grep -q GATE5.4_RESULT= $LOG && ! grep -q "Memory across TP ranks are imbalanced" $LOG; then
        cp $LOG logs/gate5.4_Qwen3-4B.log
        break
      fi
      IMB=$(grep "Memory across TP ranks are imbalanced" $LOG | head -1)
      NOTE="$NOTE | attempt $ATTEMPT port $PORT: $IMB"
      sleep 8
    done
  '"
```

## 5. Prompts and ragged-length evidence

| uid | Prompt | Qwen3-4B tokenized length |
|---|---|---:|
| 0 (short) | `"Paris is"` | 2 |
| 1 (long) | `"The largest planet in our solar system by mass and volume is"` | 12 |

The two prompts are the Gate 5.3 pair reused unchanged. Pre-flight
tokenization (already documented at Gate 5.3 §5) confirmed
`unequal: True`. The driver additionally asserts inequality inside
`generate()`'s setup path so a silent regression into an
equal-length batch cannot skip the mixed-KV decode invariant:

```python
if tokenized_lengths[0] == tokenized_lengths[1]:
    raise RuntimeError(f"Gate 5.4 requires unequal-length prompts; got ...")
```

Both ranks logged `prompt_token_lengths == [2, 12]`.

## 6. Per-step FIA metadata evidence (Case D)

A script-local monkey-patch of `AscendFIABackend.prepare_metadata`
records `(batch_size, query_seq_lens, kv_seq_lens)` per forward
pass (same pattern as Gates 4.4 / 4.11 / 4.14). The runtime source
under `python/minisgl/attention/ascend_fia.py` is unchanged.

Both ranks produced identical 8-step trajectories:

| step | `batch_size` | `query_lengths` | `kv_lengths` | Mixed-KV? |
|---:|:---:|:---:|:---:|:---:|
| 0 (prefill) | 2 | `[2, 12]` | `[2, 12]` | — |
| 1 | 2 | `[1, 1]` | `[3, 13]` | ✓ |
| 2 | 2 | `[1, 1]` | `[4, 14]` | ✓ |
| 3 | 2 | `[1, 1]` | `[5, 15]` | ✓ |
| 4 | 2 | `[1, 1]` | `[6, 16]` | ✓ |
| 5 | 2 | `[1, 1]` | `[7, 17]` | ✓ |
| 6 | 2 | `[1, 1]` | `[8, 18]` | ✓ |
| 7 | 2 | `[1, 1]` | `[9, 19]` | ✓ |

`mixed_kv_decode_step_count == 7` on both ranks. Every pure-decode
step (7 in total, matching `max_new_tokens - 1 = 7` decode steps
after the fused-prefill step 0) satisfies the Gate 5.4 acceptance
predicate: `query_lengths == [1, 1]` (both uids in decode) AND
`kv_lengths[0] != kv_lengths[1]` (per-uid KV lengths diverge). The
per-step `kv_lengths` grow by exactly +1 per uid per step, as
expected under greedy sampling with `ignore_eos=True`, and the
initial gap of 10 tokens (`12 - 2`) is preserved throughout the
decode window.

## 7. Per-rank output evidence (Case C)

Both ranks returned byte-identical outputs per uid:

| uid | Rank 0 `output_text` | Rank 1 `output_text` |
|---|---|---|
| 0 | `" the capital of France, and the capital"` | `" the capital of France, and the capital"` |
| 1 | `" Jupiter. It is a gas giant,"` | `" Jupiter. It is a gas giant,"` |

| uid | Rank 0 `output_token_ids` | Rank 1 `output_token_ids` |
|---|---|---|
| 0 | `[279, 6722, 315, 9625, 11, 323, 279, 6722]` | `[279, 6722, 315, 9625, 11, 323, 279, 6722]` |
| 1 | `[49689, 13, 1084, 374, 264, 6819, 14538, 11]` | `[49689, 13, 1084, 374, 264, 6819, 14538, 11]` |

`actual_output_tokens_per_request == [8, 8]` on both ranks.
Rank-0 vs rank-1 equality was verified by exact byte match on
`output_texts[i]` and every element of `output_token_ids[i]` for
every uid. The outputs are byte-identical to those recorded at
Gate 5.3 for the same prompt pair, confirming that surfacing the
`prepare_metadata` shape via a passive monkey-patch does not
perturb the forward pass — the wrapper calls the original method
first and only records afterwards.

## 8. Per-rank allocator evidence

Baseline captured after weight load, before Case C. Post-case
`available_size` returned to baseline exactly on both ranks.

| Field | Rank 0 | Rank 1 |
|---|---:|---:|
| `total_pages` | 43109 | 43109 |
| `baseline_free_pages` | 43109 | 43109 |
| `baseline_available_tokens` | 689744 | 689744 |
| `free_pages_before_after` (Case C) | `[43109, 43108]` | `[43109, 43108]` |
| `free_pages_after_case` | 43108 | 43108 |
| `available_tokens_after_case` | 689744 | 689744 |
| `deferred_abort_uids` | 0 | 0 |
| `cache_integrity_ok` | true | true |

Baseline is bit-identical across ranks. The 1-page reduction in
`free_pages_after_case` (43108 vs baseline 43109) is retained
evictable radix-cache prefix for one of the two prompt strings —
the same benign pattern documented at Gates 4.5 / 4.10 / 4.11 /
4.12 / 4.13 / 4.14 / 5.3. It is absorbed by
`available_size = free_slots + evictable_prefix_pages`, so
`available_tokens_after_case == baseline_available_tokens`
(689744 == 689744) holds exactly on both ranks. `check_integrity()`
confirms radix-cache linkage; not a page leak.

Baseline (`689744` / `43109`) matches Gate 5.1 (Qwen3-4B B=1 single
request) exactly; the +16-token / +1-page drift observed at
Gates 5.2 / 5.3 (`689760` / `43110`) did not repeat on this fresh
boot. Both allocations sit inside the allocator's page-size
(16-token) granularity envelope and are not a regression, as
already documented at Gate 5.2 §7.

`generate_ms` (diagnostic only — **not** a Gate 5.4 evidence
field): rank 0 = 2694.81 ms, rank 1 = 2671.95 ms. Gate 5.4 makes
no timing claim.

## 9. Cold-start retry note

**Attempt 1 succeeded cleanly.** The run did not trip the
pre/post-load `_sync_get_memory()` imbalance check (2 GiB tolerance
at `python/minisgl/engine/engine.py:246`, unchanged since Gate
4.14). Both ranks reported `cold_start_attempt_id == 1` and
`memory_sync_retry_note == ""`.

The outer shell loop was authorised (per Gate 5.4 spec §5, matching
Gates 5.1 / 5.2 / 5.3) to sleep 8 s and re-launch with a fresh
`--master_port` up to 2 more times on cold-start imbalance. It was
not exercised on this run.

## 10. Public-hygiene grep summary

Post-authoring verification against the six known leaked substring
classes documented at Gate 4.13a / 4.13b (host / user / password /
container / IP / composite):

```
$ git grep -l -E "<OLD_SSHPASS_PATTERN>|<OLD_PASSWORD_PATTERN>|<OLD_HOST_PATTERN>|<OLD_CONTAINER_PATTERN>" \
    scripts/gate5_4_qwen4b_tp2_b2_mixed_kv_decode.py \
    docs/ascend_port/gate5.4_qwen4b_tp2_b2_mixed_kv_decode_verdict.md
(no output — zero files)
```

Loopback `127.0.0.1` and bind `0.0.0.0` do not appear in either
artefact. All host references in the verdict use placeholders
(`<HOST>`, `<PORT>`, `<USER>`, `<CONTAINER>`, `<REMOTE_PATH>`).

## 11. Regression evidence

Optional pytest per Gate 5.4 spec §7 was skipped — Gate 5.4
modifies zero runtime and zero test files. The Gate 4.13
regression measurement of `51 / 51 PASS` per-file pytest on
`tests/misc/` is unchanged by construction.

`git diff --check` on the freeze branch tip: clean.

## 12. Known limitations

* **Single-model, mixed-KV decode over a static B=2 batch only.**
  Qwen3-4B has been proven at B=1 (Gate 5.1), B=2 equal-length
  (Gate 5.2), B=2 ragged prefill (Gate 5.3), and now B=2 mixed-KV
  decode (this gate). Dynamic admission and B > 2 on Qwen3-4B are
  **not** covered. The Gate 2.2f documented "ragged + non-zero
  `cached_len` + `extend_len > 1`" unsupported FIA branch is
  intentionally not exercised. Dynamic admission on Qwen3-4B will
  require its own follow-up gate modelled on Gate 4.12.
* **Mixed-KV evidence is captured via a passive script-local
  monkey-patch.** The wrapper calls the original
  `AscendFIABackend.prepare_metadata` first and only records the
  resulting `FIAMetadata` afterwards; it does not mutate metadata
  and does not alter the forward pass. Byte-identical outputs vs
  Gate 5.3 confirm the hook is non-perturbing.
* **No timing evidence.** `generate_ms` is a diagnostic dump only.
  Gate 5.4 makes no throughput, latency, or steady-state claim.
* **TP=2 only.** TP=4 / TP=8, TP runtime elasticity, runtime TP
  switching, Graph Re-Linker, Tensor-Remap-Kernel are all out of
  scope.
* **One Qwen3 dense model.** Qwen3-14B, Qwen3-32B, quantized
  weights, and Qwen3 MoE variants are not part of this gate.
* **`use_pynccl=False` is mandatory on NPU.** The all-gather /
  all-reduce path exercised here is the HCCL + gloo sidecar
  collective path.
* **Radix-cache retention (1 page) between baseline and post-case
  `free_pages`** is benign; `available_size` normalises back to
  baseline exactly.
* **Cold-start `_sync_get_memory()` variability** documented at
  Gate 4.9 remains outside `python/minisgl/`. The Gate 5.4
  outer shell loop authorises up to 2 retries with fresh
  `master_port` and 8 s sleep. On this run only attempt 1 was
  required.
* **Not a benchmark.** No throughput, latency, or cross-stack
  comparisons are made or implied.

## 13. Decision matrix

| Question | Answer |
|---|---|
| Does `/mnt/nvme/models/Qwen3-4B` exist on the validation host? | Yes |
| Does `LLM.__init__` return on both ranks under TP=2? | Yes (attempt 1) |
| Does weight load succeed on both ranks under TP=2? | Yes |
| Does post-load `check_integrity()` pass on both ranks? | Yes |
| Are the two prompts unequal-length (`[2, 12]`) on Qwen3-4B? | Yes |
| Does Case C `generate()` return? | Yes |
| Is `actual_output_tokens_per_request == [8, 8]` on both ranks? | Yes |
| Is rank 0 `output_texts[i]` byte-identical to rank 1 for every uid? | Yes |
| Is rank 0 `output_token_ids[i]` byte-identical to rank 1 for every uid? | Yes |
| Is `mixed_kv_decode_step_count >= 1` on both ranks? | Yes (7 on both) |
| Does every decode step show `query_lengths == [1, 1]` and `kv_lengths[0] != kv_lengths[1]`? | Yes |
| Does `available_tokens_after_case == baseline`? | Yes on both ranks |
| Is `deferred_abort_uids == 0`? | Yes on both ranks |
| Does `check_integrity()` pass post-case? | Yes on both ranks |
| Did the driver touch `python/minisgl/`? | No |
| Did the driver touch tests? | No |
| Did the driver claim performance superiority? | No |
| Did the driver compare against SGLang / vLLM / TGI? | No |
| Is `use_pynccl=False`? | Yes |
| Is the model limited to Qwen3-4B? | Yes |
| Is TP fixed at 2? | Yes |
| Is B fixed at 2 (unequal-length, mixed-KV decode)? | Yes |

**Verdict: PASS.**

## 14. Freeze boundary

The frozen artefacts for Gate 5.4 are:

* `scripts/gate5_4_qwen4b_tp2_b2_mixed_kv_decode.py`
* `docs/ascend_port/gate5.4_qwen4b_tp2_b2_mixed_kv_decode_verdict.md`

No files under `python/minisgl/` were modified at this gate. No
tests were modified at this gate. The freeze commit SHA is
recorded in this document header once the driver + verdict pair is
committed, and it is recorded on the `ascend-port` tip once the
branch is merged with `--no-ff`.
