# Gate 5.5 Verdict — Qwen3-4B Fixed-TP2 Dynamic Admission B: 1 → 2 → 1

**Gate ID:** 5.5 (Qwen3-4B fixed-TP2 dynamic admission B: 1→2→1 on Ascend 910B1)
**Verdict:** PASS
**Branch:** `gate5.5-qwen4b-tp2-dynamic-admission-b1-b2-b1`
**Base commit:** `7f87788` (tip of `ascend-port`, Gate 5.4a merge)
**Freeze commit:** `d8ed3d2`
**Original pre-amend commit:** `9450ff5` (superseded by header-fix
amend; retained as provenance only — the final frozen SHA on the
`gate5.5-qwen4b-tp2-dynamic-admission-b1-b2-b1` branch tip is
`d8ed3d2`)
**Date:** 2026-07-12
**Kind:** Functional dynamic-admission bring-up — Qwen3-4B on 2 ×
Ascend 910B1 under fixed TP=2, eager, `npu_fia`, bf16, greedy.
Records structured JSONL per rank for Cases A/B/C plus a
`GATE5.5_RESULT=PASS` footer. No timing statistics, no repeats,
no warmup. Neither runtime nor tests are touched.

> **This is fixed-TP2 Ascend adaptation.**
> **It is not TP elasticity.**
> **It is not TP switching.**
> **It is not a benchmark.**
> `B` here is active request count (batch size), not TP degree.
> No timing statistics are collected; the wall-clock in the JSONL
> `generate_ms` field is diagnostic only. B=3, B: 1→2→3→2→1,
> TP > 2, TP elasticity, runtime TP switching, Graph Re-Linker,
> Tensor-Remap-Kernel, non-Qwen3 architectures, and Qwen3-14B / 32B
> / quantized / MoE variants are all out of scope. The Gate 2.2f
> documented "ragged + non-zero `cached_len` + `extend_len > 1`"
> unsupported FIA branch is intentionally avoided.

This verdict does not modify Gate 1 / 2.1 / 2.2 / 2.3 / 2.4 / 2.5 /
3.1 / 3.2 / 3.3 / 3.4 / 4.1 / 4.2 / 4.3 / 4.4 / 4.5 / 4.6 / 4.7 /
4.8 / 4.9 / 4.10 / 4.11 / 4.12 / 4.13 / 4.13a / 4.13b / 4.14 / 4.15
/ 4.15a / 5.1 / 5.2 / 5.3 / 5.4 / 5.4a, does not mutate release
tag `v0.1.0a1`, does not touch the GitHub Release / CHANGELOG /
release notes / README, and does not extend the Ascend port beyond
the fixed-TP2 dynamic-admission Qwen3-4B envelope. The only new
artefacts at this gate are one bring-up script and this verdict;
no runtime source under `python/minisgl/` is modified; no test
file is modified.

---

## 1. Verdict summary

**PASS on all three cases (A / B / C) across both TP ranks.**

| Case | Description | Rank 0 | Rank 1 |
|---|---|:---:|:---:|
| A | TP=2 init (`LLM.__init__` returns) | **PASS** | **PASS** |
| B | Qwen3-4B TP=2 weight load + post-load `check_integrity()` | **PASS** | **PASS** |
| C | Dynamic admission B: 1 → 2 → 1 (A alone → A+B joint decode → B alone) | **PASS** | **PASS** |

Every record on both ranks reported:

* `request_uids == [0, 1]` (A=0, B=1)
* `prompt_token_lengths == [2, 12]`
* `max_new_tokens_per_request == [8, 8]`
* `actual_output_tokens_per_request == [8, 8]`
* `batch_timeline == [1, 1, 1, 2, 2, 2, 2, 2, 2, 1]` (contains
  the ordered subsequence `1 → 2 → 1`)
* `admission_events == {"0": 0, "1": 2}` (B strictly after A)
* `completion_events == {"0": 8, "1": 9}` (A finishes first, B
  continues alone at step 9)
* `joint_decode_step_count == 6`
* `admission_status == "PASS"`, `timeline_status == "PASS"`
* `available_tokens_after_case == baseline_available_tokens`
* `deferred_abort_uids == 0`
* `cache_integrity_ok == true`
* `output_texts[i]` and `output_token_ids[i]` byte-identical
  across rank 0 and rank 1 for every uid

Footer:

```
GATE5.5_RESULT=PASS
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
| Request A | `"Paris is"` (uid 0), `max_tokens=8` |
| Request B | `"The largest planet in our solar system by mass and volume is"` (uid 1), `max_tokens=8` |
| Repeats | 1 (functional, no warmup, no measured repeats) |

## 3. Model path check

`/mnt/nvme/models/Qwen3-4B` present on the validation host
container. Both ranks logged `model_path` == the same path. The
on-disk config (`Qwen3ForCausalLM`, `hidden_size=2560`,
`num_hidden_layers=36`, `num_attention_heads=32`,
`num_key_value_heads=8` (GQA 32/8), `head_dim=128`,
`intermediate_size=9728`, `vocab_size=151936`,
`tie_word_embeddings=true`, `bf16`) was already documented at
Gate 5.1 §4 and is unchanged for this gate.

## 4. Launch command

Same outer-shell retry pattern as Gates 4.10 / 4.11 / 4.12 / 4.13 /
4.14 / 5.1 / 5.2 / 5.3 / 5.4: sleep 8 s between attempts, bump
`--master_port` (PORT_BASE=29850 + ATTEMPT*10), thread
`--cold-start-attempt-id` and `--memory-sync-retry-note` into the
driver. **Attempt 1 (port 29860) succeeded cleanly — no retry
needed.**

> **Public hygiene note:** remote host, username, password, and
> container identifiers are redacted in this document.

```bash
ssh -p <PORT> <USER>@<HOST> \
  "docker exec <CONTAINER> bash -c '
    set +e
    cd <REMOTE_PATH>/mini-sglang-ascend-gate5.5 &&
    mkdir -p logs &&
    PORT_BASE=29850
    NOTE=""
    for ATTEMPT in 1 2 3; do
      PORT=$((PORT_BASE + ATTEMPT * 10))
      LOG=logs/gate5.5_Qwen3-4B_attempt${ATTEMPT}.log
      PYTHONPATH=./python:$PYTHONPATH torchrun --nproc_per_node=2 --master_port=$PORT \
        scripts/gate5_5_qwen4b_tp2_dynamic_admission_b1_b2_b1.py \
        --model-path /mnt/nvme/models/Qwen3-4B \
        --cold-start-attempt-id $ATTEMPT \
        --memory-sync-retry-note "$NOTE" > $LOG 2>&1
      if grep -q GATE5.5_RESULT= $LOG && ! grep -q "Memory across TP ranks are imbalanced" $LOG; then
        cp $LOG logs/gate5.5_Qwen3-4B.log
        break
      fi
      IMB=$(grep "Memory across TP ranks are imbalanced" $LOG | head -1)
      NOTE="$NOTE | attempt $ATTEMPT port $PORT: $IMB"
      sleep 8
    done
  '"
```

## 5. Prompts and tokenization evidence

| uid | Prompt | Qwen3-4B tokenized length | `max_tokens` |
|---|---|---:|---:|
| 0 (A) | `"Paris is"` | 2 | 8 |
| 1 (B) | `"The largest planet in our solar system by mass and volume is"` | 12 | 8 |

Prompts reuse the Gate 5.3 / 5.4 pair unchanged. Both ranks
logged `prompt_token_lengths == [2, 12]` — different lengths mean
the joint-decode window shows clearly-diverging KV extents per
uid, which makes the mixed-batch decode invariant readable
alongside the admission timeline.

## 6. Batch timeline evidence (Case C)

Both ranks recorded the same 10-step trace via the script-local
`AscendFIABackend.prepare_metadata` monkey-patch (same pattern as
Gates 4.4 / 4.11 / 4.14 / 5.4 — the runtime source under
`python/minisgl/attention/ascend_fia.py` is unchanged).

| step | `batch_size` | `active_uids` | `query_lengths` | `kv_lengths` | Phase |
|---:|:---:|:---:|:---:|:---:|:---|
| 0 | 1 | `[0]` | `[2]` | `[2]` | A prefill |
| 1 | 1 | `[0]` | `[1]` | `[3]` | A decode alone → triggers admit |
| 2 | 1 | `[1]` | `[12]` | `[12]` | B prefill |
| 3 | 2 | `[0, 1]` | `[1, 1]` | `[4, 13]` | joint decode step 1 |
| 4 | 2 | `[0, 1]` | `[1, 1]` | `[5, 14]` | joint decode step 2 |
| 5 | 2 | `[0, 1]` | `[1, 1]` | `[6, 15]` | joint decode step 3 |
| 6 | 2 | `[0, 1]` | `[1, 1]` | `[7, 16]` | joint decode step 4 |
| 7 | 2 | `[0, 1]` | `[1, 1]` | `[8, 17]` | joint decode step 5 |
| 8 | 2 | `[0, 1]` | `[1, 1]` | `[9, 18]` | joint decode step 6 (A's last) |
| 9 | 1 | `[1]` | `[1]` | `[19]` | B decode alone (shrink to 1) |

`batch_timeline == [1, 1, 1, 2, 2, 2, 2, 2, 2, 1]` on both ranks.
The required subsequence `1 → 2 → 1` appears in order (first `1`
at step 0, first `2` at step 3, final `1` at step 9). Both ranks
made identical admission decisions — HCCL lockstep guarantees the
`StaggeredLLM._staged_b` reveal happens on the same tick, and the
uid counter increments symmetrically.

## 7. Admission / completion evidence

| uid | Role | `first_seen` step | `last_seen` step | Total steps active | Output tokens generated |
|---|---|---:|---:|---:|---:|
| 0 (A) | prefills first, admits alone | 0 | 8 | 8 | 8 |
| 1 (B) | admitted after A's first decode | 2 | 9 | 8 | 8 |

Both ranks logged `admission_events == {"0": 0, "1": 2}` and
`completion_events == {"0": 8, "1": 9}`, satisfying the required
inequalities:

* `admission_events["0"] < admission_events["1"]` — B strictly
  admitted after A (`0 < 2`)
* `completion_events["0"] < completion_events["1"]` — A finishes
  first (last active at step 8), then B continues alone at step 9
  (shrink back to `batch_size == 1`)
* `joint_decode_step_count == 6` — the joint window (steps 3–8)
  has 6 pure-decode steps with `batch_size == 2` and
  `query_lengths == [1, 1]`

## 8. Per-rank output evidence (Case C)

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
Gates 5.3 / 5.4 for the same prompt pair, confirming that
staggered admission does not perturb the greedy trajectory of
either request — A's decodes at steps 1, 3, 4, 5, 6, 7, 8 recreate
the exact token stream A produces in the joint-arrival batch;
similarly for B at steps 3–9.

## 9. Per-rank allocator evidence

Baseline captured after weight load, before Case C. Post-case
`available_size` returned to baseline exactly on both ranks.

| Field | Rank 0 | Rank 1 |
|---|---:|---:|
| `total_pages` | 43110 | 43110 |
| `baseline_free_pages` | 43110 | 43110 |
| `baseline_available_tokens` | 689760 | 689760 |
| `free_pages_before_after` (Case C) | `[43110, 43109]` | `[43110, 43109]` |
| `free_pages_after_case` | 43109 | 43109 |
| `available_tokens_after_case` | 689760 | 689760 |
| `deferred_abort_uids` | 0 | 0 |
| `cache_integrity_ok` | true | true |

Baseline is bit-identical across ranks. The 1-page reduction in
`free_pages_after_case` (43109 vs baseline 43110) is retained
evictable radix-cache prefix for one of the two prompt strings —
the same benign pattern documented at Gates 4.5 / 4.10 / 4.11 /
4.12 / 4.13 / 4.14 / 5.3 / 5.4. It is absorbed by
`available_size = free_slots + evictable_prefix_pages`, so
`available_tokens_after_case == baseline_available_tokens`
(689760 == 689760) holds exactly on both ranks. `check_integrity()`
confirms radix-cache linkage; not a page leak. Baseline
(`689760` / `43110`) matches Gates 5.2 / 5.3 exactly.

`generate_ms` (diagnostic only — **not** a Gate 5.5 evidence
field): rank 0 = 2864.31 ms, rank 1 = 2870.17 ms. Gate 5.5 makes
no timing claim.

## 10. Cold-start retry note

**Attempt 1 succeeded cleanly.** The run did not trip the
pre/post-load `_sync_get_memory()` imbalance check (2 GiB tolerance
at `python/minisgl/engine/engine.py:246`, unchanged since Gate
4.14). Both ranks reported `cold_start_attempt_id == 1` and
`memory_sync_retry_note == ""`.

The outer shell loop was authorised (per Gate 5.5 spec §5, matching
Gates 5.1 / 5.2 / 5.3 / 5.4) to sleep 8 s and re-launch with a
fresh `--master_port` up to 2 more times on cold-start imbalance.
It was not exercised on this run.

## 11. Public-hygiene grep summary

Post-authoring verification against the six known leaked substring
classes documented at Gate 4.13a / 4.13b (host / user / password /
container / IP / composite):

```
$ git grep -l -E "<OLD_SSHPASS_PATTERN>|<OLD_PASSWORD_PATTERN>|<OLD_HOST_PATTERN>|<OLD_CONTAINER_PATTERN>" \
    scripts/gate5_5_qwen4b_tp2_dynamic_admission_b1_b2_b1.py \
    docs/ascend_port/gate5.5_qwen4b_tp2_dynamic_admission_b1_b2_b1_verdict.md
(no output — zero files)
```

Loopback `127.0.0.1` and bind `0.0.0.0` do not appear in either
artefact. All host references in the verdict use placeholders
(`<HOST>`, `<PORT>`, `<USER>`, `<CONTAINER>`, `<REMOTE_PATH>`).

## 12. Regression evidence

Optional pytest per Gate 5.5 spec §9 was skipped — Gate 5.5
modifies zero runtime and zero test files. The Gate 4.13
regression measurement of `51 / 51 PASS` per-file pytest on
`tests/misc/` is unchanged by construction.

`git diff --check` on the freeze branch tip: clean.

## 13. Known limitations

* **Fixed-TP2, single-model, B: 1 → 2 → 1 timeline only.**
  Qwen3-4B has been proven at B=1 (Gate 5.1), B=2 equal-length
  (Gate 5.2), B=2 ragged prefill (Gate 5.3), B=2 mixed-KV decode
  (Gate 5.4), and now dynamic admission B: 1→2→1 (this gate).
  B=3, B: 1→2→3→2→1, dynamic grow-shrink beyond 2, and dynamic
  eviction/preemption on Qwen3-4B are **not** covered. Those
  capabilities will require their own follow-up gates modelled on
  Gate 4.6 / 4.7.
* **Staggered admission is driven by a script-local
  `StaggeredLLM` subclass + a passive `prepare_metadata`
  monkey-patch.** The subclass overrides `offline_receive_msg` to
  gate the reveal of request B, and the hook only records
  `FIAMetadata` after the original `prepare_metadata` sets it.
  Neither mutates runtime state. Byte-identical outputs vs
  Gates 5.3 / 5.4 confirm the hook + subclass are non-perturbing.
* **No timing evidence.** `generate_ms` is a diagnostic dump only.
  Gate 5.5 makes no throughput, latency, or steady-state claim.
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
  Gate 4.9 remains outside `python/minisgl/`. The Gate 5.5
  outer shell loop authorises up to 2 retries with fresh
  `master_port` and 8 s sleep. On this run only attempt 1 was
  required.
* **Not a benchmark.** No throughput, latency, or cross-stack
  comparisons are made or implied.

## 14. Decision matrix

| Question | Answer |
|---|---|
| Does `/mnt/nvme/models/Qwen3-4B` exist on the validation host? | Yes |
| Does `LLM.__init__` return on both ranks under TP=2? | Yes (attempt 1) |
| Does weight load succeed on both ranks under TP=2? | Yes |
| Does post-load `check_integrity()` pass on both ranks? | Yes |
| Are the two prompts unequal-length (`[2, 12]`) on Qwen3-4B? | Yes |
| Does `run_forever()` drain cleanly? | Yes |
| Is `actual_output_tokens_per_request == [8, 8]` on both ranks? | Yes |
| Is rank 0 `output_texts[i]` byte-identical to rank 1 for every uid? | Yes |
| Is rank 0 `output_token_ids[i]` byte-identical to rank 1 for every uid? | Yes |
| Does `batch_timeline` contain the ordered subsequence `1 → 2 → 1`? | Yes on both ranks (`[1,1,1,2,2,2,2,2,2,1]`) |
| Is request B admitted strictly after request A? | Yes on both ranks (A step 0 < B step 2) |
| Is `joint_decode_step_count >= 1`? | Yes (6 on both ranks) |
| Does the remaining request continue after its sibling leaves? | Yes (B alone at step 9) |
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
| Is the timeline limited to B: 1 → 2 → 1? | Yes |

**Verdict: PASS.**

## 15. Freeze boundary

The frozen artefacts for Gate 5.5 are:

* `scripts/gate5_5_qwen4b_tp2_dynamic_admission_b1_b2_b1.py`
* `docs/ascend_port/gate5.5_qwen4b_tp2_dynamic_admission_b1_b2_b1_verdict.md`

No files under `python/minisgl/` were modified at this gate. No
tests were modified at this gate. The freeze commit SHA is
recorded in this document header once the driver + verdict pair is
committed, and it is recorded on the `ascend-port` tip once the
branch is merged with `--no-ff`.
