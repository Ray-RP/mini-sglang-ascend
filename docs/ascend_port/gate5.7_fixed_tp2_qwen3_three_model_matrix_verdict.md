# Gate 5.7 Verdict — fixed-TP2 Qwen3 three-model capability matrix

**Gate ID:** 5.7 (fixed-TP2 Qwen3 three-model functional capability matrix on Ascend 910B1)
**Verdict:** PASS
**Branch:** `gate5.7-fixed-tp2-qwen3-three-model-matrix`
**Base commit:** `46a22b6` (tip of `ascend-port`, Gate 5.6 merge)
**Freeze commit:** `d5d62e4`
**Original pre-amend commit:** `b4cd3fa` (superseded by header-fix
amend; retained as provenance only — the final frozen SHA on the
`gate5.7-fixed-tp2-qwen3-three-model-matrix` branch tip is
`d5d62e4`)
**Date:** 2026-07-12
**Kind:** Functional three-model capability matrix — Qwen3-0.6B,
Qwen3-1.7B, and Qwen3-4B on 2 × Ascend 910B1 under fixed TP=2,
eager, `npu_fia`, bf16, greedy. Records structured per-rank
per-case JSONL for the six functional cases A/B/C/D/E/F,
per-model rank-0 `GATE5.7_MODEL_RESULT` footers, and a single
aggregated `GATE5.7_MATRIX_RESULT=PASS`. Neither runtime nor
tests are touched.

> **This is fixed-TP2 Ascend adaptation.**
> **It is not TP elasticity.**
> **It is not TP switching.**
> **It is not a benchmark.**
> **It is not a cross-stack comparison.**
> **It is not a performance-superiority claim.**
> Functional per-case status only (PASS / PARTIAL / BLOCKED); no
> timing statistics are collected. B > 2, TP > 2, TP elasticity,
> runtime TP switching, Graph Re-Linker, Tensor-Remap-Kernel,
> non-Qwen3 architectures, and Qwen3-14B / 32B / quantized / MoE
> variants are all out of scope. The Gate 2.2f documented
> "ragged + non-zero `cached_len` + `extend_len > 1`" unsupported
> FIA branch is intentionally avoided.

This verdict does not modify Gate 1 / 2.1 / 2.2 / 2.3 / 2.4 / 2.5 /
3.1 / 3.2 / 3.3 / 3.4 / 4.1 / 4.2 / 4.3 / 4.4 / 4.5 / 4.6 / 4.7 /
4.8 / 4.9 / 4.10 / 4.11 / 4.12 / 4.13 / 4.13a / 4.13b / 4.14 / 4.15
/ 4.15a / 5.1 / 5.2 / 5.3 / 5.4 / 5.4a / 5.5 / 5.5a / 5.6, does
not mutate release tag `v0.1.0a1`, does not touch the GitHub
Release / CHANGELOG / release notes / README, and does not extend
the Ascend port beyond the fixed-TP2 Qwen3 three-model
capability-matrix envelope. The only new artefacts at this gate
are one bring-up script and this verdict; no runtime source under
`python/minisgl/` is modified; no test file is modified.

---

## 1. Verdict summary

**PASS on all six cases (A / B / C / D / E / F) across both TP
ranks, for all three models (Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B).**

| Model | A | B | C | D | E | F | Model verdict |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen3-0.6B | PASS | PASS | PASS | PASS | PASS | PASS | **PASS** |
| Qwen3-1.7B | PASS | PASS | PASS | PASS | PASS | PASS | **PASS** |
| Qwen3-4B   | PASS | PASS | PASS | PASS | PASS | PASS | **PASS** |

Every per-case record on both ranks for all three models
reported:

* `actual_output_tokens_per_request == requested_max_new_tokens`
* `available_tokens_after_case == baseline_available_tokens`
* `deferred_abort_uids == 0`
* `cache_integrity_ok == true`
* `output_texts` and `output_token_ids` byte-identical rank 0 vs rank 1
* `status == "PASS"`

Per-model footers on rank 0:

```
GATE5.7_MODEL_RESULT model=Qwen3-0.6B PASS
GATE5.7_MODEL_RESULT model=Qwen3-1.7B PASS
GATE5.7_MODEL_RESULT model=Qwen3-4B   PASS
```

Aggregated footer:

```
GATE5.7_MATRIX_RESULT=PASS
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
| Models | Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B (all under `/mnt/nvme/models/`) |
| dtype | bf16 |
| TP | 2 (torchrun `--nproc_per_node=2`, one process pair per model) |
| Attention backend | `npu_fia` |
| Rendezvous | `MINISGL_DISTRIBUTED_ADDR=env://` (reuses torchrun's TCPStore) |
| Distributed collectives | HCCL primary, gloo sidecar; `use_pynccl=False` |
| CUDAGraph batch sizes | `cuda_graph_bs=[]` (torch_npu has no CUDAGraph) |
| Execution mode | eager |
| Sampling | greedy (temperature=0.0, top_k=1, top_p=1.0, `ignore_eos=True`) |
| `memory_ratio` | 0.85 |
| `page_size` | 16 |
| `max_running_req` | 4 |
| Cases per model | 6 (A / B / C / D / E / F) |
| Repeats per case | 1 (functional; no warmup, no measured repeats) |

**Per-model process isolation is mandatory.** Ascend
`DistributedInfo` / TP-group state is pinned globally at the
first `LLM()` init inside a process; a second `LLM()` in the
same process raises `RuntimeError: TP info has been set`. The
driver therefore accepts `--model-path` / `--model-name` and
processes exactly one model per torchrun invocation, and the
outer shell launches three independent torchrun process pairs
(one per model) on staggered `--master_port` bases.

## 3. Model path check

Present on the validation host container:

* `/mnt/nvme/models/Qwen3-0.6B`
* `/mnt/nvme/models/Qwen3-1.7B`
* `/mnt/nvme/models/Qwen3-4B`

All per-case JSONL records on both ranks logged `model_path` ==
the corresponding path.

## 4. Launch command

Same outer-shell retry pattern as Gates 4.10 / 4.11 / 4.12 / 4.13 /
4.14 / 5.1 / 5.2 / 5.3 / 5.4 / 5.5 / 5.6, wrapped in a per-model
outer loop that bumps `PORT_BASE` by 100 between models to avoid
`--master_port` collisions across model launches: sleep 8 s
between attempts, bump `--master_port` (PORT_BASE + ATTEMPT*10),
thread `--cold-start-attempt-id` and `--memory-sync-retry-note`
into the driver. **Attempt 1 succeeded cleanly for all three
models on the first port (30020 / 30120 / 30220) — no retry
needed.**

> **Public hygiene note:** remote host, username, password, and
> container identifiers are redacted in this document.

```bash
ssh -p <PORT> <USER>@<HOST> \
  "docker exec <CONTAINER> bash -c '
    set +e
    cd <REMOTE_PATH>/mini-sglang-ascend-gate5.7 &&
    mkdir -p logs
    MATRIX=PASS
    PORT_BASE=30010
    NOTE=""
    for MODEL in Qwen3-0.6B Qwen3-1.7B Qwen3-4B; do
      MODEL_RESULT=BLOCKED
      for ATTEMPT in 1 2 3; do
        PORT=$((PORT_BASE + ATTEMPT * 10))
        LOG=logs/gate5.7_${MODEL}_attempt${ATTEMPT}.log
        PYTHONPATH=./python:$PYTHONPATH torchrun --nproc_per_node=2 --master_port=$PORT \
          scripts/gate5_7_fixed_tp2_qwen3_three_model_matrix.py \
          --model-path /mnt/nvme/models/$MODEL --model-name $MODEL \
          --cold-start-attempt-id $ATTEMPT \
          --memory-sync-retry-note "$NOTE" > $LOG 2>&1
        if grep -q "GATE5.7_MODEL_RESULT model=$MODEL" $LOG && ! grep -q "Memory across TP ranks are imbalanced" $LOG; then
          MODEL_RESULT=$(grep "GATE5.7_MODEL_RESULT model=$MODEL" $LOG | awk '"'"'{print $NF}'"'"')
          cp $LOG logs/gate5.7_${MODEL}.log
          break
        fi
        sleep 8
      done
      PORT_BASE=$((PORT_BASE + 100))
      if [ "$MODEL_RESULT" != "PASS" ]; then
        if [ "$MATRIX" = "PASS" ]; then MATRIX=$MODEL_RESULT;
        elif [ "$MATRIX" = "PARTIAL" ] && [ "$MODEL_RESULT" = "BLOCKED" ]; then MATRIX=BLOCKED;
        fi
      fi
    done
    echo "GATE5.7_MATRIX_RESULT=$MATRIX"
  '"
```

## 5. Case matrix (locked, same for all three models)

| Case | Kind | Prompts (fixed strings) | `max_new_tokens` |
|---|---|---|---|
| A | `static_b1` | `"Paris is"` | `[8]` |
| B | `static_b1` | `"Paris is"` | `[16]` |
| C | `static_b2` | `"The capital of France is"`, `"The capital of Italy is"` | `[8, 8]` |
| D | `static_b2` | `"Paris is"`, `"The largest planet in our solar system by mass and volume is"` | `[8, 8]` |
| E | `static_b2` | `"Paris is"`, `"The largest planet in our solar system by mass and volume is"` | `[8, 8]` |
| F | `dynamic_admission` | `"Paris is"`, `"The largest planet in our solar system by mass and volume is"` | `[8, 8]` |

Case E reuses D's prompt pair but is scored against the mixed-KV
invariant (`kv_lengths[0] != kv_lengths[1]` on ≥ 1 pure decode
step) documented at Gate 5.4 §6. Case F uses the
`StaggeredLLM(LLM)` staged-admission pattern documented at
Gate 5.5 §6 — request B is revealed only after the metadata hook
observes A alone decoding (`batch_size == 1`,
`query_lengths == [1]`).

Tokenizer-dependent prompt token lengths do vary across the
three Qwen3 checkpoints; per-model actual lengths are recorded
in each per-case JSONL (`prompt_token_lengths` field).

## 6. Per-case outcomes

For each model, all 6 cases produced `status == "PASS"` on both
ranks with byte-identical `output_texts` and `output_token_ids`
across rank 0 and rank 1. Selected diagnostic fields:

### Qwen3-0.6B

| Case | `actual_output_tokens_per_request` | `mixed_kv_decode_step_count` | `timeline_contains_1_2_1` | `free_pages_before_after` |
|---|---|:---:|:---:|---|
| A | `[8]`     | 0 | false | `[59551, 59551]` |
| B | `[16]`    | 0 | false | `[59551, 59550]` |
| C | `[8, 8]`  | 0 | false | `[59551, 59550]` |
| D | `[8, 8]`  | 7 | false | `[59551, 59549]` |
| E | `[8, 8]`  | 7 | false | `[59551, 59549]` |
| F | `[8, 8]`  | 6 | **true** | `[59551, 59549]` |

### Qwen3-1.7B

| Case | `actual_output_tokens_per_request` | `mixed_kv_decode_step_count` | `timeline_contains_1_2_1` | `free_pages_before_after` |
|---|---|:---:|:---:|---|
| A | `[8]`     | 0 | false | `[58161, 58161]` |
| B | `[16]`    | 0 | false | `[58161, 58160]` |
| C | `[8, 8]`  | 0 | false | `[58161, 58160]` |
| D | `[8, 8]`  | 7 | false | `[58161, 58159]` |
| E | `[8, 8]`  | 7 | false | `[58161, 58159]` |
| F | `[8, 8]`  | 6 | **true** | `[58161, 58159]` |

### Qwen3-4B

| Case | `actual_output_tokens_per_request` | `mixed_kv_decode_step_count` | `timeline_contains_1_2_1` | `free_pages_before_after` |
|---|---|:---:|:---:|---|
| A | `[8]`     | 0 | false | `[43105, 43105]` |
| B | `[16]`    | 0 | false | `[43105, 43104]` |
| C | `[8, 8]`  | 0 | false | `[43105, 43104]` |
| D | `[8, 8]`  | 7 | false | `[43105, 43103]` |
| E | `[8, 8]`  | 7 | false | `[43105, 43103]` |
| F | `[8, 8]`  | 6 | **true** | `[43105, 43103]` |

Notes:

* Cases A / B / C do not exercise a batched decode with mismatched
  KV extents, so `mixed_kv_decode_step_count == 0` there is expected;
  the mixed-KV acceptance predicate only applies to Case E.
* Cases D / E / F share the same unequal-length prompt pair, so
  their pure-decode steps all satisfy `kv_lengths[0] != kv_lengths[1]`
  once both requests are in the joint decode. D and E produce 7
  such steps (steady joint decode); F produces 6 because A alone
  decodes once before B is admitted and B alone decodes once after
  A completes.
* Only Case F is scored against the timeline-`1→2→1` acceptance
  predicate; A/B/C/D/E return `false` by construction (they do
  not stage admissions).
* The `free_pages_before_after` deltas (0–2 pages) are retained
  evictable radix-cache prefix, absorbed by `available_size =
  free_slots + evictable_prefix_pages`, so `available_tokens_after
  _case == baseline_available_tokens` holds exactly on every row.

## 7. Case E mixed-KV evidence (all three models)

For all three models, Case E's first pure-decode snapshot was:

```
{'step_id': 1, 'kv_lengths': [3, 13]}
```

i.e. after A's 2-token prefill and B's 12-token prefill, the very
first joint decode step already exposes `kv_lengths[0] != kv_lengths[1]`
(3 vs 13). Subsequent 6 joint-decode steps continue the pattern
with `kv_lengths == [4,14], [5,15], [6,16], [7,17], [8,18], [9,19]`,
matching the Gate 5.4 §6 invariant that the mixed-batch decode
kernel must accept unequal per-request KV extents while every
request contributes exactly one query token per decode.

## 8. Case F dynamic admission evidence (all three models)

Batch timeline was byte-identical across the three models and both
ranks:

```
batch_timeline = [1, 1, 1, 2, 2, 2, 2, 2, 2, 1]
timeline_contains_1_2_1 = true
```

Explanation:

| Step | `batch_size` | Phase |
|---:|:---:|:---|
| 0 | 1 | A prefill (`query_lengths=[2]`) |
| 1 | 1 | A decode alone (triggers B reveal) |
| 2 | 1 | B prefill (`query_lengths=[12]`) |
| 3–8 | 2 | joint decode (`query_lengths=[1,1]`) |
| 9 | 1 | B decode alone after A completes |

The ordered subsequence `1 → 2 → 1` (Gate 5.5 §7 acceptance
predicate) is satisfied identically for all three models: HCCL
lock-step guarantees the `StaggeredLLM._staged_b` reveal happens
on the same tick on both ranks, and the model-dependent
tokenizer differences do not alter the timeline shape.

## 9. Cross-rank output equality (all three models, all six cases)

Verified via log post-processing over
`logs/gate5.7_{Qwen3-0.6B,Qwen3-1.7B,Qwen3-4B}.log`:

```
for model in Qwen3-0.6B Qwen3-1.7B Qwen3-4B:
  for case in A B C D E F:
    output_texts[rank=0]      == output_texts[rank=1]      ✓
    output_token_ids[rank=0]  == output_token_ids[rank=1]  ✓
```

36 records total (3 models × 6 cases × 2 ranks) — all matched
byte-for-byte per uid.

Qwen3-4B Cases D / E / F outputs `[" the capital of France, and
the capital", " Jupiter. It is a gas giant,"]` are byte-identical
to those recorded at Gates 5.3 / 5.4 / 5.5 / 5.6 for the same
prompt pair, confirming that the metadata hook and (for Case F)
the `StaggeredLLM` subclass are non-perturbing.

## 10. Per-model allocator baseline

Baseline captured after weight load, before Case A on each model.
Byte-identical across ranks per model.

| Model | `total_pages` | `baseline_free_pages` | `baseline_available_tokens` |
|---|---:|---:|---:|
| Qwen3-0.6B | 59552 | 59551 | 952816 |
| Qwen3-1.7B | 58162 | 58161 | 930576 |
| Qwen3-4B   | 43109 | 43105 | 689680 |

For every measured case on both ranks and all three models,
`available_tokens_after_case == baseline_available_tokens`. Small
deltas in `free_pages_after_case` (0–2 pages vs baseline) are
retained evictable radix-cache prefix — the same benign pattern
documented at Gates 4.5 / 4.10 / 4.11 / 4.12 / 4.13 / 4.14 / 5.3 /
5.4 / 5.5 / 5.6. `check_integrity()` confirmed radix-cache
linkage on every record.

Qwen3-4B baseline `689680 / 43105` differs from Gate 5.6's
`689744 / 43109` by a small handful of pages — normal
container-boot variance in `_sync_get_memory` accounting at cold
start. Both are consistent with the Gate 5.1 §7 KV allocation
band; Gate 5.7 makes no claim of exact reproducibility of the
baseline pool size across container restarts. What is preserved
exactly is the `available_tokens_after_case == baseline_available
_tokens` invariant within each run.

## 11. Cold-start retry note

**Attempt 1 succeeded cleanly for all three models.** Neither
the Qwen3-0.6B, Qwen3-1.7B, nor Qwen3-4B run tripped the
pre/post-load `_sync_get_memory()` imbalance check (2 GiB
tolerance at `python/minisgl/engine/engine.py:246`, unchanged
since Gate 4.14). Every per-case JSONL record on both ranks and
all three models logged `cold_start_attempt_id == 1` and
`memory_sync_retry_note == ""`.

The outer shell loop was authorised (per Gate 5.7 spec §5,
matching Gates 5.1 – 5.6) to sleep 8 s and re-launch with a
fresh `--master_port` up to 2 more times per model on cold-start
imbalance. It was not exercised on this run.

## 12. Public-hygiene grep summary

Post-authoring verification against the six known leaked
substring classes documented at Gate 4.13a / 4.13b (host / user /
password / container / IP / composite):

```
$ git grep -l -E "<OLD_SSHPASS_PATTERN>|<OLD_PASSWORD_PATTERN>|<OLD_HOST_PATTERN>|<OLD_CONTAINER_PATTERN>" \
    scripts/gate5_7_fixed_tp2_qwen3_three_model_matrix.py \
    docs/ascend_port/gate5.7_fixed_tp2_qwen3_three_model_matrix_verdict.md
(no output — zero files)
```

Loopback `127.0.0.1` and bind `0.0.0.0` do not appear in either
artefact. All host references in the verdict use placeholders
(`<HOST>`, `<PORT>`, `<USER>`, `<CONTAINER>`, `<REMOTE_PATH>`).

## 13. Regression evidence

Optional pytest per Gate 5.7 spec §9 was skipped — Gate 5.7
modifies zero runtime and zero test files. The Gate 4.13
regression measurement of `51 / 51 PASS` per-file pytest on
`tests/misc/` is unchanged by construction.

`git diff --check` on the freeze branch tip: clean.

## 14. Relationship to prior Qwen3 gates

Gate 5.7 does not replay Gates 5.1 – 5.6's per-case internal
allocator / timing plumbing; it consumes the same case shapes
those gates locked, but scores them under a *matrix* rather than
a *single-model per gate* frame:

* Gates 5.1 / 5.2 / 5.3 / 5.4 / 5.5 each cover exactly one
  Qwen3-4B case shape (B=1, B=2 equal, B=2 ragged, B=2 mixed-KV,
  dynamic admission 1→2→1);
* Gate 5.6 replays all six shapes back-to-back on Qwen3-4B *with*
  the timing-snapshot hook + warmup=1 + measured_repeats=3;
* Gate 4.13 recorded the analogous six-shape timing snapshot for
  Qwen3-1.7B under the same envelope;
* Gate 5.7 is the first three-model *coverage* gate: it proves
  that the same six-case functional envelope passes cleanly on
  Qwen3-0.6B *and* Qwen3-1.7B *and* Qwen3-4B, under the same
  fixed-TP2 constraints. No timing measurement is attached; the
  gate records only the functional PASS/PARTIAL/BLOCKED verdict
  per case per rank per model.

Gate 5.7's rows are **not** a Qwen3-0.6B vs Qwen3-1.7B vs
Qwen3-4B comparison and **not** a speedup or accuracy claim. They
are three independent internal reproducibility snapshots on the
same rig, proving each model separately runs cleanly through the
same six-case matrix under the same functional envelope.

## 15. Known limitations

* **Fixed-TP2, three-model, six pre-set cases only.** The three
  dense Qwen3 checkpoints ≤ 4B parameters have been proven; Qwen3
  variants at 14B / 32B / quantized / MoE (Qwen3-Next, Qwen3-Coder-Next,
  Qwen3-ASR-*, etc.) are **not** covered.
* **Functional-only.** No timing, throughput, latency, or
  steady-state claim is made. No cross-stack comparison to SGLang
  / vLLM / TGI is made or implied. The metadata-snapshot hook is
  script-local and identical across models.
* **TP=2 only.** TP=4 / TP=8, TP runtime elasticity, runtime TP
  switching, Graph Re-Linker, Tensor-Remap-Kernel are all out of
  scope.
* **Per-model process isolation is mandatory.** `DistributedInfo`
  / TP-group state is pinned globally at first `LLM()`;
  multi-model within one process raises
  `RuntimeError: TP info has been set`. The outer shell launches
  one torchrun process pair per model.
* **`use_pynccl=False` is mandatory on NPU.** The all-gather /
  all-reduce path exercised here is the HCCL + gloo sidecar
  collective path.
* **Radix-cache retention (0–2 pages)** between baseline and
  post-case `free_pages` is benign; `available_size` normalises
  back to baseline exactly on every measured record.
* **Cold-start `_sync_get_memory()` variability** documented at
  Gate 4.9 remains outside `python/minisgl/`. The Gate 5.7 outer
  shell loop authorises up to 2 retries per model with fresh
  `master_port` and 8 s sleep. On this run only attempt 1 was
  required for each model.
* **Qwen3-4B baseline drifts by a few pages** across gate runs
  (Gate 5.6 `689744 / 43109` vs Gate 5.7 `689680 / 43105`);
  Gate 5.7 makes no claim of exact reproducibility of the pool
  size across container restarts, only of the within-run
  `available_tokens_after_case == baseline_available_tokens`
  invariant.

## 16. Decision matrix

| Question | Answer |
|---|---|
| Do `/mnt/nvme/models/Qwen3-{0.6B,1.7B,4B}` all exist on the validation host? | Yes |
| Does `LLM.__init__` return on both ranks under TP=2 for all three models? | Yes (attempt 1 each) |
| Does weight load succeed on both ranks for all three models? | Yes |
| Does post-load `check_integrity()` pass on both ranks for all three models? | Yes |
| Did each of the 6 cases record per-rank JSONL on all three models? | Yes (36 records total) |
| Did every record produce `actual_output_tokens_per_request == requested_max_new_tokens`? | Yes |
| Did every record satisfy `available_tokens_after_case == baseline` on both ranks? | Yes |
| Was `deferred_abort_uids == 0` on every record? | Yes |
| Did `check_integrity()` pass on every record? | Yes |
| Did rank 0 and rank 1 `output_texts` / `output_token_ids` match by uid for every case × every model? | Yes (36-way byte-identical) |
| Did Case E produce ≥ 1 pure-decode snapshot with `query_lengths == [1,1]` and unequal `kv_lengths` for every model? | Yes (7 such steps per model) |
| Does Case F `batch_timeline` contain the ordered subsequence `[1,2,1]` for every model? | Yes (`[1,1,1,2,2,2,2,2,2,1]` identical across all three) |
| Did the driver touch `python/minisgl/`? | No |
| Did the driver touch tests? | No |
| Did the driver claim performance superiority? | No |
| Did the driver compare against SGLang / vLLM / TGI? | No |
| Is `use_pynccl=False`? | Yes |
| Is TP fixed at 2? | Yes |
| Are models limited to Qwen3-{0.6B, 1.7B, 4B}? | Yes |
| Is the case set limited to the fixed A/B/C/D/E/F six-case matrix? | Yes |

**Verdict: PASS.**

## 17. Freeze boundary

The frozen artefacts for Gate 5.7 are:

* `scripts/gate5_7_fixed_tp2_qwen3_three_model_matrix.py`
* `docs/ascend_port/gate5.7_fixed_tp2_qwen3_three_model_matrix_verdict.md`

No files under `python/minisgl/` were modified at this gate. No
tests were modified at this gate. The freeze commit SHA is
recorded in this document header once the driver + verdict pair
is committed, and it is recorded on the `ascend-port` tip once
the branch is merged with `--no-ff`.
