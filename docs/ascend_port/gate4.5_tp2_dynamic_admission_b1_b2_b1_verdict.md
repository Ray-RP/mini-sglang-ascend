# Gate 4.5 Verdict — TP=2 dynamic admission B: 1 → 2 → 1 (Qwen3-0.6B)

**Gate ID:** 4.5 (TP=2 Ascend dynamic admission B: 1 → 2 → 1 on Qwen3-0.6B)
**Verdict:** PASS
**Branch:** `gate4.5-tp2-dynamic-admission-b1-b2-b1`
**Base commit:** `e942b11` (tip of `ascend-port`, Gate 4.4 merge)
**Freeze commit:** `08688d2`
**Date:** 2026-07-11
**Kind:** Real-hardware Ascend 910B1 TP=2 dynamic-admission proof —
two ranks × Qwen3-0.6B × request A starts alone, request B arrives
mid-generation, both jointly decode for six steps, A finishes first,
B finishes alone one step later. Captured per-forward-step
`FIAMetadata` snapshots show a `batch_timeline` of
`[1, 1, 1, 2, 2, 2, 2, 2, 2, 1]` on both ranks — proving the required
`B: 1 → 2 → 1` invariant end-to-end with per-rank allocator
invariants held and cross-rank per-uid output equality.

This verdict does not modify Gate 1 / 2.1 / 2.2 / 2.3 / 2.4 / 2.5 /
3.1 / 3.2 / 3.3 / 3.4 / 4.1 / 4.2 / 4.3 / 4.4, does not mutate
release tag `v0.1.0a1`, does not touch the GitHub Release, CHANGELOG,
or release notes, and does not extend the Ascend port to TP > 2,
TP=2 B > 2, dynamic admission with B: 1 → 2 → 3 → 2 → 1, TP=2
timing benchmark, non-Qwen3 architectures, or Qwen3-1.7B / Qwen3-4B
/ 14B / 32B / quantized / MoE variants. The result produced
(`actual_output_tokens_per_request == [8, 8]`,
`batch_timeline == [1, 1, 1, 2, 2, 2, 2, 2, 2, 1]`) is a
**correctness proof** for the minimal dynamic-admission shape; no
throughput / latency / tokens-per-second claim is made. The only
new artefacts at this gate are one bring-up script and this
verdict; no runtime source under `python/minisgl/` is modified. The
staggered-admission driver uses a script-local subclass override on
`LLM.offline_receive_msg` plus a script-local monkey-patch on
`AscendFIABackend.prepare_metadata` — both live in the driver
process and do not touch the checked-in package.

---

## 1. Verdict summary

**PASS on all three cases across both ranks.**

| Case | Description | rank 0 | rank 1 |
|---|---|---|---|
| A | init-only smoke — `_init_communication` returns, both ranks report `init_status=PASS` | **PASS** | **PASS** |
| B | model-load smoke — per-rank weight sharding + post-load `check_integrity()` | **PASS** | **PASS** |
| C | dynamic admission timeline — request A starts alone, request B admitted mid-generation, at least one joint decode step, one request finishes first, sibling continues | **PASS** | **PASS** |

Cases A / B still collapse into the same `LLM` boot (`set_tp_info` is
a one-shot); reaching a successful `_snapshot(cache_manager)` after
`LLM.__init__` returns proves init + load simultaneously.

Timeline invariants proven on both ranks:

* `request_uids == [0, 1]` (A=uid 0 admitted first, B=uid 1 admitted second)
* `prompt_token_lengths == [2, 12]` — deliberately unequal so the two
  requests occupy different KV budgets and cannot silently collapse
* `admission_events == {"0": 0, "1": 2}` — B's first appearance in a
  batch is strictly after A's first appearance (0 < 2)
* `completion_events == {"0": 8, "1": 9}` — A leaves the batch first
  (last seen at step 8), B continues alone for one more step (step 9)
* `batch_timeline == [1, 1, 1, 2, 2, 2, 2, 2, 2, 1]` — the required
  `B: 1 → 2 → 1` sub-sequence is present as `1 (steps 0..2) → 2
  (steps 3..8) → 1 (step 9)`
* `joint_decode_step_count == 6` — six consecutive decode steps have
  `batch_size == 2` with `query_lengths == [1, 1]`

Post-case allocator invariants held on both ranks:

* `available_tokens_after_case == baseline_available_tokens` (952880 on both ranks)
* `deferred_abort_uids == 0`
* `cache_integrity_ok == true`

Structured logs (both ranks, JSON pretty-printed excerpt for rank 0;
rank 1 emits the byte-identical trace with `device: "npu:1"`):

```
GATE4.5_JSONL rank=0 {
  "rank": 0, "world_size": 2, "tp_size": 2,
  "device": "npu:0",
  "request_uids": [0, 1],
  "prompt_token_lengths": [2, 12],
  "max_new_tokens_per_request": [8, 8],
  "baseline_available_tokens": 952880, "baseline_free_pages": 59555, "total_pages": 59555,
  "init_status": "PASS", "load_status": "PASS",
  "prefill_status": "PASS", "decode_status": "PASS",
  "admission_status": "PASS", "timeline_status": "PASS",
  "batch_timeline": [1, 1, 1, 2, 2, 2, 2, 2, 2, 1],
  "admission_events": {"0": 0, "1": 2},
  "completion_events": {"0": 8, "1": 9},
  "joint_decode_step_count": 6,
  "all_step_snapshots": [
    {"step_id": 0, "batch_size": 1, "active_uids": [0],    "query_lengths": [2],     "kv_lengths": [2]},
    {"step_id": 1, "batch_size": 1, "active_uids": [0],    "query_lengths": [1],     "kv_lengths": [3]},
    {"step_id": 2, "batch_size": 1, "active_uids": [1],    "query_lengths": [12],    "kv_lengths": [12]},
    {"step_id": 3, "batch_size": 2, "active_uids": [0, 1], "query_lengths": [1, 1],  "kv_lengths": [4, 13]},
    {"step_id": 4, "batch_size": 2, "active_uids": [0, 1], "query_lengths": [1, 1],  "kv_lengths": [5, 14]},
    {"step_id": 5, "batch_size": 2, "active_uids": [0, 1], "query_lengths": [1, 1],  "kv_lengths": [6, 15]},
    {"step_id": 6, "batch_size": 2, "active_uids": [0, 1], "query_lengths": [1, 1],  "kv_lengths": [7, 16]},
    {"step_id": 7, "batch_size": 2, "active_uids": [0, 1], "query_lengths": [1, 1],  "kv_lengths": [8, 17]},
    {"step_id": 8, "batch_size": 2, "active_uids": [0, 1], "query_lengths": [1, 1],  "kv_lengths": [9, 18]},
    {"step_id": 9, "batch_size": 1, "active_uids": [1],    "query_lengths": [1],     "kv_lengths": [19]}
  ],
  "actual_output_tokens_per_request": [8, 8],
  "output_texts": [" a city in the southern part of France", "...? A. Mercury B. Venus"],
  "available_tokens_after_case": 952880, "free_pages_after_case": 59554,
  "free_pages_before_after": [59555, 59554],
  "deferred_abort_uids": 0, "cache_integrity_ok": true,
  "status": "PASS", "failure_stage": null
}
```

Rank 0 and rank 1 produce byte-identical snapshot traces, identical
`admission_events` and `completion_events`, identical
`joint_decode_step_count`, identical `output_texts`, identical
allocator numbers. Cross-rank symmetry across every field is itself
a signal that the TP=2 scheduler / paged-cache / dynamic-admission
path is not diverging between ranks.

---

## 2. Envelope (locked at this gate)

```
Hardware:          Ascend 910B1 (2 dies × 64 GiB HBM)
                   rank 0 → npu:0, rank 1 → npu:1
Container:         <CONTAINER> on remote <HOST>:<PORT>
Software:          Python 3.11.14
                   torch 2.9.0+cpu
                   torch_npu 2.9.0.post1+gitee7ba04
                   CANN 8.5.1 (compiler build 20250725)
Parallelism:       TP=2 (world_size=2, rank ∈ {0,1})
Execution:         eager (cuda_graph_bs=[]; torch_npu has no CUDAGraph)
Attention backend: npu_fia (FIA prefill cached_len==0 branch + FIA
                   mixed-KV decode branch)
page_size:         16
memory_ratio:      0.85
max_running_req:   4
Sampling:          greedy (temperature=0.0, top_k=1, top_p=1.0, ignore_eos=True)
Batching:          B: 1 → 2 → 1 dynamic (A alone, then A+B, then B alone)
Requests:          A: max_tokens=8, prompt="Paris is"                      (2 tokens)
                   B: max_tokens=8, prompt="The largest planet in our
                                            solar system by mass and
                                            volume is"                     (12 tokens)
Model:             /mnt/nvme/models/Qwen3-0.6B (dense bf16 Qwen3ForCausalLM)
Launcher:          torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0
                            --master_addr=127.0.0.1 --master_port=29406
Distributed init:  MINISGL_DISTRIBUTED_ADDR=env:// (reuses torchrun store)
                   backend=hccl (primary) + gloo (sidecar via new_group)
                   use_pynccl=False (mandatory on NPU)
Driver:            scripts/gate4_5_tp2_dynamic_admission_b1_b2_b1.py
                   per-rank worker under torchrun; both ranks
                   independently instantiate a script-local
                   ``StaggeredLLM(LLM)`` subclass whose overridden
                   ``offline_receive_msg`` reveals request B only
                   after the metadata hook observes ≥1 decode step
                   with A alone. Both ranks run identical forwards
                   in HCCL lockstep, so both ranks admit B on the
                   same tick.
Metadata capture:  Script-only monkey-patch. Wraps
                   AscendFIABackend.prepare_metadata for the driver
                   process only; the runtime source under
                   python/minisgl/attention/ascend_fia.py is not
                   modified. The same hook records the per-step
                   FIAMetadata AND flips the admission-ready flag.
```

---

## 3. Launch command

Executed on remote container `<CONTAINER>` at working directory
`/mnt/nvme/LR-606/mini-sglang-ascend-gate45`.

```bash
PYTHONPATH=python torchrun \
  --nproc_per_node=2 --nnodes=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port=29406 \
  scripts/gate4_5_tp2_dynamic_admission_b1_b2_b1.py
```

The script's structured stdout is the primary evidence; each rank
emits exactly one `GATE4.5_JSONL rank=<r> {...}` line containing the
full per-forward-step snapshot list, admission events, and
completion events.

---

## 4. Prompt / uid mapping evidence

| Request | uid | prompt | tokens | max_tokens |
|---|---|---|---|---|
| A | 0 | `"Paris is"` | 2 | 8 |
| B | 1 | `"The largest planet in our solar system by mass and volume is"` | 12 | 8 |

Both ranks tokenize the same prompts (same tokenizer, same
`self.counter` incrementing) and both assign uid 0 to A and uid 1 to
B. The script asserts `request_uids == [0, 1]` before consuming the
snapshot list.

The staggered-admission mechanism is entirely offline: `LLM` runs in
`offline_mode=True`, so `receive_msg` is bound to
`offline_receive_msg` by `SchedulerIOMixin.__init__`. The subclass
overrides `offline_receive_msg` — same MRO binding, so the wrapper
gets called each scheduler tick. On each tick the wrapper checks
`state.ready_to_admit_b`; when the flag is set (by the metadata
hook, which is the only mutator) and B has not yet been revealed, it
appends `(prompt_b, sp_b)` to `self.pending_requests` and delegates
to `super().offline_receive_msg`. The next call falls through the
parent's normal loop; B enters the scheduler as a `UserMsg` and
lands in `prefill_manager`.

---

## 5. Batch timeline evidence (the primary Gate 4.5 evidence)

Captured per-forward-step snapshots (both ranks agree byte-for-byte):

| step_id | batch_size | active_uids | query_lengths | kv_lengths | interpretation |
|---:|---:|---|---|---|---|
| 0 | 1 | `[0]`    | `[2]`    | `[2]`     | prefill A alone (cached_len==0 branch) |
| 1 | 1 | `[0]`    | `[1]`    | `[3]`     | decode A alone — **hook flips `ready_to_admit_b`** here |
| 2 | 1 | `[1]`    | `[12]`   | `[12]`    | prefill B alone (A is in decode_manager but prefill wins scheduling) |
| 3 | 2 | `[0, 1]` | `[1, 1]` | `[4, 13]` | first joint decode step |
| 4 | 2 | `[0, 1]` | `[1, 1]` | `[5, 14]` | joint decode |
| 5 | 2 | `[0, 1]` | `[1, 1]` | `[6, 15]` | joint decode |
| 6 | 2 | `[0, 1]` | `[1, 1]` | `[7, 16]` | joint decode |
| 7 | 2 | `[0, 1]` | `[1, 1]` | `[8, 17]` | joint decode |
| 8 | 2 | `[0, 1]` | `[1, 1]` | `[9, 18]` | last joint decode; A finishes after this step (8 total outputs) |
| 9 | 1 | `[1]`    | `[1]`    | `[19]`    | B decodes alone; B finishes after this step (8 total outputs) |

Derived counts (both ranks agree):

* `batch_timeline` = `[1, 1, 1, 2, 2, 2, 2, 2, 2, 1]` — length 10
  (1 prefill A + 1 decode A + 1 prefill B + 6 joint decode + 1 solo
  decode B)
* Contains `B: 1 → 2 → 1` subsequence in order: `1` at step 0..2,
  `2` at step 3..8, `1` at step 9 → **timeline_status = PASS**
* `admission_events` = `{"0": 0, "1": 2}` — A's first batch entry
  is at step 0, B's first batch entry is at step 2; strictly
  admissioned after A → **admission_status = PASS**
* `completion_events` = `{"0": 8, "1": 9}` — A's last batch entry
  is at step 8, B's last batch entry is at step 9; A leaves first,
  B continues alone
* `joint_decode_step_count` = 6 — six decode steps have
  `batch_size == 2` with `query_lengths == [1, 1]`, satisfying the
  Gate 4.5 requirement of ≥1 joint decode step

The KV-length delta on joint decode steps is constant: `13-4=14-5=…=18-9=10`,
matching `long_prompt_len - short_prompt_len + (b_first_decode_kv - a_first_decode_kv)`
after B's prefill runs. This is the same mixed-KV invariant attested
by Gate 4.4, now observed in a batch that dynamically grew from
B=1 to B=2 to B=1.

Rank 0 and rank 1 produce byte-identical snapshot lists — the two
ranks see the same scheduler shape and pass identical
`(query_seq_lens, kv_seq_lens, active_uids)` to their FIA operator
on every step. Because the scheduler is process-local (each rank
runs its own copy), this cross-rank agreement is a real signal that
the paged-cache / extend-len / device-len bookkeeping is
deterministic and symmetric across ranks under dynamic admission.

---

## 6. Admission / completion event evidence

| Event | uid | step_id | forward-pass semantics |
|---|---|---|---|
| A admitted | 0 | 0 | request A first appears in a batch (prefill) |
| B admitted | 1 | 2 | request B first appears in a batch (prefill), 2 steps after A |
| A completed | 0 | 8 | request A's last batch entry; produces its 8th output token |
| B completed | 1 | 9 | request B's last batch entry; produces its 8th output token |

Ordering invariants:

* `admission_events["1"] > admission_events["0"]`  →  `2 > 0` ✓
  — B was admitted strictly after A (staggered arrival).
* `completion_events["1"] > completion_events["0"]`  →  `9 > 8` ✓
  — A finished first; B ran solo for exactly one more step
  (proving the sibling continues after the first request leaves).
* `completion_events["0"] > admission_events["1"]`  →  `8 > 2` ✓
  — A was still active when B was admitted (they overlapped;
  admission was truly dynamic, not sequential).

Wall-clock informational (not a timing claim): `run_forever()`
drained in 2663.58 ms on rank 0 and 2612.74 ms on rank 1 for the
full dynamic-admission trace with metadata capture. Not a
benchmark; the metadata-snapshot hook adds unbudgeted per-step
overhead by design.

---

## 7. Per-rank output evidence

| Field | rank 0 | rank 1 |
|---|---|---|
| `device` | `npu:0` | `npu:1` |
| `request_uids` | `[0, 1]` | `[0, 1]` |
| `prompt_token_lengths` | `[2, 12]` | `[2, 12]` |
| `max_new_tokens_per_request` | `[8, 8]` | `[8, 8]` |
| `init_status` | `PASS` | `PASS` |
| `load_status` | `PASS` | `PASS` |
| `prefill_status` | `PASS` | `PASS` |
| `decode_status` | `PASS` | `PASS` |
| `admission_status` | `PASS` | `PASS` |
| `timeline_status` | `PASS` | `PASS` |
| `actual_output_tokens_per_request` | `[8, 8]` | `[8, 8]` |
| `output_texts[0]` (uid 0 = A) | `" a city in the southern part of France"` | `" a city in the southern part of France"` |
| `output_texts[1]` (uid 1 = B) | `"...? A. Mercury B. Venus"` | `"...? A. Mercury B. Venus"` |
| `status` | `PASS` | `PASS` |
| `failure_stage` | `null` | `null` |

Cross-rank per-uid output equality is exact (byte-identical string)
on both uids. The two output strings match the Gate 4.3 and Gate 4.4
per-uid outputs under the same prompts + envelope, even though the
batch topology at Gate 4.5 is dynamic (grow from B=1 to B=2, then
shrink back to B=1). This confirms:

* Dynamic admission of B mid-generation did not corrupt A's KV pages
  or perturb A's decode trajectory — A produces the same tokens it
  would have produced in a solo B=1 run.
* Late arrival + mid-batch prefill of B did not perturb B's own
  decode trajectory — B produces the same tokens it would have
  produced in a solo B=1 run.
* All-gathered `ParallelLMHead` continues to reassemble a
  fully-replicated logits tensor identically on both ranks under
  dynamic admission.
* Per-uid `device_len` bookkeeping stays correct across the grow
  and shrink transitions on both ranks.

---

## 8. Per-rank allocator evidence

| rank | baseline_available_tokens | after case | baseline_free_pages | after case | free_pages_before_after | total_pages | deferred_abort_uids | cache_integrity_ok |
|---|---|---|---|---|---|---|---|---|
| 0 | 952880 | 952880 | 59555 | 59554 | `[59555, 59554]` | 59555 | 0 | true |
| 1 | 952880 | 952880 | 59555 | 59554 | `[59555, 59554]` | 59555 | 0 | true |

Interpretation:

* Baseline `available_tokens = 952880` per rank — identical to
  Gates 4.1, 4.2, 4.3, 4.4 baselines under the same TP=2 envelope.
* `available_tokens` returned exactly to baseline on both ranks
  after the dynamic-admission batch completed — the primary
  allocator invariant. Neither the grow (B=1 → 2) nor the shrink
  (B=2 → 1) leaked any per-request pages; both uids' prefill +
  decode pages were fully released once each request hit
  `max_tokens=8`.
* `free_pages` drift of 1 (59555 → 59554) on both ranks — one
  evictable radix-cache prefix page retained after the batch,
  matching the Gate 4.3 / 4.4 pattern under the same prompts. Per
  Gate 3.1 §4, `available_size = free_slots + evictable_prefix_pages`
  (in tokens), so 1 evictable page ≈ 16 tokens are counted back
  into `available_tokens`, keeping the token-level invariant exact
  while raw `free_pages` is off by 1.
* `deferred_abort_uids == 0` — no abort path was entered.
* `cache_integrity_ok == true` — allocator invariants held
  throughout, including through the mid-batch prefill of B, the six
  joint decode steps, and the transition back to solo B decode.

Both ranks show the exact same drift pattern (identical baseline,
identical after-case, identical evictable-page retention). This
symmetry across ranks under dynamic admission is itself a signal
that the TP=2 allocator does not diverge between ranks even when
requests arrive on different scheduler ticks.

---

## 9. First failing stage

**None.** No failure surfaced on the smoke run. Cases A / B / C all
reported `PASS` on both ranks. `failure_stage` and
`failure_trace_summary` are `null` on both ranks.

No source-file change under `python/minisgl/` was required at this
gate. The Gate 4.1 minimum fixes (`LLM.__init__` `tp_info` kwarg,
`EngineConfig.distributed_addr` env-var override) already inherited
from `ascend-port` at `e942b11` provided the full driver + launcher
path. The staggered admission is entirely expressed at the driver
layer via a script-local `LLM` subclass whose overridden
`offline_receive_msg` decides which pending requests to reveal to
the scheduler on each tick.

The metadata-snapshot hook is a script-only monkey-patch on the
imported `AscendFIABackend.prepare_metadata` classmethod. It runs
after the original method sets `batch.attn_metadata` and never
mutates the metadata; the underlying runtime source is not
modified.

---

## 10. Regression evidence

Per-file rows on the same container / same tree:

| File                                         | Rows |
|---|---|
| `test_scheduler_abort_ack.py`                | 8 passed |
| `test_scheduler_overlap_abort_fence.py`      | 7 passed |
| `test_scheduler_prepare_batch_txn.py`        | 5 passed |
| `test_engine_forward_sampler_atomic.py`      | 5 passed |
| `test_scheduler_shutdown_drain.py`           | 8 passed |
| `test_exposed_path_abort_ack.py`             | 2 passed |
| `test_shell_cancel_cleanup.py`               | 2 passed |
| `test_pyproject_config.py`                   | 14 passed |
| **Total**                                    | **51 passed** |

Matches the recorded counts of Gate 2.5, Gate 3.1, Gate 3.2, Gate
3.3, Gate 3.4, Gate 4.1, Gate 4.2, Gate 4.3, Gate 4.4.

Zero source-code changes under `python/minisgl/` at this gate — the
regression rows are expected to be unchanged by construction; the
run confirms it.

---

## 11. Support matrix delta (Gate 4.4 → Gate 4.5)

| Capability                                                | Gate 4.4 | Gate 4.5 |
|---|---|---|
| TP=2 driver support in offline `LLM`                      | PASS | PASS (unchanged) |
| TP=2 launcher rendezvous under torchrun (`env://`)        | PASS | PASS (unchanged) |
| TP=2 Qwen3-0.6B B=2 ragged prefill (cached_len==0) N=8    | PASS | not re-attested |
| TP=2 Qwen3-0.6B B=2 mixed-KV decode explicit evidence     | PASS | not re-attested |
| TP=2 FIA ragged-prefill dispatch                          | PASS | PASS (re-observed at step 0 and step 2) |
| TP=2 FIA mixed-KV decode dispatch                         | PASS | PASS (re-observed at steps 3..8) |
| **TP=2 dynamic admission staggered arrival (A then B)**   | UNKNOWN | **PASS** (`admission_events == {"0": 0, "1": 2}`) |
| **TP=2 batch timeline grow B: 1 → 2**                     | UNKNOWN | **PASS** (step 3 batch_size 1 → 2) |
| **TP=2 joint decode with two independently-arrived uids** | UNKNOWN | **PASS** (6 joint decode steps) |
| **TP=2 batch timeline shrink B: 2 → 1**                   | UNKNOWN | **PASS** (step 9 batch_size 2 → 1 after A finishes) |
| **TP=2 remaining request continues after sibling leaves** | UNKNOWN | **PASS** (B produces its 8th token at step 9 alone) |
| **TP=2 dynamic-admission allocator invariant**            | UNKNOWN | **PASS** (952880 → 952880 on both ranks) |
| **TP=2 dynamic-admission cross-rank per-uid determinism** | UNKNOWN | **PASS** (byte-identical `output_texts` and snapshot trace) |
| TP=2 dynamic admission with B: 1 → 2 → 3 → 2 → 1          | UNKNOWN | UNKNOWN (out of scope) |
| TP=2 ragged with non-zero cached_len + extend_len > 1     | UNKNOWN | UNKNOWN (Gate 2.2f `NotImplementedError`, not exercised) |
| TP=2 B > 2                                                | UNKNOWN | UNKNOWN (out of scope) |
| TP=2 timing benchmark                                     | UNKNOWN | UNKNOWN (out of scope) |
| TP > 2                                                    | UNKNOWN | UNKNOWN (out of scope) |
| Non-Qwen3 architecture families under TP=2                | UNKNOWN | UNKNOWN (out of scope) |
| Qwen3-1.7B / 4B / 14B / 32B under TP=2                    | UNKNOWN | UNKNOWN (out of scope) |
| Regression: 8 hermetic suites (per-file)                  | 51 passed | 51 passed (unchanged) |

---

## 12. What is NOT proven at this gate

Explicit exclusions carried forward from the Gate 4.5 opening:

* **B > 2 or B: 1 → 2 → 3 → 2 → 1 under TP=2.** Only B: 1 → 2 → 1
  exercised. Growing to B=3 or shrinking through intermediate B=2 is
  out of scope.
* **Ragged + non-zero cached_len + extend_len > 1 under TP=2.**
  Deliberately not exercised — this is the Gate 2.2f documented FIA
  `NotImplementedError` boundary. Each uid runs one prefill call
  and no radix-cache prefix hit is set up.
* **TP=2 timing benchmark.** No TTFT / e2e / tokens-per-second is
  reported. The wall-clock numbers observed (~2.61–2.66 s for the
  full dynamic-admission trace with metadata capture overhead) are
  not timing measurements.
* **CUDAGraph** under TP=2 — `cuda_graph_bs=[]` is locked at eager
  mode. torch_npu does not implement CUDAGraph.
* **Long-sequence / context-length sweep under TP=2.** Both prompts
  fit in one page each.
* **TP > 2** and multi-node (NNODES > 1).
* **Qwen3-1.7B / Qwen3-4B / 14B / 32B under TP=2**, quantized
  (Qwen3-32B-FP8), MoE (Qwen3-30B-A3B), Qwen3-Next-*, Qwen3-ASR-*,
  Qwen3-Coder-Next. All out of scope.
* **Non-Qwen3 model families under TP=2**.
* **HTTP server under TP=2**, non-stream HTTP cancel, offline
  `LLM.abort()`, chunked prefill. Same boundaries as prior gates.
* **Forward/sampler exception recovery** inside the scheduler under
  TP=2 with dynamic admission. Unchanged from Gate 2.5.
* **Server restart mid-generation.** The driver runs one process
  per rank and does not restart.
* **Long soak / rolling dynamic-admission stress under TP=2.** Only
  one A + B admission cycle is executed.
* **Runtime source refactor of `LLM.offline_receive_msg` or
  `AscendFIABackend`.** The staggered admission and the metadata
  capture are both script-local; the runtime source under
  `python/minisgl/` is untouched. The gate does not attest any new
  invariant on the source files themselves.

Verdict decision matrix (from gate open):

| Outcome | Definition | This gate |
|---|---|---|
| PASS    | TP=2 dynamic admission reaches B: 1→2→1, second request is admitted after first is active, at least one joint decode step occurs, remaining sibling continues, rank outputs match by uid, allocator returns to baseline on both ranks | **✔ (this verdict; 6 joint decode steps, batch_timeline [1,1,1,2,2,2,2,2,2,1], A finishes at step 8, B continues alone at step 9)** |
| PARTIAL | TP=2 generation passes, but true staggered admission or B: 1→2→1 evidence is incomplete | not reached |
| BLOCKED | TP=2 launch or model load no longer works, or current API cannot express dynamic admission without broader runtime changes | not reached |

---

## 13. Freeze boundary

This gate freezes the fact that Mini-SGLang-Ascend at the freeze
commit — descending from `e942b11` with only the new bring-up
script and this verdict document added — completes a TP=2
Qwen3-0.6B dynamic-admission run on 2× Ascend 910B1 under the
frozen eager `npu_fia` bf16 greedy `use_pynccl=False`
`MINISGL_DISTRIBUTED_ADDR=env://` envelope, with:

* `request_uids == [0, 1]` and `prompt_token_lengths == [2, 12]`
  on both ranks
* `batch_timeline == [1, 1, 1, 2, 2, 2, 2, 2, 2, 1]` on both ranks
  — the required `B: 1 → 2 → 1` subsequence present
* `admission_events == {"0": 0, "1": 2}` on both ranks — B
  admitted strictly after A
* `completion_events == {"0": 8, "1": 9}` on both ranks — A
  finishes first, B runs one more solo step
* `joint_decode_step_count == 6` on both ranks — six consecutive
  joint decode steps with `query_lengths == [1, 1]`
* `actual_output_tokens_per_request == [8, 8]` on both ranks
* `output_texts` bit-identical cross-rank per uid
* `available_tokens` returning to baseline (952880) on both ranks
* `free_pages` drift of 1 page (retained as evictable radix-cache
  entry, counted back into `available_tokens`)
* `deferred_abort_uids == 0` on both ranks
* `cache_integrity_ok == true` on both ranks
* 8-file regression 51/51 passing
* Zero source-code changes under `python/minisgl/`

It does not claim TP > 2 support.
It does not claim TP=2 B > 2 support.
It does not claim dynamic admission with B: 1 → 2 → 3 → 2 → 1.
It does not claim TP=2 ragged with non-zero cached_len + extend_len > 1.
It does not claim TP=2 timing / throughput / latency parity.
It does not claim TP=2 for any other model.
It does not modify any prior gate verdict, the release tag
`v0.1.0a1`, or the GitHub Release.
It adds no code under `python/minisgl/` or `tests/`; the only new
files are `scripts/gate4_5_tp2_dynamic_admission_b1_b2_b1.py` and
this verdict.
