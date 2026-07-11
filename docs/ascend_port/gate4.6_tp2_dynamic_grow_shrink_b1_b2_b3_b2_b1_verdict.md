# Gate 4.6 Verdict — TP=2 dynamic grow-shrink B: 1 → 2 → 3 → 2 → 1 (Qwen3-0.6B)

**Gate ID:** 4.6 (TP=2 Ascend dynamic grow-shrink B: 1 → 2 → 3 → 2 → 1 on Qwen3-0.6B)
**Verdict:** PASS
**Branch:** `gate4.6-tp2-dynamic-grow-shrink-b1-b2-b3-b2-b1`
**Base commit:** `127d537` (tip of `ascend-port`, Gate 4.5 merge)
**Freeze commit:** `1bc60ce`
**Date:** 2026-07-11
**Kind:** Real-hardware Ascend 910B1 TP=2 dynamic grow-shrink proof —
two ranks × Qwen3-0.6B × request A starts alone, request B arrives
after A's first decode step, request C arrives while A and B are
jointly decoding, all three jointly decode for three steps,
A finishes first, B and C decode jointly for three more steps,
B finishes second, C decodes alone for three steps. Captured
per-forward-step `FIAMetadata` snapshots show a `batch_timeline` of
`[1, 1, 1, 2, 1, 3, 3, 3, 2, 2, 2, 1, 1, 1]` on both ranks —
proving the required `B: 1 → 2 → 3 → 2 → 1` ordered-subsequence
invariant end-to-end with per-rank allocator invariants held and
cross-rank per-uid output equality.

This verdict does not modify Gate 1 / 2.1 / 2.2 / 2.3 / 2.4 / 2.5 /
3.1 / 3.2 / 3.3 / 3.4 / 4.1 / 4.2 / 4.3 / 4.4 / 4.5, does not mutate
release tag `v0.1.0a1`, does not touch the GitHub Release, CHANGELOG,
or release notes, and does not extend the Ascend port to TP > 2,
TP=2 B > 3, TP=2 timing benchmark, non-Qwen3 architectures, or
Qwen3-1.7B / Qwen3-4B / 14B / 32B / quantized / MoE variants. The
result produced (`actual_output_tokens_per_request == [6, 8, 10]`,
`batch_timeline == [1, 1, 1, 2, 1, 3, 3, 3, 2, 2, 2, 1, 1, 1]`) is
a **correctness proof** for the dynamic grow-shrink shape; no
throughput / latency / tokens-per-second claim is made. The only
new artefacts at this gate are one bring-up script and this
verdict; no runtime source under `python/minisgl/` is modified. The
staggered-admission driver uses a script-local subclass override on
`LLM.offline_receive_msg` (with two staged slots and two admission
flags) plus a script-local monkey-patch on
`AscendFIABackend.prepare_metadata` — both live in the driver
process and do not touch the checked-in package.

---

## 1. Verdict summary

**PASS on all three cases across both ranks.**

| Case | Description | rank 0 | rank 1 |
|---|---|---|---|
| A | init-only smoke — `_init_communication` returns, both ranks report `init_status=PASS` | **PASS** | **PASS** |
| B | model-load smoke — per-rank weight sharding + post-load `check_integrity()` | **PASS** | **PASS** |
| C | dynamic grow-shrink timeline — A starts alone, B admitted while A active, C admitted while A+B active, ≥1 joint B=3 step, one finishes → B=2, another finishes → B=1 | **PASS** | **PASS** |

Cases A / B still collapse into the same `LLM` boot (`set_tp_info` is
a one-shot); reaching a successful `_snapshot(cache_manager)` after
`LLM.__init__` returns proves init + load simultaneously.

Timeline invariants proven on both ranks:

* `request_uids == [0, 1, 2]` (A=uid 0, B=uid 1, C=uid 2 — admission order)
* `prompt_token_lengths == [2, 5, 12]` — deliberately different so
  the three requests occupy different KV budgets and can be
  distinguished per step
* `max_new_tokens_per_request == [6, 8, 10]` — deliberately
  increasing so the completion order is A → B → C, giving an
  unambiguous shrink half of the timeline
* `admission_events == {"0": 0, "1": 2, "2": 4}` — strictly
  increasing: A first appears at step 0, B at step 2, C at step 4
* `completion_events == {"0": 7, "1": 10, "2": 13}` — strictly
  increasing: A leaves at step 7 (after producing 6 tokens),
  B leaves at step 10 (after producing 8 tokens), C leaves at step
  13 (after producing 10 tokens)
* `batch_timeline == [1, 1, 1, 2, 1, 3, 3, 3, 2, 2, 2, 1, 1, 1]` —
  the required `B: 1 → 2 → 3 → 2 → 1` ordered subsequence is
  present: `1` at step 0, `2` at step 3, `3` at step 5, `2` at
  step 8, `1` at step 11
* `joint_decode_step_count_b2 == 4` — four decode steps have
  `batch_size == 2` with `query_lengths == [1, 1]` (one with {A, B}
  at step 3, three with {B, C} at steps 8..10)
* `triple_decode_step_count == 3` — three decode steps have
  `batch_size == 3` with `query_lengths == [1, 1, 1]` at steps
  5, 6, 7 — the primary Gate 4.6 evidence

Post-case allocator invariants held on both ranks:

* `available_tokens_after_case == baseline_available_tokens` (952880 on both ranks)
* `deferred_abort_uids == 0`
* `cache_integrity_ok == true`

Structured logs (both ranks, JSON pretty-printed excerpt for rank 0;
rank 1 emits the byte-identical trace with `device: "npu:1"`):

```
GATE4.6_JSONL rank=0 {
  "rank": 0, "world_size": 2, "tp_size": 2,
  "device": "npu:0",
  "request_uids": [0, 1, 2],
  "prompt_token_lengths": [2, 5, 12],
  "max_new_tokens_per_request": [6, 8, 10],
  "baseline_available_tokens": 952880, "baseline_free_pages": 59555, "total_pages": 59555,
  "init_status": "PASS", "load_status": "PASS",
  "prefill_status": "PASS", "decode_status": "PASS",
  "admission_status": "PASS", "timeline_status": "PASS",
  "batch_timeline": [1, 1, 1, 2, 1, 3, 3, 3, 2, 2, 2, 1, 1, 1],
  "admission_events": {"0": 0, "1": 2, "2": 4},
  "completion_events": {"0": 7, "1": 10, "2": 13},
  "joint_decode_step_count_b2": 4,
  "triple_decode_step_count": 3,
  "all_step_snapshots": [
    {"step_id":  0, "batch_size": 1, "active_uids": [0],       "query_lengths": [2],       "kv_lengths": [2]},
    {"step_id":  1, "batch_size": 1, "active_uids": [0],       "query_lengths": [1],       "kv_lengths": [3]},
    {"step_id":  2, "batch_size": 1, "active_uids": [1],       "query_lengths": [5],       "kv_lengths": [5]},
    {"step_id":  3, "batch_size": 2, "active_uids": [0, 1],    "query_lengths": [1, 1],    "kv_lengths": [4, 6]},
    {"step_id":  4, "batch_size": 1, "active_uids": [2],       "query_lengths": [12],      "kv_lengths": [12]},
    {"step_id":  5, "batch_size": 3, "active_uids": [0, 1, 2], "query_lengths": [1, 1, 1], "kv_lengths": [5, 7, 13]},
    {"step_id":  6, "batch_size": 3, "active_uids": [0, 1, 2], "query_lengths": [1, 1, 1], "kv_lengths": [6, 8, 14]},
    {"step_id":  7, "batch_size": 3, "active_uids": [0, 1, 2], "query_lengths": [1, 1, 1], "kv_lengths": [7, 9, 15]},
    {"step_id":  8, "batch_size": 2, "active_uids": [1, 2],    "query_lengths": [1, 1],    "kv_lengths": [10, 16]},
    {"step_id":  9, "batch_size": 2, "active_uids": [1, 2],    "query_lengths": [1, 1],    "kv_lengths": [11, 17]},
    {"step_id": 10, "batch_size": 2, "active_uids": [1, 2],    "query_lengths": [1, 1],    "kv_lengths": [12, 18]},
    {"step_id": 11, "batch_size": 1, "active_uids": [2],       "query_lengths": [1],       "kv_lengths": [19]},
    {"step_id": 12, "batch_size": 1, "active_uids": [2],       "query_lengths": [1],       "kv_lengths": [20]},
    {"step_id": 13, "batch_size": 1, "active_uids": [2],       "query_lengths": [1],       "kv_lengths": [21]}
  ],
  "actual_output_tokens_per_request": [6, 8, 10],
  "output_texts": [" a city in the southern part", " Paris. The capital of Italy is Rome", "...? A. Mercury B. Venus C."],
  "available_tokens_after_case": 952880, "free_pages_after_case": 59554,
  "free_pages_before_after": [59555, 59554],
  "deferred_abort_uids": 0, "cache_integrity_ok": true,
  "status": "PASS", "failure_stage": null
}
```

Rank 0 and rank 1 produce byte-identical snapshot traces, identical
`admission_events` and `completion_events`, identical
`joint_decode_step_count_b2` and `triple_decode_step_count`,
identical `output_texts`, identical allocator numbers. Cross-rank
symmetry across every field is itself a signal that the TP=2
scheduler / paged-cache / dynamic-admission path is not diverging
between ranks under the more demanding three-request shape.

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
Batching:          B: 1 → 2 → 3 → 2 → 1 dynamic
                   (A alone; A+B; A+B+C; B+C; C alone)
Requests:          A: max_tokens=6,  prompt="Paris is"                   (2 tokens)
                   B: max_tokens=8,  prompt="The capital of France is"   (5 tokens)
                   C: max_tokens=10, prompt="The largest planet in our
                                             solar system by mass and
                                             volume is"                  (12 tokens)
Model:             /mnt/nvme/models/Qwen3-0.6B (dense bf16 Qwen3ForCausalLM)
Launcher:          torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0
                            --master_addr=127.0.0.1 --master_port=29407
Distributed init:  MINISGL_DISTRIBUTED_ADDR=env:// (reuses torchrun store)
                   backend=hccl (primary) + gloo (sidecar via new_group)
                   use_pynccl=False (mandatory on NPU)
Driver:            scripts/gate4_6_tp2_dynamic_grow_shrink_b1_b2_b3_b2_b1.py
                   per-rank worker under torchrun; both ranks
                   independently instantiate a script-local
                   ``GrowShrinkLLM(LLM)`` subclass whose overridden
                   ``offline_receive_msg`` (a) reveals request B only
                   after the metadata hook observes ≥1 decode step
                   with A alone, and (b) reveals request C only after
                   the metadata hook observes ≥1 joint decode step
                   containing exactly {A, B}. Both ranks run
                   identical forwards in HCCL lockstep, so both
                   ranks flip both admission flags on the same tick.
Metadata capture:  Script-only monkey-patch. Wraps
                   AscendFIABackend.prepare_metadata for the driver
                   process only; the runtime source under
                   python/minisgl/attention/ascend_fia.py is not
                   modified. The same hook records the per-step
                   FIAMetadata AND drives BOTH admission flags.
```

---

## 3. Launch command

Executed on remote container `<CONTAINER>` at working directory
`/mnt/nvme/LR-606/mini-sglang-ascend-gate46`.

```bash
PYTHONPATH=./python torchrun \
  --nproc_per_node=2 --nnodes=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port=29407 \
  scripts/gate4_6_tp2_dynamic_grow_shrink_b1_b2_b3_b2_b1.py
```

The script's structured stdout is the primary evidence; each rank
emits exactly one `GATE4.6_JSONL rank=<r> {...}` line containing the
full per-forward-step snapshot list, admission events, and
completion events.

---

## 4. Prompt / uid mapping evidence

| Request | uid | prompt | tokens | max_tokens |
|---|---|---|---|---|
| A | 0 | `"Paris is"` | 2 | 6 |
| B | 1 | `"The capital of France is"` | 5 | 8 |
| C | 2 | `"The largest planet in our solar system by mass and volume is"` | 12 | 10 |

Both ranks tokenize the same prompts (same tokenizer, same
`self.counter` incrementing) and both assign uid 0 to A, uid 1 to B,
uid 2 to C. The script asserts `request_uids == [0, 1, 2]` before
consuming the snapshot list.

The staggered-admission mechanism is entirely offline: `LLM` runs in
`offline_mode=True`, so `receive_msg` is bound to
`offline_receive_msg` by `SchedulerIOMixin.__init__`. The subclass
overrides `offline_receive_msg` — same MRO binding, so the wrapper
gets called each scheduler tick. On each tick the wrapper checks
both `state.ready_to_admit_b` and `state.ready_to_admit_c`; when
either flag is set (by the metadata hook, which is the only
mutator) and the corresponding staged request has not yet been
revealed, the wrapper appends `(prompt, sp)` to
`self.pending_requests` and delegates to `super().offline_receive_msg`.
The next tick falls through the parent's normal loop; the newly
revealed request enters the scheduler as a `UserMsg` and lands in
`prefill_manager`.

Because `ready_to_admit_c` requires an observed batch-size-2 joint
decode step, C cannot be revealed until B has actually been admitted
AND B has decoded jointly with A at least once. This guarantees that
when C is admitted, both A and B are still active — the required
precondition for the B=3 joint decode step.

---

## 5. Batch timeline evidence (the primary Gate 4.6 evidence)

Captured per-forward-step snapshots (both ranks agree byte-for-byte):

| step_id | batch_size | active_uids  | query_lengths | kv_lengths     | interpretation |
|---:|---:|---|---|---|---|
|  0 | 1 | `[0]`       | `[2]`       | `[2]`         | prefill A alone (cached_len==0 branch) |
|  1 | 1 | `[0]`       | `[1]`       | `[3]`         | decode A alone — **hook flips `ready_to_admit_b`** here |
|  2 | 1 | `[1]`       | `[5]`       | `[5]`         | prefill B alone (A in decode_manager but prefill wins scheduling) |
|  3 | 2 | `[0, 1]`    | `[1, 1]`    | `[4, 6]`      | joint decode A+B — **hook flips `ready_to_admit_c`** here |
|  4 | 1 | `[2]`       | `[12]`      | `[12]`        | prefill C alone (A and B in decode_manager but prefill wins scheduling) |
|  5 | 3 | `[0, 1, 2]` | `[1, 1, 1]` | `[5, 7, 13]`  | first triple joint decode (A+B+C) — Gate 4.6 primary evidence |
|  6 | 3 | `[0, 1, 2]` | `[1, 1, 1]` | `[6, 8, 14]`  | second triple joint decode |
|  7 | 3 | `[0, 1, 2]` | `[1, 1, 1]` | `[7, 9, 15]`  | third triple joint decode; **A finishes after this step** (6 total outputs) |
|  8 | 2 | `[1, 2]`    | `[1, 1]`    | `[10, 16]`    | shrink to B=2: A gone, B+C jointly decode |
|  9 | 2 | `[1, 2]`    | `[1, 1]`    | `[11, 17]`    | B+C joint decode |
| 10 | 2 | `[1, 2]`    | `[1, 1]`    | `[12, 18]`    | last B+C joint decode; **B finishes after this step** (8 total outputs) |
| 11 | 1 | `[2]`       | `[1]`       | `[19]`        | shrink to B=1: only C left |
| 12 | 1 | `[2]`       | `[1]`       | `[20]`        | C decodes alone |
| 13 | 1 | `[2]`       | `[1]`       | `[21]`        | last C solo decode; **C finishes after this step** (10 total outputs) |

Derived counts (both ranks agree):

* `batch_timeline` = `[1, 1, 1, 2, 1, 3, 3, 3, 2, 2, 2, 1, 1, 1]` —
  length 14 (2 prefills + decode of A alone, 1 prefill of B, 1
  joint A+B decode, 1 prefill of C, 3 triple joint decodes, 3 B+C
  joint decodes, 3 solo C decodes)
* Contains `B: 1 → 2 → 3 → 2 → 1` subsequence in order:
  index 0 (bs=1) → index 3 (bs=2) → index 5 (bs=3) → index 8 (bs=2)
  → index 11 (bs=1). The strict five-state machine
  `1 → 2 → 3 → 2 → 1` advances to completion on this trace
  → **timeline_status = PASS**
* `admission_events` = `{"0": 0, "1": 2, "2": 4}` — A's first
  batch entry is at step 0, B's at step 2, C's at step 4; strictly
  increasing admission order → **admission_status = PASS**
* `completion_events` = `{"0": 7, "1": 10, "2": 13}` — A leaves
  first at step 7, B at step 10, C at step 13; strictly increasing
  completion order matches the intended `max_new_tokens_per_request`
  ordering (6 < 8 < 10)
* `joint_decode_step_count_b2` = 4 — four decode steps with
  `batch_size == 2` and `query_lengths == [1, 1]` (step 3 with
  {A, B}; steps 8, 9, 10 with {B, C})
* `triple_decode_step_count` = 3 — three decode steps with
  `batch_size == 3` and `query_lengths == [1, 1, 1]` at steps 5,
  6, 7 — satisfying the Gate 4.6 requirement of ≥1 joint B=3 step

KV-length arithmetic on the triple decode window:

* At step 5 (first B=3 decode): kv_lengths = `[5, 7, 13]`. A entered
  the triple with kv=4 (post its prefill of 2 tok + 1 joint decode
  with B at step 3, kv delta +1 = 5 — wait, at step 3 A's kv was 4
  because A's prefill produced kv=2, decode at step 1 produced kv=3,
  the joint decode at step 3 used kv=4 (post step-3 update: 5)).
  Concretely: post-prefill of C (kv=12), C's first decode at step 5
  ends at kv=13 (per-uid kv+=1). The three uids at step 5 report kv
  = 5, 7, 13 — each is exactly `prompt_len_i + steps_decoded_i`,
  confirming per-uid `device_len` bookkeeping stays correct across
  the grow transitions.
* Each subsequent B=3 step increments every uid's kv by exactly 1
  (5→6→7 for A, 7→8→9 for B, 13→14→15 for C). This is the mixed-KV
  invariant now attested at B=3 rather than just at B=2.

Rank 0 and rank 1 produce byte-identical snapshot lists — the two
ranks see the same scheduler shape and pass identical
`(query_seq_lens, kv_seq_lens, active_uids)` to their FIA operator
on every step, on all 14 forward passes. Because the scheduler is
process-local (each rank runs its own copy), this cross-rank
agreement is a real signal that the paged-cache / extend-len /
device-len bookkeeping is deterministic and symmetric across ranks
under a batch that grows through B=3 and shrinks back.

---

## 6. Admission / completion event evidence

| Event | uid | step_id | forward-pass semantics |
|---|---|---|---|
| A admitted   | 0 | 0  | request A first appears in a batch (prefill) |
| B admitted   | 1 | 2  | request B first appears in a batch (prefill), 2 steps after A |
| C admitted   | 2 | 4  | request C first appears in a batch (prefill), 2 steps after B |
| A completed  | 0 | 7  | request A's last batch entry; produces its 6th output token |
| B completed  | 1 | 10 | request B's last batch entry; produces its 8th output token |
| C completed  | 2 | 13 | request C's last batch entry; produces its 10th output token |

Ordering invariants:

* `admission_events["0"] < admission_events["1"] < admission_events["2"]`
  → `0 < 2 < 4` ✓ — strict admission order preserved on both ranks
  (uids assigned by `self.counter`).
* `completion_events["0"] < completion_events["1"] < completion_events["2"]`
  → `7 < 10 < 13` ✓ — completion order matches the intended
  `max_new_tokens_per_request` gradient (A=6 finishes first,
  B=8 second, C=10 last). No collisions or tie-breaks needed.
* `admission_events["2"] < completion_events["0"]` → `4 < 7` ✓
  — C was admitted while A was still active, satisfying the
  precondition for the B=3 joint decode step.
* `admission_events["2"] < completion_events["1"]` → `4 < 10` ✓
  — C was admitted while B was still active.
* `admission_events["1"] < completion_events["0"]` → `2 < 7` ✓
  — B was admitted while A was still active (from Gate 4.5).

Wall-clock informational (not a timing claim): `run_forever()`
drained in 2903.85 ms on rank 0 and 2937.35 ms on rank 1 for the
full grow-shrink trace with metadata capture. Not a benchmark; the
metadata-snapshot hook adds unbudgeted per-step overhead by design.

---

## 7. Per-rank output evidence

| Field | rank 0 | rank 1 |
|---|---|---|
| `device` | `npu:0` | `npu:1` |
| `request_uids` | `[0, 1, 2]` | `[0, 1, 2]` |
| `prompt_token_lengths` | `[2, 5, 12]` | `[2, 5, 12]` |
| `max_new_tokens_per_request` | `[6, 8, 10]` | `[6, 8, 10]` |
| `init_status` | `PASS` | `PASS` |
| `load_status` | `PASS` | `PASS` |
| `prefill_status` | `PASS` | `PASS` |
| `decode_status` | `PASS` | `PASS` |
| `admission_status` | `PASS` | `PASS` |
| `timeline_status` | `PASS` | `PASS` |
| `actual_output_tokens_per_request` | `[6, 8, 10]` | `[6, 8, 10]` |
| `output_texts[0]` (uid 0 = A) | `" a city in the southern part"` | `" a city in the southern part"` |
| `output_texts[1]` (uid 1 = B) | `" Paris. The capital of Italy is Rome"` | `" Paris. The capital of Italy is Rome"` |
| `output_texts[2]` (uid 2 = C) | `"...? A. Mercury B. Venus C."` | `"...? A. Mercury B. Venus C."` |
| `status` | `PASS` | `PASS` |
| `failure_stage` | `null` | `null` |

Cross-rank per-uid output equality is exact (byte-identical string)
on all three uids. Notes on the outputs vs prior gates:

* A's output (`" a city in the southern part"` at 6 tokens) is the
  6-token prefix of A's Gate 4.5 output (which was 8 tokens:
  `" a city in the southern part of France"`) — expected because
  greedy on gathered logits produces the same token sequence
  regardless of when the request is truncated by `max_tokens`.
* C's output (`"...? A. Mercury B. Venus C."` at 10 tokens) is
  2 tokens longer than the same prompt's Gate 4.5 result
  (`"...? A. Mercury B. Venus"` at 8 tokens). Same greedy
  trajectory, just longer.
* B's output (`" Paris. The capital of Italy is Rome"` at 8 tokens)
  is new to this gate — B's prompt (`"The capital of France is"`)
  is not the Gate 4.5 B prompt, and gate 4.6 exercises three
  distinct prompts to keep the KV lengths clearly separable in the
  snapshot trace.

Per-uid greedy determinism confirms:

* Dynamic admission of B and later C mid-generation did not corrupt
  A's KV pages or perturb A's decode trajectory — A produces the
  same tokens it would have produced in a solo B=1 run for its
  first six tokens.
* Late arrival + mid-batch prefill of C did not perturb A's or B's
  decode trajectories — both continue producing the same tokens
  they would have produced without C.
* All-gathered `ParallelLMHead` continues to reassemble a
  fully-replicated logits tensor identically on both ranks under
  a three-request dynamic-admission batch.
* Per-uid `device_len` bookkeeping stays correct across every
  grow and shrink transition on both ranks — including the two
  grow steps (B=1→2 at step 3 and B=2→3 at step 5) and the two
  shrink steps (B=3→2 at step 8 and B=2→1 at step 11).

---

## 8. Per-rank allocator evidence

| rank | baseline_available_tokens | after case | baseline_free_pages | after case | free_pages_before_after | total_pages | deferred_abort_uids | cache_integrity_ok |
|---|---|---|---|---|---|---|---|---|
| 0 | 952880 | 952880 | 59555 | 59554 | `[59555, 59554]` | 59555 | 0 | true |
| 1 | 952880 | 952880 | 59555 | 59554 | `[59555, 59554]` | 59555 | 0 | true |

Interpretation:

* Baseline `available_tokens = 952880` per rank — identical to
  Gates 4.1, 4.2, 4.3, 4.4, 4.5 baselines under the same TP=2
  envelope.
* `available_tokens` returned exactly to baseline on both ranks
  after the grow-shrink batch completed — the primary allocator
  invariant. The two grow transitions (B=1→2 and B=2→3) and the
  two shrink transitions (B=3→2 and B=2→1) each proceeded without
  leaking pages. All three uids' prefill + decode pages were fully
  released once each request hit its respective `max_tokens`
  (6 / 8 / 10).
* `free_pages` drift of 1 (59555 → 59554) on both ranks — one
  evictable radix-cache prefix page retained after the batch,
  matching the Gate 4.3 / 4.4 / 4.5 pattern under the same short
  prompt "Paris is". Per Gate 3.1 §4,
  `available_size = free_slots + evictable_prefix_pages` (in
  tokens), so 1 evictable page ≈ 16 tokens are counted back into
  `available_tokens`, keeping the token-level invariant exact
  while raw `free_pages` is off by 1.
* `deferred_abort_uids == 0` — no abort path was entered.
* `cache_integrity_ok == true` — allocator invariants held
  throughout, including through the two mid-batch prefills (B at
  step 2 and C at step 4), the three triple joint decode steps,
  and both shrink transitions.

Both ranks show the exact same drift pattern (identical baseline,
identical after-case, identical evictable-page retention). This
symmetry across ranks under a three-request dynamic-admission
batch is itself a signal that the TP=2 allocator does not diverge
between ranks even when requests arrive on different scheduler
ticks and complete in different orders.

---

## 9. First failing stage

**None.** No failure surfaced on the smoke run. Cases A / B / C all
reported `PASS` on both ranks. `failure_stage` and
`failure_trace_summary` are `null` on both ranks.

No source-file change under `python/minisgl/` was required at this
gate. The Gate 4.1 minimum fixes (`LLM.__init__` `tp_info` kwarg,
`EngineConfig.distributed_addr` env-var override) and Gate 4.5's
established staggered-admission pattern already inherited from
`ascend-port` at `127d537` provided the full driver + launcher +
subclass-override path. Extension to three requests is entirely
expressed at the driver layer via two staged slots (`_staged_b`,
`_staged_c`) and two shared-state flags (`ready_to_admit_b`,
`ready_to_admit_c`); no new runtime concepts were needed.

The metadata-snapshot hook remains a script-only monkey-patch on
the imported `AscendFIABackend.prepare_metadata` classmethod. It
runs after the original method sets `batch.attn_metadata` and
never mutates the metadata; the underlying runtime source is not
modified. The same hook fires both admission flags.

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
3.3, Gate 3.4, Gate 4.1, Gate 4.2, Gate 4.3, Gate 4.4, Gate 4.5.

Zero source-code changes under `python/minisgl/` at this gate — the
regression rows are expected to be unchanged by construction; the
run confirms it.

Note on batched invocation: running all eight files in a single
`pytest` command triggers a pre-existing pytest cross-file leakage
that surfaces as an `AttributeError: 'Message' object has no
attribute 'model_dump'` inside `api_server.py` when
`test_shell_cancel_cleanup.py` runs after certain scheduler test
files. This behaviour is identical on Gate 4.5's tree (verified by
re-running the same batch on `mini-sglang-ascend-gate45`) and is
therefore pre-existing to Gate 4.6. Per-file invocation — the mode
locked by every prior gate — is hermetic and unaffected: 51/51
passing on both trees.

---

## 11. Support matrix delta (Gate 4.5 → Gate 4.6)

| Capability                                                | Gate 4.5 | Gate 4.6 |
|---|---|---|
| TP=2 driver support in offline `LLM`                      | PASS | PASS (unchanged) |
| TP=2 launcher rendezvous under torchrun (`env://`)        | PASS | PASS (unchanged) |
| TP=2 Qwen3-0.6B B=2 ragged prefill (cached_len==0) N=8    | PASS | not re-attested |
| TP=2 Qwen3-0.6B B=2 mixed-KV decode explicit evidence     | PASS | not re-attested |
| TP=2 FIA ragged-prefill dispatch                          | PASS | PASS (re-observed at steps 0, 2, 4) |
| TP=2 FIA mixed-KV decode dispatch                         | PASS | PASS (re-observed at steps 3, 5-10) |
| TP=2 dynamic admission staggered arrival (A then B)       | PASS | PASS (re-observed: `admission_events["0"]=0 < ["1"]=2`) |
| TP=2 batch timeline grow B: 1 → 2                         | PASS | PASS (re-observed at step 3) |
| TP=2 joint decode with two independently-arrived uids     | PASS | PASS (re-observed: 4 B=2 joint decode steps) |
| TP=2 batch timeline shrink B: 2 → 1                       | PASS | PASS (re-observed at step 11) |
| TP=2 remaining request continues after sibling leaves     | PASS | PASS (C decodes solo at steps 11, 12, 13) |
| TP=2 dynamic-admission allocator invariant                | PASS | PASS (952880 → 952880 on both ranks) |
| TP=2 dynamic-admission cross-rank per-uid determinism     | PASS | PASS (byte-identical `output_texts` and snapshot trace) |
| **TP=2 dynamic admission of a third request (A+B → A+B+C)** | UNKNOWN | **PASS** (`admission_events["2"]=4 < completion_events["0"]=7`) |
| **TP=2 batch timeline grow B: 2 → 3**                     | UNKNOWN | **PASS** (step 5 batch_size 2 → 3) |
| **TP=2 joint decode with three independently-arrived uids** | UNKNOWN | **PASS** (3 B=3 joint decode steps at 5, 6, 7) |
| **TP=2 batch timeline shrink B: 3 → 2**                   | UNKNOWN | **PASS** (step 8 batch_size 3 → 2 after A finishes) |
| **TP=2 two consecutive shrink transitions (3→2→1)**       | UNKNOWN | **PASS** (steps 8 and 11) |
| **TP=2 batch_timeline contains ordered `[1,2,3,2,1]`**    | UNKNOWN | **PASS** (`[1, 1, 1, 2, 1, 3, 3, 3, 2, 2, 2, 1, 1, 1]`) |
| **TP=2 three-way per-uid `device_len` bookkeeping**       | UNKNOWN | **PASS** (kv_lengths delta = +1 per uid per step across grow/shrink) |
| **TP=2 three-way cross-rank per-uid determinism**         | UNKNOWN | **PASS** (rank 0 and rank 1 byte-identical on all three uids) |
| **TP=2 three-way allocator invariant**                    | UNKNOWN | **PASS** (952880 → 952880 on both ranks; 1-page radix drift) |
| **TP=2 different max_new_tokens per uid**                 | UNKNOWN | **PASS** (A=6, B=8, C=10; each request stops at exactly its `max_tokens`) |
| TP=2 ragged with non-zero cached_len + extend_len > 1     | UNKNOWN | UNKNOWN (Gate 2.2f `NotImplementedError`, not exercised) |
| TP=2 B > 3                                                | UNKNOWN | UNKNOWN (out of scope) |
| TP=2 timing benchmark                                     | UNKNOWN | UNKNOWN (out of scope) |
| TP > 2                                                    | UNKNOWN | UNKNOWN (out of scope) |
| Non-Qwen3 architecture families under TP=2                | UNKNOWN | UNKNOWN (out of scope) |
| Qwen3-1.7B / 4B / 14B / 32B under TP=2                    | UNKNOWN | UNKNOWN (out of scope) |
| Regression: 8 hermetic suites (per-file)                  | 51 passed | 51 passed (unchanged) |

---

## 12. What is NOT proven at this gate

Explicit exclusions carried forward from the Gate 4.6 opening:

* **B > 3 under TP=2.** Only B: 1 → 2 → 3 → 2 → 1 exercised.
  Growing to B=4 or beyond is out of scope.
* **Ragged + non-zero cached_len + extend_len > 1 under TP=2.**
  Deliberately not exercised — this is the Gate 2.2f documented
  FIA `NotImplementedError` boundary. Each uid runs one prefill
  call and no radix-cache prefix hit is set up.
* **TP=2 timing benchmark.** No TTFT / e2e / tokens-per-second is
  reported. The wall-clock numbers observed (~2.90–2.94 s for the
  full grow-shrink trace with metadata capture overhead) are not
  timing measurements.
* **CUDAGraph** under TP=2 — `cuda_graph_bs=[]` is locked at eager
  mode. torch_npu does not implement CUDAGraph.
* **Long-sequence / context-length sweep under TP=2.** All three
  prompts fit in one page each.
* **TP > 2** and multi-node (NNODES > 1).
* **Qwen3-1.7B / Qwen3-4B / 14B / 32B under TP=2**, quantized
  (Qwen3-32B-FP8), MoE (Qwen3-30B-A3B), Qwen3-Next-*, Qwen3-ASR-*,
  Qwen3-Coder-Next. All out of scope.
* **Non-Qwen3 model families under TP=2**.
* **HTTP server under TP=2**, non-stream HTTP cancel, offline
  `LLM.abort()`, chunked prefill. Same boundaries as prior gates.
* **Forward/sampler exception recovery** inside the scheduler under
  TP=2 with three-way dynamic admission. Unchanged from Gate 2.5.
* **Server restart mid-generation.** The driver runs one process
  per rank and does not restart.
* **Long soak / rolling three-way grow-shrink stress under TP=2.**
  Only one A + B + C admission cycle is executed.
* **Runtime source refactor of `LLM.offline_receive_msg` or
  `AscendFIABackend`.** The staggered admission and the metadata
  capture are both script-local; the runtime source under
  `python/minisgl/` is untouched. The gate does not attest any new
  invariant on the source files themselves.
* **Out-of-order completion.** A / B / C were designed to complete
  in admission order (max_tokens 6 < 8 < 10). Behaviour under an
  earlier-arriving request finishing later than a later-arriving
  request is not attested.

Verdict decision matrix (from gate open):

| Outcome | Definition | This gate |
|---|---|---|
| PASS    | TP=2 dynamic grow-shrink reaches B: 1→2→3→2→1, batch_timeline contains the ordered subsequence, ≥1 joint B=3 decode step, second and third requests admitted while prior requests active, ranks match per uid, allocator returns to baseline on both ranks | **✔ (this verdict; 3 triple decode steps, batch_timeline [1,1,1,2,1,3,3,3,2,2,2,1,1,1])** |
| PARTIAL | TP=2 generation passes, but true three-way staggered admission or B: 1→2→3→2→1 evidence is incomplete | not reached |
| BLOCKED | TP=2 launch or model load no longer works, or current API cannot express three-way dynamic admission without broader runtime changes | not reached |

---

## 13. Freeze boundary

This gate freezes the fact that Mini-SGLang-Ascend at the freeze
commit — descending from `127d537` with only the new bring-up
script and this verdict document added — completes a TP=2
Qwen3-0.6B three-request dynamic grow-shrink run on 2× Ascend 910B1
under the frozen eager `npu_fia` bf16 greedy `use_pynccl=False`
`MINISGL_DISTRIBUTED_ADDR=env://` envelope, with:

* `request_uids == [0, 1, 2]` and `prompt_token_lengths == [2, 5, 12]`
  on both ranks
* `max_new_tokens_per_request == [6, 8, 10]` on both ranks
* `batch_timeline == [1, 1, 1, 2, 1, 3, 3, 3, 2, 2, 2, 1, 1, 1]` on
  both ranks — the required `B: 1 → 2 → 3 → 2 → 1` ordered
  subsequence present
* `admission_events == {"0": 0, "1": 2, "2": 4}` on both ranks —
  strict admission order A → B → C
* `completion_events == {"0": 7, "1": 10, "2": 13}` on both ranks —
  strict completion order A → B → C
* `joint_decode_step_count_b2 == 4` and
  `triple_decode_step_count == 3` on both ranks — three consecutive
  triple joint decode steps with `query_lengths == [1, 1, 1]`
* `actual_output_tokens_per_request == [6, 8, 10]` on both ranks
* `output_texts` bit-identical cross-rank per uid on all three uids
* `available_tokens` returning to baseline (952880) on both ranks
* `free_pages` drift of 1 page (retained as evictable radix-cache
  entry, counted back into `available_tokens`)
* `deferred_abort_uids == 0` on both ranks
* `cache_integrity_ok == true` on both ranks
* 8-file per-file regression 51/51 passing
* Zero source-code changes under `python/minisgl/`

It does not claim TP > 2 support.
It does not claim TP=2 B > 3 support.
It does not claim TP=2 ragged with non-zero cached_len + extend_len > 1.
It does not claim TP=2 timing / throughput / latency parity.
It does not claim TP=2 for any other model.
It does not claim out-of-order completion under three-way admission.
It does not modify any prior gate verdict, the release tag
`v0.1.0a1`, or the GitHub Release.
It adds no code under `python/minisgl/` or `tests/`; the only new
files are `scripts/gate4_6_tp2_dynamic_grow_shrink_b1_b2_b3_b2_b1.py`
and this verdict.
