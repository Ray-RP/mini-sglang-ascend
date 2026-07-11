"""Gate 4.6 TP=2 dynamic grow-shrink B: 1 → 2 → 3 → 2 → 1 smoke driver.

Runs a Qwen3-0.6B TP=2 dynamic grow-shrink end-to-end on Ascend 910B1
and captures per-forward-step ``FIAMetadata`` snapshots so the batch
timeline ``[1, ..., 2, ..., 3, ..., 2, ..., 1]`` is directly visible in
the structured log:

Cases:
    A. TP=2 init
    B. TP=2 model load
    C. dynamic grow-shrink timeline:
       * request A starts alone (B=1 prefill + ≥1 decode)
       * request B is admitted while A is still active (grow to B=2)
       * request C is admitted while A and B are still both active
         (grow to B=3)
       * A, B, C jointly decode at least one step (batch_size == 3)
       * one request finishes first; the remaining two continue as
         a joint decode (shrink to B=2)
       * a second request finishes; the last one continues alone
         (shrink back to B=1)
    D. cross-rank per-uid output equality (rank 0's output_texts[uid]
       == rank 1's output_texts[uid]), allocator returns to baseline,
       deferred_abort_uids == 0, cache_integrity_ok == True.

Envelope (locked at Gate 4.6):
    Ascend 910B1, TP=2, eager, npu_fia, bf16, greedy
    Different max_new_tokens per uid (A=6, B=8, C=10) to force a
    clean completion order (A finishes first, then B, then C) so the
    shrink half of the timeline (3 → 2 → 1) is unambiguous.
    memory_ratio=0.85, page_size=16, max_running_req=4
    cuda_graph_bs=[] (torch_npu has no CUDAGraph)

Staggered-admission strategy (script-local, no runtime change):
    ``LLM.offline_receive_msg`` normally reveals every entry in
    ``self.pending_requests`` up to ``prefill_budget`` on each tick.
    To force staggered arrival we subclass ``LLM`` and override
    ``offline_receive_msg`` so it (a) starts with only request A in
    ``pending_requests`` and (b) reveals request B only after the
    metadata hook has observed at least one decode step containing
    ONLY request A, and (c) reveals request C only after the hook has
    observed at least one joint (batch_size == 2) decode step
    containing exactly {A, B}. Both TP ranks make the same admission
    decisions because both ranks run identical forwards in HCCL
    lockstep — same per-step batch shape → same hook flag flips →
    same offline_receive_msg decisions → uids assigned symmetrically
    (A=0, B=1, C=2 on both ranks).

Metadata capture strategy:
    ``AscendFIABackend.prepare_metadata(batch)`` runs exactly once
    per forward pass. The driver monkey-patches the method to record
    per-step ``(batch_size, active_uids, query_seq_lens, kv_seq_lens)``
    into an in-memory ring right after the original method sets
    ``batch.attn_metadata``. The same hook drives BOTH admission
    triggers (``ready_to_admit_b`` and ``ready_to_admit_c``). The
    runtime source under ``python/minisgl/attention/ascend_fia.py``
    is not modified.

Structured log format (per rank):
    A single JSON object on stdout tagged
    ``GATE4.6_JSONL rank=<r> ...`` with the fields listed in Gate 4.6
    spec §4, including:
      * ``batch_timeline`` — the per-step ``batch_size`` sequence
      * ``admission_events`` — the step index where each uid first
        entered a batch
      * ``completion_events`` — the last step index where each uid
        appeared in a batch
      * ``triple_decode_step_count`` — number of steps with
        batch_size == 3 and query_lengths == [1, 1, 1]
      * ``all_step_snapshots`` — the full per-forward-step trace

Exit code (rank 0):
    0 on PASS, 1 on FAIL, 2 on BLOCKED.

Out of scope (per Gate 4.6 spec §5):
    B > 3, TP=4, Qwen3-1.7B TP=2, timing benchmark, chunked prefill,
    exception recovery, long soak, server path.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StepSnapshot:
    step_id: int
    batch_size: int
    active_uids: List[int]
    query_lengths: List[int]
    kv_lengths: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "batch_size": self.batch_size,
            "active_uids": list(self.active_uids),
            "query_lengths": list(self.query_lengths),
            "kv_lengths": list(self.kv_lengths),
        }


@dataclass
class RankLog:
    rank: int
    world_size: int
    tp_size: int
    model_path: str
    device: str = ""
    request_uids: List[int] = field(default_factory=list)
    prompt_token_lengths: List[int] = field(default_factory=list)
    max_new_tokens_per_request: List[int] = field(default_factory=list)
    baseline_available_tokens: Optional[int] = None
    baseline_free_pages: Optional[int] = None
    total_pages: Optional[int] = None
    init_status: str = "PENDING"
    load_status: str = "PENDING"
    prefill_status: str = "PENDING"
    decode_status: str = "PENDING"
    admission_status: str = "PENDING"
    timeline_status: str = "PENDING"
    batch_timeline: Optional[List[int]] = None
    admission_events: Optional[Dict[str, int]] = None
    completion_events: Optional[Dict[str, int]] = None
    joint_decode_step_count_b2: Optional[int] = None
    triple_decode_step_count: Optional[int] = None
    all_step_snapshots: Optional[List[Dict[str, Any]]] = None
    decode_step_snapshots: Optional[List[Dict[str, Any]]] = None
    actual_output_tokens_per_request: Optional[List[int]] = None
    output_texts: Optional[List[str]] = None
    available_tokens_after_case: Optional[int] = None
    free_pages_after_case: Optional[int] = None
    free_pages_before_after: Optional[List[int]] = None
    deferred_abort_uids: Optional[int] = None
    cache_integrity_ok: Optional[bool] = None
    status: str = "BLOCKED"
    failure_stage: Optional[str] = None
    failure_trace_summary: Optional[str] = None

    def emit(self) -> None:
        payload: Dict[str, Any] = {
            "rank": self.rank,
            "world_size": self.world_size,
            "tp_size": self.tp_size,
            "model_path": self.model_path,
            "device": self.device,
            "request_uids": self.request_uids,
            "prompt_token_lengths": self.prompt_token_lengths,
            "max_new_tokens_per_request": self.max_new_tokens_per_request,
            "baseline_available_tokens": self.baseline_available_tokens,
            "baseline_free_pages": self.baseline_free_pages,
            "total_pages": self.total_pages,
            "init_status": self.init_status,
            "load_status": self.load_status,
            "prefill_status": self.prefill_status,
            "decode_status": self.decode_status,
            "admission_status": self.admission_status,
            "timeline_status": self.timeline_status,
            "batch_timeline": self.batch_timeline,
            "admission_events": self.admission_events,
            "completion_events": self.completion_events,
            "joint_decode_step_count_b2": self.joint_decode_step_count_b2,
            "triple_decode_step_count": self.triple_decode_step_count,
            "all_step_snapshots": self.all_step_snapshots,
            "decode_step_snapshots": self.decode_step_snapshots,
            "actual_output_tokens_per_request": self.actual_output_tokens_per_request,
            "output_texts": self.output_texts,
            "available_tokens_after_case": self.available_tokens_after_case,
            "free_pages_after_case": self.free_pages_after_case,
            "free_pages_before_after": self.free_pages_before_after,
            "deferred_abort_uids": self.deferred_abort_uids,
            "cache_integrity_ok": self.cache_integrity_ok,
            "status": self.status,
            "failure_stage": self.failure_stage,
            "failure_trace_summary": self.failure_trace_summary,
        }
        print(
            f"GATE4.6_JSONL rank={self.rank} {json.dumps(payload, ensure_ascii=False)}",
            flush=True,
        )


def _err(rank: int, msg: str) -> None:
    print(f"[gate4.6][rank={rank}] {msg}", flush=True)


def _snapshot(cache_manager) -> Dict[str, Any]:
    """Read allocator invariants without mutating state."""
    total_pages = cache_manager.num_pages
    free_pages = len(cache_manager.free_slots)
    try:
        available_tokens = cache_manager.available_size
    except Exception:
        available_tokens = None
    try:
        cache_manager.check_integrity()
        integrity_ok = True
    except Exception as exc:
        integrity_ok = False
        _err(-1, f"cache integrity check raised: {exc!r}")
    return {
        "total_pages": total_pages,
        "free_pages": free_pages,
        "available_tokens": available_tokens,
        "integrity_ok": integrity_ok,
    }


# ---------------------------------------------------------------------
# Shared mutable state between the metadata hook and the LLM subclass.
# ---------------------------------------------------------------------
@dataclass
class SharedState:
    snapshots: List[StepSnapshot] = field(default_factory=list)
    # Flag: set by the hook the first time it sees a decode step
    # containing only uid A alone. The next tick's offline_receive_msg
    # will read it and admit uid B.
    ready_to_admit_b: bool = False
    # Flag: set by the hook the first time it sees a joint decode step
    # (batch_size == 2, query_lengths == [1, 1]) containing exactly
    # {A, B}. The next tick's offline_receive_msg will read it and
    # admit uid C. Gating on the joint-decode step (not merely on B
    # being admitted) guarantees that C is admitted while BOTH A and
    # B are still active, giving Gate 4.6 the required B=3 joint
    # decode step.
    ready_to_admit_c: bool = False
    a_uid: int = 0
    b_uid: int = 1
    c_uid: int = 2


def _install_metadata_snapshot_hook(state: SharedState) -> None:
    """Wrap ``AscendFIABackend.prepare_metadata`` to record per-step shape.

    Script-only monkey-patch — the runtime source under
    ``python/minisgl/attention/ascend_fia.py`` is not modified. The
    wrapper defers to the original method (which sets
    ``batch.attn_metadata = FIAMetadata(...)`` at the end of its work),
    then records the freshly-assigned metadata alongside the active
    uids read directly from ``batch.reqs``. It also flips the two
    admission trigger flags:

      * ``ready_to_admit_b`` on the first pure-decode step
        (query=[1]) that contains ONLY uid A;
      * ``ready_to_admit_c`` on the first joint decode step
        (batch_size == 2, query_lengths == [1, 1]) whose active_uids
        are exactly {A, B}.
    """
    from minisgl.attention.ascend_fia import AscendFIABackend, FIAMetadata

    original = AscendFIABackend.prepare_metadata

    def patched(self, batch):  # type: ignore[no-redef]
        original(self, batch)
        m = batch.attn_metadata
        if isinstance(m, FIAMetadata):
            active_uids = [int(r.uid) for r in batch.reqs]
            state.snapshots.append(
                StepSnapshot(
                    step_id=len(state.snapshots),
                    batch_size=int(m.batch_size),
                    active_uids=active_uids,
                    query_lengths=list(m.query_seq_lens),
                    kv_lengths=list(m.kv_seq_lens),
                )
            )
            if (
                not state.ready_to_admit_b
                and int(m.batch_size) == 1
                and list(m.query_seq_lens) == [1]
                and active_uids == [state.a_uid]
            ):
                state.ready_to_admit_b = True
            if (
                not state.ready_to_admit_c
                and int(m.batch_size) == 2
                and list(m.query_seq_lens) == [1, 1]
                and set(active_uids) == {state.a_uid, state.b_uid}
            ):
                state.ready_to_admit_c = True

    AscendFIABackend.prepare_metadata = patched  # type: ignore[assignment]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate 4.6 TP=2 dynamic grow-shrink B: 1→2→3→2→1 on Qwen3-0.6B",
    )
    parser.add_argument(
        "--model-path",
        default="/mnt/nvme/models/Qwen3-0.6B",
    )
    # Different max_new_tokens per uid so the completion order is
    # deterministic: A finishes first, then B, then C. Combined with
    # staggered admission this locks the shrink half of the timeline
    # (3 → 2 → 1) into a single unambiguous trace.
    parser.add_argument("--max-new-tokens-a", type=int, default=6)
    parser.add_argument("--max-new-tokens-b", type=int, default=8)
    parser.add_argument("--max-new-tokens-c", type=int, default=10)
    parser.add_argument(
        "--memory-ratio",
        type=float,
        default=0.85,
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--max-running-req",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--attention-backend",
        default="npu_fia",
    )
    parser.add_argument(
        "--prompt-a",
        default="Paris is",
        help="Request A prompt (starts first, finishes first).",
    )
    parser.add_argument(
        "--prompt-b",
        default="The capital of France is",
        help="Request B prompt (admitted after A decodes ≥1 step alone).",
    )
    parser.add_argument(
        "--prompt-c",
        default=(
            "The largest planet in our solar system by mass and volume is"
        ),
        help=(
            "Request C prompt (admitted after A and B jointly decode "
            "≥1 step)."
        ),
    )
    return parser.parse_args()


def _run_worker(args: argparse.Namespace) -> int:
    rank_env = os.environ.get("RANK")
    world_size_env = os.environ.get("WORLD_SIZE")
    local_rank_env = os.environ.get("LOCAL_RANK")
    if rank_env is None or world_size_env is None or local_rank_env is None:
        print(
            "[gate4.6] LOCAL_RANK / RANK / WORLD_SIZE must be set by the "
            "launcher (torchrun --nproc_per_node=2 ...)",
            file=sys.stderr,
            flush=True,
        )
        return 2

    rank = int(rank_env)
    world_size = int(world_size_env)
    local_rank = int(local_rank_env)

    # Gate 4.1 minimum fix carried forward: reuse torchrun's rendezvous store.
    os.environ.setdefault("MINISGL_DISTRIBUTED_ADDR", "env://")

    log = RankLog(
        rank=rank,
        world_size=world_size,
        tp_size=world_size,
        model_path=args.model_path,
    )

    if world_size != 2:
        log.failure_stage = "launch"
        log.failure_trace_summary = f"Gate 4.6 pins world_size=2; got {world_size}"
        log.status = "BLOCKED"
        log.emit()
        return 2

    state = SharedState()
    llm = None

    # ------------------------------------------------------------------
    # A. Init + B. Load (share one LLM boot; set_tp_info is one-shot)
    # ------------------------------------------------------------------
    try:
        import torch
        from minisgl.core import SamplingParams
        from minisgl.distributed import DistributedInfo
        from minisgl.llm import LLM
        from minisgl.llm.llm import RequestAllFinished
        from minisgl.message import BaseBackendMsg

        tp_info = DistributedInfo(rank=rank, size=world_size)
        log.device = f"npu:{local_rank}"

        # Install the metadata snapshot hook BEFORE LLM() so every
        # prepare_metadata call (including any at boot) is captured.
        _install_metadata_snapshot_hook(state)

        # Staggered-admission LLM subclass. Only A is in
        # ``pending_requests`` at start; B lives in ``_staged_b`` and C
        # in ``_staged_c``. B is revealed to the scheduler on the first
        # tick after ``state.ready_to_admit_b`` flips; C is revealed on
        # the first tick after ``state.ready_to_admit_c`` flips (which
        # itself only flips after A and B have jointly decoded ≥1
        # step, guaranteeing the B=3 joint decode step Gate 4.6
        # requires).
        class GrowShrinkLLM(LLM):
            def __init__(self_inner, *a, **kw):
                super().__init__(*a, **kw)
                self_inner._staged_b = None  # type: ignore[attr-defined]
                self_inner._staged_c = None  # type: ignore[attr-defined]
                self_inner._b_revealed = False  # type: ignore[attr-defined]
                self_inner._c_revealed = False  # type: ignore[attr-defined]

            def offline_receive_msg(
                self_inner, blocking: bool = False
            ) -> List["BaseBackendMsg"]:
                # Reveal B iff (a) not yet revealed, (b) the hook has
                # observed A alone decoding at least once. Both ranks
                # arrive at this condition on the SAME tick because
                # both ranks run identical forwards in HCCL lockstep.
                if (
                    not self_inner._b_revealed
                    and state.ready_to_admit_b
                    and self_inner._staged_b is not None
                ):
                    self_inner.pending_requests.append(self_inner._staged_b)
                    self_inner._staged_b = None
                    self_inner._b_revealed = True
                # Reveal C iff (a) not yet revealed, (b) the hook has
                # observed A+B jointly decoding at least once.
                if (
                    not self_inner._c_revealed
                    and state.ready_to_admit_c
                    and self_inner._staged_c is not None
                ):
                    self_inner.pending_requests.append(self_inner._staged_c)
                    self_inner._staged_c = None
                    self_inner._c_revealed = True
                return super().offline_receive_msg(blocking=blocking)

        llm = GrowShrinkLLM(
            model_path=args.model_path,
            dtype=torch.bfloat16,
            tp_info=tp_info,
            use_pynccl=False,
            attention_backend=args.attention_backend,
            max_running_req=args.max_running_req,
            memory_ratio=args.memory_ratio,
            cuda_graph_bs=[],
            page_size=args.page_size,
        )

        log.init_status = "PASS"
        log.load_status = "PASS"

        baseline = _snapshot(llm.cache_manager)
        log.baseline_available_tokens = baseline["available_tokens"]
        log.baseline_free_pages = baseline["free_pages"]
        log.total_pages = baseline["total_pages"]
        if not baseline["integrity_ok"]:
            log.status = "FAIL"
            log.failure_stage = "post_load_integrity"
            log.failure_trace_summary = "check_integrity() raised after weight load"
            log.emit()
            return 1

    except BaseException as exc:
        log.failure_stage = "init" if log.init_status == "PENDING" else "load"
        log.failure_trace_summary = (
            f"{type(exc).__name__}: {exc}\n"
            + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:]
        )
        log.status = "BLOCKED" if log.failure_stage == "init" else "FAIL"
        log.emit()
        return 2 if log.status == "BLOCKED" else 1

    # Discard any pre-run snapshots (empty batches never reach FIA,
    # so in practice this is 0 — but the guard keeps the trace clean).
    snapshots_pre_run = len(state.snapshots)

    # ------------------------------------------------------------------
    # C. TP=2 dynamic grow-shrink B: 1 → 2 → 3 → 2 → 1
    # ------------------------------------------------------------------
    try:
        sp_a = SamplingParams(
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            max_tokens=args.max_new_tokens_a,
            ignore_eos=True,
        )
        sp_b = SamplingParams(
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            max_tokens=args.max_new_tokens_b,
            ignore_eos=True,
        )
        sp_c = SamplingParams(
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            max_tokens=args.max_new_tokens_c,
            ignore_eos=True,
        )

        prompts = [args.prompt_a, args.prompt_b, args.prompt_c]
        tokenized_lengths = [len(llm.tokenizer.encode(p)) for p in prompts]
        log.prompt_token_lengths = tokenized_lengths
        log.max_new_tokens_per_request = [
            args.max_new_tokens_a,
            args.max_new_tokens_b,
            args.max_new_tokens_c,
        ]

        # Stage: only A is visible via pending_requests; B and C are
        # held in the subclass' _staged_b / _staged_c slots until the
        # hook signals ready. Uid assignment order is fixed by
        # LLM.counter: A=0, B=1, C=2 on both ranks.
        llm.pending_requests = [(prompts[0], sp_a)]
        llm.status_map = {}
        llm.counter = 0
        llm._staged_b = (prompts[1], sp_b)  # type: ignore[attr-defined]
        llm._staged_c = (prompts[2], sp_c)  # type: ignore[attr-defined]
        llm._b_revealed = False  # type: ignore[attr-defined]
        llm._c_revealed = False  # type: ignore[attr-defined]
        state.ready_to_admit_b = False
        state.ready_to_admit_c = False
        state.a_uid = 0
        state.b_uid = 1
        state.c_uid = 2

        log.request_uids = [state.a_uid, state.b_uid, state.c_uid]

        run_t0 = time.perf_counter()
        try:
            llm.run_forever()
        except RequestAllFinished:
            pass
        run_t1 = time.perf_counter()
        _err(rank, f"run_forever() drained in {(run_t1 - run_t0)*1000.0:.2f} ms")

        # All three requests should now be in status_map.
        expected_uids = [state.a_uid, state.b_uid, state.c_uid]
        for uid in expected_uids:
            if uid not in llm.status_map:
                raise RuntimeError(
                    f"expected uids {expected_uids} in status_map; "
                    f"got {sorted(llm.status_map.keys())}"
                )

        results = []
        for uid in expected_uids:
            status = llm.status_map[uid]
            text = llm.tokenizer.decode(status.output_ids)
            results.append({"uid": uid, "token_ids": status.output_ids, "text": text})

        actual_tokens_per_req = [len(r["token_ids"]) for r in results]
        expected = [
            args.max_new_tokens_a,
            args.max_new_tokens_b,
            args.max_new_tokens_c,
        ]
        if actual_tokens_per_req != expected:
            raise RuntimeError(
                f"expected {expected} output tokens per request, "
                f"got {actual_tokens_per_req}"
            )

        log.prefill_status = "PASS"
        log.decode_status = "PASS"
        log.actual_output_tokens_per_request = actual_tokens_per_req
        log.output_texts = [r["text"] for r in results]

        # Slice snapshots that belong to this run.
        run_snapshots = state.snapshots[snapshots_pre_run:]
        for i, s in enumerate(run_snapshots):
            s.step_id = i

        log.all_step_snapshots = [s.to_dict() for s in run_snapshots]
        log.batch_timeline = [s.batch_size for s in run_snapshots]
        log.decode_step_snapshots = [
            s.to_dict()
            for s in run_snapshots
            if all(q == 1 for q in s.query_lengths)
        ]
        log.joint_decode_step_count_b2 = sum(
            1
            for s in run_snapshots
            if s.batch_size == 2 and s.query_lengths == [1, 1]
        )
        log.triple_decode_step_count = sum(
            1
            for s in run_snapshots
            if s.batch_size == 3 and s.query_lengths == [1, 1, 1]
        )

        # Derive admission / completion events by scanning active_uids.
        first_seen: Dict[int, int] = {}
        last_seen: Dict[int, int] = {}
        for s in run_snapshots:
            for u in s.active_uids:
                if u not in first_seen:
                    first_seen[u] = s.step_id
                last_seen[u] = s.step_id
        log.admission_events = {str(u): first_seen[u] for u in sorted(first_seen)}
        log.completion_events = {str(u): last_seen[u] for u in sorted(last_seen)}

        # Timeline invariant: the sequence [1, 2, 3, 2, 1] must appear
        # as an ordered subsequence (not necessarily contiguous). Walk
        # a five-state machine over batch_timeline.
        timeline = log.batch_timeline
        target = [1, 2, 3, 2, 1]
        idx = 0
        for bs in timeline:
            if idx < len(target) and bs == target[idx]:
                idx += 1
            if idx == len(target):
                break
        timeline_ok = idx == len(target)

        # Admission ordering: A must be admitted before B, and B
        # before C (uids are assigned by increment of self.counter,
        # so first_seen[a] < first_seen[b] < first_seen[c] is the
        # required trace).
        a_first = first_seen.get(state.a_uid)
        b_first = first_seen.get(state.b_uid)
        c_first = first_seen.get(state.c_uid)
        admission_ok = (
            a_first is not None
            and b_first is not None
            and c_first is not None
            and a_first < b_first < c_first
        )

        joint_b2_ok = (
            log.joint_decode_step_count_b2 is not None
            and log.joint_decode_step_count_b2 >= 1
        )
        joint_b3_ok = (
            log.triple_decode_step_count is not None
            and log.triple_decode_step_count >= 1
        )

        log.admission_status = "PASS" if admission_ok else "FAIL"
        log.timeline_status = (
            "PASS" if (timeline_ok and joint_b2_ok and joint_b3_ok) else "FAIL"
        )

        if not (admission_ok and timeline_ok and joint_b2_ok and joint_b3_ok):
            raise RuntimeError(
                "Gate 4.6 dynamic grow-shrink invariants not met: "
                f"admission_ok={admission_ok} (a_first={a_first!r}, "
                f"b_first={b_first!r}, c_first={c_first!r}); "
                f"timeline_ok={timeline_ok} (batch_timeline={timeline!r}); "
                f"joint_decode_step_count_b2={log.joint_decode_step_count_b2!r}; "
                f"triple_decode_step_count={log.triple_decode_step_count!r}"
            )

    except BaseException as exc:
        if log.prefill_status == "PENDING":
            log.failure_stage = "prefill"
        elif log.decode_status == "PENDING":
            log.failure_stage = "decode"
        elif log.admission_status != "PASS":
            log.failure_stage = "admission_evidence"
        elif log.timeline_status != "PASS":
            log.failure_stage = "timeline_evidence"
        else:
            log.failure_stage = "post_generate"
        log.failure_trace_summary = (
            f"{type(exc).__name__}: {exc}\n"
            + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:]
        )
        log.status = "FAIL"
        try:
            if llm is not None:
                llm.shutdown()
        except BaseException as shutdown_exc:
            _err(rank, f"shutdown after failure raised: {shutdown_exc!r}")
        log.emit()
        return 1

    # ------------------------------------------------------------------
    # Post-case allocator invariant + symmetric shutdown
    # ------------------------------------------------------------------
    try:
        after = _snapshot(llm.cache_manager)
        log.available_tokens_after_case = after["available_tokens"]
        log.free_pages_after_case = after["free_pages"]
        log.free_pages_before_after = [log.baseline_free_pages, after["free_pages"]]
        log.deferred_abort_uids = len(llm.deferred_abort_uids)
        log.cache_integrity_ok = after["integrity_ok"]

        invariants_ok = (
            after["integrity_ok"]
            and log.deferred_abort_uids == 0
            and after["available_tokens"] == log.baseline_available_tokens
        )
        if invariants_ok:
            log.status = "PASS"
        else:
            log.status = "FAIL"
            log.failure_stage = "post_case_invariant"
            log.failure_trace_summary = (
                f"available_tokens_after={after['available_tokens']!r} "
                f"vs baseline={log.baseline_available_tokens!r}; "
                f"deferred_abort_uids={log.deferred_abort_uids!r}; "
                f"cache_integrity_ok={after['integrity_ok']!r}"
            )
    except BaseException as exc:
        log.failure_stage = "post_case_invariant"
        log.failure_trace_summary = (
            f"{type(exc).__name__}: {exc}\n"
            + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:]
        )
        log.status = "FAIL"

    try:
        llm.shutdown()
    except BaseException as exc:
        _err(rank, f"shutdown raised: {exc!r}")
        if log.status == "PASS":
            log.status = "FAIL"
            log.failure_stage = "shutdown"
            log.failure_trace_summary = f"{type(exc).__name__}: {exc}"

    log.emit()
    return 0 if log.status == "PASS" else 1


def main() -> int:
    args = _parse_args()
    return _run_worker(args)


if __name__ == "__main__":
    sys.exit(main())
