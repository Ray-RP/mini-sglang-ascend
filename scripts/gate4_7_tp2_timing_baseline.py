"""Gate 4.7 TP=2 timing baseline smoke driver on Qwen3-0.6B.

Runs six timing cases end-to-end on 2× Ascend 910B1, each with 1
warmup pass and 3 measured passes, and emits a structured per-rank
per-repeat JSONL trace plus a per-case median/min/max summary.

Cases (locked at Gate 4.7):
    A. B=1 single request, max_new_tokens=8
    B. B=1 single request, max_new_tokens=16
    C. B=2 equal-length, max_new_tokens=8
    D. B=2 ragged prefill (unequal-length), max_new_tokens=8
    E. B=2 mixed-KV decode evidence (unequal prefills), max_new_tokens=8
    F. dynamic grow-shrink B: 1 → 2 → 3 → 2 → 1

Envelope (locked at Gate 4.7):
    Ascend 910B1, TP=2, eager, npu_fia, bf16, greedy
    memory_ratio=0.85, page_size=16, max_running_req=4
    cuda_graph_bs=[] (torch_npu has no CUDAGraph)
    warmup = 1, measured_repeats = 3

Timing semantics:
    * ``t0`` — ``time.perf_counter()`` immediately before the
      offline generate / run_forever call for the repeat
    * ``ttft_ms`` — approximated as the wall time of the FIRST
      pure-decode step (query_lengths all == 1) minus t0. The first
      pure-decode step's ``prepare_metadata`` fires AFTER the first
      forward pass has produced its first output token and returned
      it to the sampler, so this is a close upper bound on TTFT
      without instrumenting the sampler itself.
    * ``e2e_latency_ms`` — ``time.perf_counter() - t0`` at generate /
      run_forever return
    * ``output_tokens_total`` — sum of per-request produced tokens
    * ``tokens_per_second`` — output_tokens_total / (e2e_latency / 1000)
    * ``ms_per_output_token`` — e2e_latency_ms / output_tokens_total

    These are internal reproducibility numbers. They are NOT
    benchmarks, they are NOT compared against SGLang / vLLM / TGI,
    they do NOT support any performance-superiority claim. The
    metadata-snapshot hook itself adds unbudgeted per-step
    overhead — every measurement here includes that overhead
    equally.

Allocator invariant (checked after every measured repeat):
    ``available_tokens_after == baseline_available_tokens``
    ``deferred_abort_uids == 0``
    ``cache_integrity_ok == True``

Case F reuses the Gate 4.6 grow-shrink pattern: script-local
``GrowShrinkLLM(LLM)`` subclass whose overridden
``offline_receive_msg`` reveals request B once the metadata hook
observes A alone decoding, and reveals request C once the hook
observes A+B jointly decoding.

Structured log format (per rank):
    * one ``GATE4.7_JSONL rank=<r> {...}`` line per (case, warmup)
      and per (case, repeat_id) — the raw measurement records
    * one ``GATE4.7_SUMMARY rank=<r> {...}`` line per case — the
      median / min / max summary
    * one final ``GATE4.7_TIMING_RESULT=<PASS|PARTIAL|BLOCKED>``
      footer line on rank 0

This script does not tune, does not sweep, does not touch
python/minisgl/.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------
# Timing-capable metadata hook (shared across all cases and repeats).
# ---------------------------------------------------------------------
@dataclass
class HookState:
    # Set at the top of each repeat; the hook filters snapshots by the
    # generation ID so warmup / repeat traces do not cross-contaminate.
    active_gen_id: int = -1
    # Per-forward-step observations recorded within the current repeat.
    # Each entry is (t_wall, batch_size, query_lengths).
    steps: List[Tuple[float, int, List[int]]] = field(default_factory=list)


def _install_metadata_snapshot_hook(state: HookState) -> None:
    """Wrap ``AscendFIABackend.prepare_metadata`` to timestamp each
    forward pass. Script-only monkey-patch — the runtime source is
    not modified.
    """
    from minisgl.attention.ascend_fia import AscendFIABackend, FIAMetadata

    original = AscendFIABackend.prepare_metadata

    def patched(self, batch):  # type: ignore[no-redef]
        original(self, batch)
        m = batch.attn_metadata
        if isinstance(m, FIAMetadata) and state.active_gen_id >= 0:
            state.steps.append(
                (
                    time.perf_counter(),
                    int(m.batch_size),
                    list(m.query_seq_lens),
                )
            )

    AscendFIABackend.prepare_metadata = patched  # type: ignore[assignment]


# ---------------------------------------------------------------------
# Case F: shared state + grow-shrink LLM subclass helpers.
# Kept parallel to Gate 4.6 so behaviour is unchanged.
# ---------------------------------------------------------------------
@dataclass
class GrowShrinkState:
    ready_to_admit_b: bool = False
    ready_to_admit_c: bool = False
    a_uid: int = 0
    b_uid: int = 1
    c_uid: int = 2


def _install_grow_shrink_hook(hook_state: HookState, gs_state: GrowShrinkState) -> None:
    """Reinstall a hook that ALSO drives the two admission flags for
    case F. Applied after ``_install_metadata_snapshot_hook`` for
    case F only; reverts to timing-only for other cases via a second
    reinstall driven by ``_install_metadata_snapshot_hook``.

    The grow-shrink hook must timestamp AND admission-flag; the
    timing-only hook must ONLY timestamp. To avoid double-patching we
    read the original once at module import time via
    ``AscendFIABackend.prepare_metadata`` and store it on the class.
    """
    from minisgl.attention.ascend_fia import AscendFIABackend, FIAMetadata

    original = AscendFIABackend.prepare_metadata

    def patched(self, batch):  # type: ignore[no-redef]
        original(self, batch)
        m = batch.attn_metadata
        if isinstance(m, FIAMetadata) and hook_state.active_gen_id >= 0:
            active_uids = [int(r.uid) for r in batch.reqs]
            hook_state.steps.append(
                (
                    time.perf_counter(),
                    int(m.batch_size),
                    list(m.query_seq_lens),
                )
            )
            if (
                not gs_state.ready_to_admit_b
                and int(m.batch_size) == 1
                and list(m.query_seq_lens) == [1]
                and active_uids == [gs_state.a_uid]
            ):
                gs_state.ready_to_admit_b = True
            if (
                not gs_state.ready_to_admit_c
                and int(m.batch_size) == 2
                and list(m.query_seq_lens) == [1, 1]
                and set(active_uids) == {gs_state.a_uid, gs_state.b_uid}
            ):
                gs_state.ready_to_admit_c = True

    AscendFIABackend.prepare_metadata = patched  # type: ignore[assignment]


# ---------------------------------------------------------------------
# Structured log records.
# ---------------------------------------------------------------------
@dataclass
class RepeatRecord:
    rank: int
    world_size: int
    tp_size: int
    model_path: str
    case_name: str
    kind: str  # "warmup" | "measured"
    repeat_id: int
    prompt_lengths: List[int]
    batch_timeline: Optional[List[int]]
    requested_max_new_tokens: List[int]
    actual_output_tokens_per_request: List[int]
    warmup_count: int
    ttft_ms: Optional[float]
    e2e_latency_ms: float
    output_tokens_total: int
    tokens_per_second: Optional[float]
    ms_per_output_token: Optional[float]
    baseline_available_tokens: int
    available_tokens_after_case: int
    free_pages_before_after: List[int]
    deferred_abort_uids: int
    cache_integrity_ok: bool
    status: str
    failure_reason: Optional[str] = None

    def emit(self) -> None:
        payload = {
            "rank": self.rank,
            "world_size": self.world_size,
            "tp_size": self.tp_size,
            "model_path": self.model_path,
            "case_name": self.case_name,
            "kind": self.kind,
            "repeat_id": self.repeat_id,
            "prompt_lengths": self.prompt_lengths,
            "batch_timeline": self.batch_timeline,
            "requested_max_new_tokens": self.requested_max_new_tokens,
            "actual_output_tokens_per_request": self.actual_output_tokens_per_request,
            "warmup_count": self.warmup_count,
            "ttft_ms": self.ttft_ms,
            "e2e_latency_ms": self.e2e_latency_ms,
            "output_tokens_total": self.output_tokens_total,
            "tokens_per_second": self.tokens_per_second,
            "ms_per_output_token": self.ms_per_output_token,
            "baseline_available_tokens": self.baseline_available_tokens,
            "available_tokens_after_case": self.available_tokens_after_case,
            "free_pages_before_after": self.free_pages_before_after,
            "deferred_abort_uids": self.deferred_abort_uids,
            "cache_integrity_ok": self.cache_integrity_ok,
            "status": self.status,
            "failure_reason": self.failure_reason,
        }
        print(
            f"GATE4.7_JSONL rank={self.rank} {json.dumps(payload, ensure_ascii=False)}",
            flush=True,
        )


def _err(rank: int, msg: str) -> None:
    print(f"[gate4.7][rank={rank}] {msg}", flush=True)


def _snapshot(cache_manager) -> Dict[str, Any]:
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
# Case definitions.
# ---------------------------------------------------------------------
@dataclass
class Case:
    name: str
    kind: str  # "static_b1" | "static_b2" | "grow_shrink"
    prompts: List[str]
    max_new_tokens: List[int]


def _build_cases() -> List[Case]:
    short = "Paris is"
    medium_equal_1 = "The capital of France is"
    medium_equal_2 = "The capital of Italy is now"  # 5 tokens on Qwen3 tokenizer? adjust to equal-length via encode check at runtime
    long = "The largest planet in our solar system by mass and volume is"
    grow_b = "The capital of France is"
    grow_c = long
    return [
        Case("A_b1_maxnew8", "static_b1", [short], [8]),
        Case("B_b1_maxnew16", "static_b1", [short], [16]),
        Case("C_b2_equal_maxnew8", "static_b2", [short, short], [8, 8]),
        Case("D_b2_ragged_maxnew8", "static_b2", [short, long], [8, 8]),
        Case("E_b2_mixed_kv_maxnew8", "static_b2", [short, long], [8, 8]),
        Case("F_grow_shrink", "grow_shrink", [short, grow_b, grow_c], [6, 8, 10]),
    ]


# ---------------------------------------------------------------------
# Per-repeat runners.
# ---------------------------------------------------------------------
def _run_static_repeat(
    llm,
    case: Case,
    hook_state: HookState,
    gen_id: int,
    sampling_params_cls,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run one repeat of a case whose prompts are all revealed to the
    scheduler at once (static B=1 or B=2). Returns (metrics, results).
    """
    sps = [
        sampling_params_cls(
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            max_tokens=n,
            ignore_eos=True,
        )
        for n in case.max_new_tokens
    ]
    # Reset scheduler state.
    llm.pending_requests = list(zip(case.prompts, sps))
    llm.status_map = {}
    llm.counter = 0

    # Reset hook.
    hook_state.steps.clear()
    hook_state.active_gen_id = gen_id

    t0 = time.perf_counter()
    from minisgl.llm.llm import RequestAllFinished
    try:
        llm.run_forever()
    except RequestAllFinished:
        pass
    t_end = time.perf_counter()
    hook_state.active_gen_id = -1

    # Collect results.
    results = []
    for uid in sorted(llm.status_map.keys()):
        status = llm.status_map[uid]
        results.append({"uid": uid, "token_ids": list(status.output_ids)})

    e2e_ms = (t_end - t0) * 1000.0

    # Derive ttft from first pure-decode step (all query_lengths == 1).
    steps = hook_state.steps
    batch_timeline = [bs for (_, bs, _) in steps]
    ttft_ms: Optional[float] = None
    for (t_step, _, qlens) in steps:
        if all(q == 1 for q in qlens):
            ttft_ms = (t_step - t0) * 1000.0
            break

    output_tokens_total = sum(len(r["token_ids"]) for r in results)
    tps = (
        output_tokens_total / (e2e_ms / 1000.0)
        if e2e_ms > 0 and output_tokens_total > 0
        else None
    )
    ms_per_tok = e2e_ms / output_tokens_total if output_tokens_total > 0 else None

    return (
        {
            "batch_timeline": batch_timeline,
            "ttft_ms": ttft_ms,
            "e2e_latency_ms": e2e_ms,
            "output_tokens_total": output_tokens_total,
            "tokens_per_second": tps,
            "ms_per_output_token": ms_per_tok,
        },
        results,
    )


def _run_grow_shrink_repeat(
    llm,
    case: Case,
    hook_state: HookState,
    gs_state: GrowShrinkState,
    gen_id: int,
    sampling_params_cls,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run one repeat of case F. Behaviour is a direct port of the
    Gate 4.6 driver: stage B and C, reveal via subclass override, run
    to drain.
    """
    sps = [
        sampling_params_cls(
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            max_tokens=n,
            ignore_eos=True,
        )
        for n in case.max_new_tokens
    ]
    llm.pending_requests = [(case.prompts[0], sps[0])]
    llm.status_map = {}
    llm.counter = 0
    llm._staged_b = (case.prompts[1], sps[1])  # type: ignore[attr-defined]
    llm._staged_c = (case.prompts[2], sps[2])  # type: ignore[attr-defined]
    llm._b_revealed = False  # type: ignore[attr-defined]
    llm._c_revealed = False  # type: ignore[attr-defined]
    gs_state.ready_to_admit_b = False
    gs_state.ready_to_admit_c = False
    gs_state.a_uid = 0
    gs_state.b_uid = 1
    gs_state.c_uid = 2

    hook_state.steps.clear()
    hook_state.active_gen_id = gen_id

    t0 = time.perf_counter()
    from minisgl.llm.llm import RequestAllFinished
    try:
        llm.run_forever()
    except RequestAllFinished:
        pass
    t_end = time.perf_counter()
    hook_state.active_gen_id = -1

    results = []
    for uid in sorted(llm.status_map.keys()):
        status = llm.status_map[uid]
        results.append({"uid": uid, "token_ids": list(status.output_ids)})

    e2e_ms = (t_end - t0) * 1000.0

    steps = hook_state.steps
    batch_timeline = [bs for (_, bs, _) in steps]
    ttft_ms: Optional[float] = None
    for (t_step, _, qlens) in steps:
        if all(q == 1 for q in qlens):
            ttft_ms = (t_step - t0) * 1000.0
            break

    output_tokens_total = sum(len(r["token_ids"]) for r in results)
    tps = (
        output_tokens_total / (e2e_ms / 1000.0)
        if e2e_ms > 0 and output_tokens_total > 0
        else None
    )
    ms_per_tok = e2e_ms / output_tokens_total if output_tokens_total > 0 else None

    return (
        {
            "batch_timeline": batch_timeline,
            "ttft_ms": ttft_ms,
            "e2e_latency_ms": e2e_ms,
            "output_tokens_total": output_tokens_total,
            "tokens_per_second": tps,
            "ms_per_output_token": ms_per_tok,
        },
        results,
    )


# ---------------------------------------------------------------------
# Summary emission.
# ---------------------------------------------------------------------
def _emit_summary(
    rank: int,
    case_name: str,
    measured: List[Dict[str, Any]],
) -> None:
    def _stats(values: List[float]) -> Optional[Dict[str, float]]:
        filtered = [v for v in values if v is not None]
        if not filtered:
            return None
        return {
            "median": statistics.median(filtered),
            "min": min(filtered),
            "max": max(filtered),
        }

    payload = {
        "rank": rank,
        "case_name": case_name,
        "repeats_recorded": len(measured),
        "ttft_ms": _stats([r["ttft_ms"] for r in measured]),
        "e2e_latency_ms": _stats([r["e2e_latency_ms"] for r in measured]),
        "tokens_per_second": _stats([r["tokens_per_second"] for r in measured]),
        "ms_per_output_token": _stats([r["ms_per_output_token"] for r in measured]),
    }
    print(
        f"GATE4.7_SUMMARY rank={rank} {json.dumps(payload, ensure_ascii=False)}",
        flush=True,
    )


# ---------------------------------------------------------------------
# Argument parsing.
# ---------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate 4.7 TP=2 timing baseline on Qwen3-0.6B",
    )
    parser.add_argument("--model-path", default="/mnt/nvme/models/Qwen3-0.6B")
    parser.add_argument("--memory-ratio", type=float, default=0.85)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--max-running-req", type=int, default=4)
    parser.add_argument("--attention-backend", default="npu_fia")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measured-repeats", type=int, default=3)
    return parser.parse_args()


# ---------------------------------------------------------------------
# Per-rank worker.
# ---------------------------------------------------------------------
def _run_worker(args: argparse.Namespace) -> int:
    rank_env = os.environ.get("RANK")
    world_size_env = os.environ.get("WORLD_SIZE")
    local_rank_env = os.environ.get("LOCAL_RANK")
    if rank_env is None or world_size_env is None or local_rank_env is None:
        print(
            "[gate4.7] LOCAL_RANK / RANK / WORLD_SIZE must be set by "
            "torchrun --nproc_per_node=2",
            file=sys.stderr,
            flush=True,
        )
        return 2

    rank = int(rank_env)
    world_size = int(world_size_env)
    local_rank = int(local_rank_env)
    os.environ.setdefault("MINISGL_DISTRIBUTED_ADDR", "env://")

    if world_size != 2:
        _err(rank, f"Gate 4.7 pins world_size=2; got {world_size}")
        if rank == 0:
            print("GATE4.7_TIMING_RESULT=BLOCKED", flush=True)
        return 2

    llm = None
    hook_state = HookState()
    gs_state = GrowShrinkState()

    try:
        import torch
        from minisgl.core import SamplingParams
        from minisgl.distributed import DistributedInfo
        from minisgl.llm import LLM
        from minisgl.message import BaseBackendMsg  # noqa: F401 (for typing anchor)

        tp_info = DistributedInfo(rank=rank, size=world_size)

        # Install timing hook first. Case F reinstalls with grow-shrink
        # logic wrapped around the same original method; installing the
        # grow-shrink hook AFTER the timing-only hook would double-time
        # every step, so we install the grow-shrink hook LAST at the
        # start of case F and never reinstall.
        _install_metadata_snapshot_hook(hook_state)

        # Build a script-local GrowShrinkLLM to support case F. All
        # other cases use the same object with pending_requests fully
        # revealed up front (staged slots stay None so the override
        # is a no-op).
        class GrowShrinkLLM(LLM):
            def __init__(self_inner, *a, **kw):
                super().__init__(*a, **kw)
                self_inner._staged_b = None  # type: ignore[attr-defined]
                self_inner._staged_c = None  # type: ignore[attr-defined]
                self_inner._b_revealed = False  # type: ignore[attr-defined]
                self_inner._c_revealed = False  # type: ignore[attr-defined]

            def offline_receive_msg(self_inner, blocking: bool = False):
                if (
                    not self_inner._b_revealed
                    and gs_state.ready_to_admit_b
                    and self_inner._staged_b is not None
                ):
                    self_inner.pending_requests.append(self_inner._staged_b)
                    self_inner._staged_b = None
                    self_inner._b_revealed = True
                if (
                    not self_inner._c_revealed
                    and gs_state.ready_to_admit_c
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

        baseline = _snapshot(llm.cache_manager)
        baseline_available_tokens = baseline["available_tokens"]
        baseline_free_pages = baseline["free_pages"]
        if not baseline["integrity_ok"]:
            _err(rank, "post-load integrity check failed")
            if rank == 0:
                print("GATE4.7_TIMING_RESULT=BLOCKED", flush=True)
            llm.shutdown()
            return 1

    except BaseException as exc:
        _err(
            rank,
            f"init/load failed: {type(exc).__name__}: {exc}\n"
            + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-2000:],
        )
        if rank == 0:
            print("GATE4.7_TIMING_RESULT=BLOCKED", flush=True)
        return 2

    cases = _build_cases()
    result_flag = "PASS"
    gen_id = 0

    # For case F only: reinstall hook with the grow-shrink admission
    # logic just before the case runs, so warmup + repeats share the
    # combined hook. This replaces (not stacks on) the timing-only
    # hook because we re-read AscendFIABackend.prepare_metadata as
    # the "original" — but by then it's already patched. Guard against
    # that by keeping a reference to the truly-original method.
    from minisgl.attention.ascend_fia import AscendFIABackend  # noqa: E402

    truly_original = AscendFIABackend.prepare_metadata

    def _restore_and_install_gs():
        AscendFIABackend.prepare_metadata = truly_original  # type: ignore[assignment]
        _install_grow_shrink_hook(hook_state, gs_state)

    def _restore_and_install_timing():
        AscendFIABackend.prepare_metadata = truly_original  # type: ignore[assignment]
        _install_metadata_snapshot_hook(hook_state)

    # NOTE: The initial hook (installed before LLM boot) already wraps
    # AscendFIABackend.prepare_metadata. `truly_original` here is
    # therefore the ALREADY-WRAPPED method, and calling it inside a
    # second wrapper would still be correct because the wrapper
    # delegates to `original(self, batch)` first. However, we want
    # to avoid stacking wrappers, so we track the raw class attr
    # captured BEFORE installing the timing hook. Do that now by
    # deferring hook install until inside the try/except above? Too
    # late — the hook is already installed. For simplicity we accept
    # that after case F the hook stack has depth 2 (timing_only ->
    # grow_shrink -> original). This adds only trivial per-step
    # overhead and does not affect timing semantics — both wrappers
    # append the same timestamp for the same forward pass, but only
    # the outermost's append survives because both mutate the same
    # `state.steps` list — the innermost writes first, the outermost
    # second, so `state.steps` grows by 2 per forward pass under the
    # combined hook. To avoid this we always run case F LAST and
    # never reinstall after it. But even simpler: we accept
    # duplication only during case F and de-duplicate downstream by
    # keeping only unique timestamps. Downstream de-dup is done
    # implicitly by `batch_timeline` reporting; ttft still uses the
    # earliest timestamp of a pure-decode step which is the first
    # append, so semantics are preserved.
    #
    # In practice we sidestep the whole stack question by NOT
    # reinstalling: case F reuses the timing-only hook plus a
    # scheduler-tick check inside the LLM subclass override. We move
    # the ready-flag logic OUT of the metadata hook and INTO the
    # subclass override, checking each tick whether a triggering
    # forward pass has already occurred by scanning the recorded
    # hook_state.steps buffer.
    #
    # Redo case F flag logic to avoid hook reinstall:
    def _case_f_flag_from_hook() -> None:
        for (_, bs, qlens) in hook_state.steps:
            if bs == 1 and qlens == [1] and not gs_state.ready_to_admit_b:
                # active_uids is not stored in this trimmed hook; we
                # only know batch shape. That is enough for
                # ready_to_admit_b because the only running request
                # at this point is A (uid 0) — no other uid has been
                # revealed yet. Once we flip ready_to_admit_b, B is
                # revealed on the next tick.
                gs_state.ready_to_admit_b = True
            if bs == 2 and qlens == [1, 1] and not gs_state.ready_to_admit_c:
                # By the time we see B=2 with all queries==1, the two
                # active uids MUST be {A, B} because C has not yet
                # been revealed.
                gs_state.ready_to_admit_c = True

    # Rewire GrowShrinkLLM.offline_receive_msg to consult the helper
    # before the parent method runs. We monkey-patch the class here to
    # avoid duplicating the class body.
    _original_offline_receive_msg = GrowShrinkLLM.offline_receive_msg  # noqa: F821

    def _wrapped_offline(self_inner, blocking: bool = False):
        _case_f_flag_from_hook()
        return _original_offline_receive_msg(self_inner, blocking)

    GrowShrinkLLM.offline_receive_msg = _wrapped_offline  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Run every case: warmup passes first (emitted with kind="warmup",
    # no summary contribution), then measured passes.
    # ------------------------------------------------------------------
    for case in cases:
        prompt_lengths = [len(llm.tokenizer.encode(p)) for p in case.prompts]

        # Warmup.
        for w in range(args.warmup):
            gen_id += 1
            try:
                if case.kind == "grow_shrink":
                    metrics, _results = _run_grow_shrink_repeat(
                        llm, case, hook_state, gs_state, gen_id, SamplingParams
                    )
                else:
                    metrics, _results = _run_static_repeat(
                        llm, case, hook_state, gen_id, SamplingParams
                    )
                after = _snapshot(llm.cache_manager)
                rec = RepeatRecord(
                    rank=rank,
                    world_size=world_size,
                    tp_size=world_size,
                    model_path=args.model_path,
                    case_name=case.name,
                    kind="warmup",
                    repeat_id=w,
                    prompt_lengths=prompt_lengths,
                    batch_timeline=metrics["batch_timeline"],
                    requested_max_new_tokens=case.max_new_tokens,
                    actual_output_tokens_per_request=(
                        [len(r["token_ids"]) for r in _results]
                        if _results else []
                    ),
                    warmup_count=args.warmup,
                    ttft_ms=metrics["ttft_ms"],
                    e2e_latency_ms=metrics["e2e_latency_ms"],
                    output_tokens_total=metrics["output_tokens_total"],
                    tokens_per_second=metrics["tokens_per_second"],
                    ms_per_output_token=metrics["ms_per_output_token"],
                    baseline_available_tokens=baseline_available_tokens,
                    available_tokens_after_case=after["available_tokens"],
                    free_pages_before_after=[baseline_free_pages, after["free_pages"]],
                    deferred_abort_uids=len(llm.deferred_abort_uids),
                    cache_integrity_ok=after["integrity_ok"],
                    status="PASS"
                    if (
                        after["available_tokens"] == baseline_available_tokens
                        and len(llm.deferred_abort_uids) == 0
                        and after["integrity_ok"]
                    )
                    else "FAIL",
                )
                rec.emit()
                if rec.status != "PASS":
                    _err(rank, f"warmup {case.name}#{w} allocator invariant failed")
                    result_flag = "PARTIAL"
            except BaseException as exc:
                _err(
                    rank,
                    f"warmup {case.name}#{w} raised: {type(exc).__name__}: {exc}",
                )
                result_flag = "PARTIAL"
                continue

        # Measured.
        measured_metrics: List[Dict[str, Any]] = []
        case_status = "PASS"
        for r_id in range(args.measured_repeats):
            gen_id += 1
            try:
                if case.kind == "grow_shrink":
                    metrics, results = _run_grow_shrink_repeat(
                        llm, case, hook_state, gs_state, gen_id, SamplingParams
                    )
                else:
                    metrics, results = _run_static_repeat(
                        llm, case, hook_state, gen_id, SamplingParams
                    )
                after = _snapshot(llm.cache_manager)
                actual_tokens = [len(r["token_ids"]) for r in results]
                inv_ok = (
                    after["available_tokens"] == baseline_available_tokens
                    and len(llm.deferred_abort_uids) == 0
                    and after["integrity_ok"]
                )
                status = "PASS" if inv_ok else "FAIL"
                rec = RepeatRecord(
                    rank=rank,
                    world_size=world_size,
                    tp_size=world_size,
                    model_path=args.model_path,
                    case_name=case.name,
                    kind="measured",
                    repeat_id=r_id,
                    prompt_lengths=prompt_lengths,
                    batch_timeline=metrics["batch_timeline"],
                    requested_max_new_tokens=case.max_new_tokens,
                    actual_output_tokens_per_request=actual_tokens,
                    warmup_count=args.warmup,
                    ttft_ms=metrics["ttft_ms"],
                    e2e_latency_ms=metrics["e2e_latency_ms"],
                    output_tokens_total=metrics["output_tokens_total"],
                    tokens_per_second=metrics["tokens_per_second"],
                    ms_per_output_token=metrics["ms_per_output_token"],
                    baseline_available_tokens=baseline_available_tokens,
                    available_tokens_after_case=after["available_tokens"],
                    free_pages_before_after=[baseline_free_pages, after["free_pages"]],
                    deferred_abort_uids=len(llm.deferred_abort_uids),
                    cache_integrity_ok=after["integrity_ok"],
                    status=status,
                )
                rec.emit()
                if status == "PASS":
                    measured_metrics.append(metrics)
                else:
                    case_status = "FAIL"
            except BaseException as exc:
                _err(
                    rank,
                    f"measured {case.name}#{r_id} raised: "
                    f"{type(exc).__name__}: {exc}",
                )
                case_status = "FAIL"

        _emit_summary(rank, case.name, measured_metrics)
        if case_status != "PASS":
            if case.name.startswith(("A_", "B_", "C_", "D_")):
                # A-D are the strict-required base cases; failing any
                # of them degrades to BLOCKED (per §8 decision matrix)
                result_flag = "BLOCKED"
            else:
                result_flag = "PARTIAL"

    try:
        llm.shutdown()
    except BaseException as exc:
        _err(rank, f"shutdown raised: {exc!r}")
        if result_flag == "PASS":
            result_flag = "PARTIAL"

    if rank == 0:
        print(f"GATE4.7_TIMING_RESULT={result_flag}", flush=True)
    return 0 if result_flag == "PASS" else (1 if result_flag == "PARTIAL" else 2)


def main() -> int:
    return _run_worker(_parse_args())


if __name__ == "__main__":
    sys.exit(main())
