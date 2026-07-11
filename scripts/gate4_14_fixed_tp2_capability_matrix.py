"""Gate 4.14 fixed-TP2 capability matrix smoke driver.

Runs six functional cases end-to-end on 2 x Ascend 910B1 for a single
model. Functional-only: no warmup, no timing statistics, no repeats.
Each case runs once; the driver records whether the runtime returns
the expected shapes, token counts, mixed-KV evidence, and dynamic
admission timeline.

Two-model coverage is delivered by running this script twice from an
outer shell loop -- once per model path -- and aggregating the two
JSONL streams into a single fixed-TP2 capability-matrix verdict.

Cases (locked at Gate 4.14):
    A. B=1 single request, max_new_tokens=8
    B. B=1 single request, max_new_tokens=16
    C. B=2 equal-length, max_new_tokens=8
    D. B=2 ragged prefill (unequal-length), max_new_tokens=8
    E. B=2 mixed-KV decode evidence (unequal prefills), max_new_tokens=8
    F. dynamic admission B: 1 -> 2 -> 1

Envelope (locked at Gate 4.14):
    Ascend 910B1, TP=2, eager, npu_fia, bf16, greedy
    memory_ratio=0.85, page_size=16, max_running_req=4
    cuda_graph_bs=[] (torch_npu has no CUDAGraph)
    use_pynccl = False (HCCL + gloo sidecar)

Allocator invariant (checked after every case):
    available_tokens_after_case == baseline_available_tokens
    deferred_abort_uids == 0
    cache_integrity_ok == True

Case F reuses the Gate 4.12 / 4.13 dynamic-admission pattern: a
script-local StaggeredLLM whose overridden offline_receive_msg
reveals staged request B once the metadata-snapshot hook observes A
alone decoding (batch_size == 1, query_lengths == [1]).

Structured log format (per rank):
    * one GATE4.14_JSONL rank=<r> {...} line per (model, case)
    * one GATE4.14_MATRIX_RESULT=<PASS|PARTIAL|BLOCKED> footer on rank 0

This script does not tune, does not sweep, does not touch
python/minisgl/. It is a functional-only capability probe.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------
# Metadata snapshot hook. Records batch_size / query_lengths / kv_lengths
# for every prepare_metadata call; the driver reads it back after each
# case to extract batch_timeline and mixed-KV evidence.
# ---------------------------------------------------------------------
@dataclass
class HookState:
    active_gen_id: int = -1
    # Per-forward-step observations recorded within the current case.
    # Each entry is (batch_size, query_lengths, kv_lengths).
    steps: List[Tuple[int, List[int], List[int]]] = field(default_factory=list)


def _install_metadata_snapshot_hook(state: HookState) -> None:
    from minisgl.attention.ascend_fia import AscendFIABackend, FIAMetadata

    original = AscendFIABackend.prepare_metadata

    def patched(self, batch):  # type: ignore[no-redef]
        original(self, batch)
        m = batch.attn_metadata
        if isinstance(m, FIAMetadata) and state.active_gen_id >= 0:
            state.steps.append(
                (
                    int(m.batch_size),
                    list(m.query_seq_lens),
                    list(m.kv_seq_lens),
                )
            )

    AscendFIABackend.prepare_metadata = patched  # type: ignore[assignment]


@dataclass
class AdmissionState:
    ready_to_admit_b: bool = False


# ---------------------------------------------------------------------
# Per-record structured log.
# ---------------------------------------------------------------------
@dataclass
class CaseRecord:
    model_name: str
    model_path: str
    rank: int
    world_size: int
    tp_size: int
    case_name: str
    prompt_token_lengths: List[int]
    requested_max_new_tokens: List[int]
    actual_output_tokens_per_request: List[int]
    output_texts: List[str]
    output_token_ids: List[List[int]]
    batch_timeline: Optional[List[int]]
    decode_metadata_summary: Optional[Dict[str, Any]]
    baseline_available_tokens: int
    available_tokens_after_case: int
    free_pages_before_after: List[int]
    deferred_abort_uids: int
    cache_integrity_ok: bool
    status: str
    failure_stage: Optional[str]
    cold_start_attempt_id: int
    memory_sync_retry_note: str

    def emit(self) -> None:
        payload = {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "rank": self.rank,
            "world_size": self.world_size,
            "tp_size": self.tp_size,
            "case_name": self.case_name,
            "prompt_token_lengths": self.prompt_token_lengths,
            "requested_max_new_tokens": self.requested_max_new_tokens,
            "actual_output_tokens_per_request": self.actual_output_tokens_per_request,
            "output_texts": self.output_texts,
            "output_token_ids": self.output_token_ids,
            "batch_timeline": self.batch_timeline,
            "decode_metadata_summary": self.decode_metadata_summary,
            "baseline_available_tokens": self.baseline_available_tokens,
            "available_tokens_after_case": self.available_tokens_after_case,
            "free_pages_before_after": self.free_pages_before_after,
            "deferred_abort_uids": self.deferred_abort_uids,
            "cache_integrity_ok": self.cache_integrity_ok,
            "status": self.status,
            "failure_stage": self.failure_stage,
            "cold_start_attempt_id": self.cold_start_attempt_id,
            "memory_sync_retry_note": self.memory_sync_retry_note,
        }
        print(
            f"GATE4.14_JSONL rank={self.rank} {json.dumps(payload, ensure_ascii=False)}",
            flush=True,
        )


def _err(rank: int, msg: str) -> None:
    print(f"[gate4.14][rank={rank}] {msg}", flush=True)


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
    kind: str  # "static_b1" | "static_b2" | "dynamic_admission"
    prompts: List[str]
    max_new_tokens: List[int]


def _build_cases() -> List[Case]:
    short = "Paris is"
    equal_a = "The capital of France is"
    equal_b = "The capital of Italy is"
    long = "The largest planet in our solar system by mass and volume is"
    return [
        Case("A_b1_maxnew8", "static_b1", [short], [8]),
        Case("B_b1_maxnew16", "static_b1", [short], [16]),
        Case("C_b2_equal_maxnew8", "static_b2", [equal_a, equal_b], [8, 8]),
        Case("D_b2_ragged_maxnew8", "static_b2", [short, long], [8, 8]),
        Case("E_b2_mixed_kv_maxnew8", "static_b2", [short, long], [8, 8]),
        Case("F_dynamic_admission_b1_b2_b1", "dynamic_admission", [short, long], [8, 8]),
    ]


# ---------------------------------------------------------------------
# Case runners.
# ---------------------------------------------------------------------
def _run_static_case(
    llm,
    case: Case,
    hook_state: HookState,
    gen_id: int,
    sampling_params_cls,
) -> List[Dict[str, Any]]:
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
    llm.pending_requests = list(zip(case.prompts, sps))
    llm.status_map = {}
    llm.counter = 0

    hook_state.steps.clear()
    hook_state.active_gen_id = gen_id

    from minisgl.llm.llm import RequestAllFinished
    try:
        llm.run_forever()
    except RequestAllFinished:
        pass
    hook_state.active_gen_id = -1

    results = []
    for uid in sorted(llm.status_map.keys()):
        status = llm.status_map[uid]
        results.append({"uid": uid, "token_ids": list(status.output_ids)})
    return results


def _run_dynamic_admission_case(
    llm,
    case: Case,
    hook_state: HookState,
    adm_state: AdmissionState,
    gen_id: int,
    sampling_params_cls,
) -> List[Dict[str, Any]]:
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
    llm._b_revealed = False  # type: ignore[attr-defined]
    adm_state.ready_to_admit_b = False

    hook_state.steps.clear()
    hook_state.active_gen_id = gen_id

    from minisgl.llm.llm import RequestAllFinished
    try:
        llm.run_forever()
    except RequestAllFinished:
        pass
    hook_state.active_gen_id = -1

    results = []
    for uid in sorted(llm.status_map.keys()):
        status = llm.status_map[uid]
        results.append({"uid": uid, "token_ids": list(status.output_ids)})
    return results


# ---------------------------------------------------------------------
# Decode metadata summary extraction.
# ---------------------------------------------------------------------
def _decode_metadata_summary(
    steps: List[Tuple[int, List[int], List[int]]],
) -> Dict[str, Any]:
    """Summarise the hook step buffer for the verdict.

    Returns:
        prefill_step: first step (batch, qlens, kv_lens) if any.
        decode_step_count: number of steps where all qlens == 1.
        mixed_kv_decode_step_count: number of decode steps whose
            kv_lens contain unequal values.
        contains_ordered_1_2_1: True iff batch_size timeline contains
            ordered subsequence [1, 2, 1].
    """
    batch_timeline = [bs for (bs, _, _) in steps]
    decode_steps = [s for s in steps if all(q == 1 for q in s[1])]
    mixed_kv = sum(
        1
        for (_, _, kvs) in decode_steps
        if len(set(kvs)) > 1
    )

    def _has_1_2_1(seq: List[int]) -> bool:
        saw_1 = False
        saw_2_after_1 = False
        for x in seq:
            if x == 1 and not saw_1:
                saw_1 = True
            elif x == 2 and saw_1 and not saw_2_after_1:
                saw_2_after_1 = True
            elif x == 1 and saw_2_after_1:
                return True
        return False

    prefill = steps[0] if steps else None
    return {
        "step_count": len(steps),
        "batch_timeline": batch_timeline,
        "decode_step_count": len(decode_steps),
        "mixed_kv_decode_step_count": mixed_kv,
        "contains_ordered_1_2_1": _has_1_2_1(batch_timeline),
        "prefill_batch_size": prefill[0] if prefill else None,
        "prefill_query_lengths": prefill[1] if prefill else None,
        "prefill_kv_lengths": prefill[2] if prefill else None,
    }


# ---------------------------------------------------------------------
# Per-case validation.
# ---------------------------------------------------------------------
def _validate_case(
    case: Case,
    prompt_lengths: List[int],
    results: List[Dict[str, Any]],
    hook_summary: Dict[str, Any],
    allocator_after: Dict[str, Any],
    baseline_available: int,
    deferred_aborts: int,
) -> Tuple[str, Optional[str]]:
    """Return (status, failure_stage). status in {PASS, FAIL}."""
    actual_tokens = [len(r["token_ids"]) for r in results]

    # A / B: single request, output token count equals requested max_new
    if case.kind == "static_b1":
        if actual_tokens != case.max_new_tokens:
            return "FAIL", "output_token_count_mismatch"
    # C: B=2 equal-length, both [8, 8]
    elif case.name == "C_b2_equal_maxnew8":
        if prompt_lengths[0] != prompt_lengths[1]:
            return "FAIL", "prompt_lengths_not_equal"
        if actual_tokens != [8, 8]:
            return "FAIL", "output_token_count_mismatch"
    # D: B=2 ragged (unequal), [8, 8]
    elif case.name == "D_b2_ragged_maxnew8":
        if prompt_lengths[0] == prompt_lengths[1]:
            return "FAIL", "prompt_lengths_not_unequal"
        if actual_tokens != [8, 8]:
            return "FAIL", "output_token_count_mismatch"
    # E: mixed-KV decode evidence (>=1 decode step qlens==[1,1] and kv unequal)
    elif case.name == "E_b2_mixed_kv_maxnew8":
        if prompt_lengths[0] == prompt_lengths[1]:
            return "FAIL", "prompt_lengths_not_unequal"
        if actual_tokens != [8, 8]:
            return "FAIL", "output_token_count_mismatch"
        if hook_summary["mixed_kv_decode_step_count"] < 1:
            return "FAIL", "no_mixed_kv_decode_step"
    # F: dynamic admission ordered subsequence [1, 2, 1]
    elif case.name == "F_dynamic_admission_b1_b2_b1":
        if actual_tokens != [8, 8]:
            return "FAIL", "output_token_count_mismatch"
        if not hook_summary["contains_ordered_1_2_1"]:
            return "FAIL", "no_ordered_1_2_1_timeline"

    if allocator_after["available_tokens"] != baseline_available:
        return "FAIL", "available_tokens_regression"
    if deferred_aborts != 0:
        return "FAIL", "deferred_aborts_present"
    if not allocator_after["integrity_ok"]:
        return "FAIL", "cache_integrity_failed"
    return "PASS", None


# ---------------------------------------------------------------------
# Argument parsing.
# ---------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate 4.14 fixed-TP2 capability matrix (single model)",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--memory-ratio", type=float, default=0.85)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--max-running-req", type=int, default=4)
    parser.add_argument("--attention-backend", default="npu_fia")
    parser.add_argument("--cold-start-attempt-id", type=int, default=1)
    parser.add_argument("--memory-sync-retry-note", default="")
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
            "[gate4.14] LOCAL_RANK / RANK / WORLD_SIZE must be set by "
            "torchrun --nproc_per_node=2",
            file=sys.stderr,
            flush=True,
        )
        return 2

    rank = int(rank_env)
    world_size = int(world_size_env)
    os.environ.setdefault("MINISGL_DISTRIBUTED_ADDR", "env://")

    if world_size != 2:
        _err(rank, f"Gate 4.14 pins world_size=2; got {world_size}")
        if rank == 0:
            print("GATE4.14_MATRIX_RESULT=BLOCKED", flush=True)
        return 2

    model_name = args.model_name or os.path.basename(
        os.path.normpath(args.model_path)
    )

    llm = None
    hook_state = HookState()
    adm_state = AdmissionState()

    try:
        import torch
        from minisgl.core import SamplingParams
        from minisgl.distributed import DistributedInfo
        from minisgl.llm import LLM

        tp_info = DistributedInfo(rank=rank, size=world_size)

        _install_metadata_snapshot_hook(hook_state)

        class StaggeredLLM(LLM):
            def __init__(self_inner, *a, **kw):
                super().__init__(*a, **kw)
                self_inner._staged_b = None  # type: ignore[attr-defined]
                self_inner._b_revealed = False  # type: ignore[attr-defined]

            def offline_receive_msg(self_inner, blocking: bool = False):
                if (
                    not adm_state.ready_to_admit_b
                    and self_inner._staged_b is not None
                ):
                    for (bs, qlens, _) in hook_state.steps:
                        if bs == 1 and qlens == [1]:
                            adm_state.ready_to_admit_b = True
                            break
                if (
                    not self_inner._b_revealed
                    and adm_state.ready_to_admit_b
                    and self_inner._staged_b is not None
                ):
                    self_inner.pending_requests.append(self_inner._staged_b)
                    self_inner._staged_b = None
                    self_inner._b_revealed = True
                return super().offline_receive_msg(blocking=blocking)

        llm = StaggeredLLM(
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
                print("GATE4.14_MATRIX_RESULT=BLOCKED", flush=True)
            llm.shutdown()
            return 1

    except BaseException as exc:
        _err(
            rank,
            f"init/load failed: {type(exc).__name__}: {exc}\n"
            + "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )[-2000:],
        )
        if rank == 0:
            print("GATE4.14_MATRIX_RESULT=BLOCKED", flush=True)
        return 2

    cases = _build_cases()
    result_flag = "PASS"
    gen_id = 0

    for case in cases:
        try:
            prompt_lengths = [len(llm.tokenizer.encode(p)) for p in case.prompts]
        except BaseException as exc:
            _err(rank, f"tokenizer.encode raised on {case.name}: {exc!r}")
            result_flag = "PARTIAL"
            continue

        gen_id += 1
        try:
            if case.kind == "dynamic_admission":
                results = _run_dynamic_admission_case(
                    llm, case, hook_state, adm_state, gen_id, SamplingParams
                )
            else:
                results = _run_static_case(
                    llm, case, hook_state, gen_id, SamplingParams
                )
        except BaseException as exc:
            _err(
                rank,
                f"case {case.name} raised: {type(exc).__name__}: {exc}\n"
                + "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                )[-1500:],
            )
            # Emit a FAIL record so the matrix is complete.
            fail_rec = CaseRecord(
                model_name=model_name,
                model_path=args.model_path,
                rank=rank,
                world_size=world_size,
                tp_size=world_size,
                case_name=case.name,
                prompt_token_lengths=prompt_lengths,
                requested_max_new_tokens=case.max_new_tokens,
                actual_output_tokens_per_request=[],
                output_texts=[],
                output_token_ids=[],
                batch_timeline=None,
                decode_metadata_summary=None,
                baseline_available_tokens=baseline_available_tokens,
                available_tokens_after_case=-1,
                free_pages_before_after=[baseline_free_pages, -1],
                deferred_abort_uids=len(llm.deferred_abort_uids),
                cache_integrity_ok=False,
                status="FAIL",
                failure_stage=f"case_run_raised:{type(exc).__name__}",
                cold_start_attempt_id=args.cold_start_attempt_id,
                memory_sync_retry_note=args.memory_sync_retry_note,
            )
            fail_rec.emit()
            # A-D failure ⇒ BLOCKED (base functional), E/F ⇒ PARTIAL.
            if case.name.startswith(("A_", "B_", "C_", "D_")):
                result_flag = "BLOCKED"
            elif result_flag == "PASS":
                result_flag = "PARTIAL"
            continue

        hook_summary = _decode_metadata_summary(hook_state.steps)
        after = _snapshot(llm.cache_manager)
        deferred = len(llm.deferred_abort_uids)

        try:
            output_texts = [
                llm.tokenizer.decode(r["token_ids"]) for r in results
            ]
        except BaseException:
            output_texts = []

        status, failure_stage = _validate_case(
            case=case,
            prompt_lengths=prompt_lengths,
            results=results,
            hook_summary=hook_summary,
            allocator_after=after,
            baseline_available=baseline_available_tokens,
            deferred_aborts=deferred,
        )

        rec = CaseRecord(
            model_name=model_name,
            model_path=args.model_path,
            rank=rank,
            world_size=world_size,
            tp_size=world_size,
            case_name=case.name,
            prompt_token_lengths=prompt_lengths,
            requested_max_new_tokens=case.max_new_tokens,
            actual_output_tokens_per_request=[len(r["token_ids"]) for r in results],
            output_texts=output_texts,
            output_token_ids=[r["token_ids"] for r in results],
            batch_timeline=hook_summary["batch_timeline"],
            decode_metadata_summary=hook_summary,
            baseline_available_tokens=baseline_available_tokens,
            available_tokens_after_case=after["available_tokens"],
            free_pages_before_after=[baseline_free_pages, after["free_pages"]],
            deferred_abort_uids=deferred,
            cache_integrity_ok=after["integrity_ok"],
            status=status,
            failure_stage=failure_stage,
            cold_start_attempt_id=args.cold_start_attempt_id,
            memory_sync_retry_note=args.memory_sync_retry_note,
        )
        rec.emit()

        if status != "PASS":
            if case.name.startswith(("A_", "B_", "C_", "D_")):
                result_flag = "BLOCKED"
            elif result_flag == "PASS":
                result_flag = "PARTIAL"

    try:
        llm.shutdown()
    except BaseException as exc:
        _err(rank, f"shutdown raised: {exc!r}")
        if result_flag == "PASS":
            result_flag = "PARTIAL"

    if rank == 0:
        print(f"GATE4.14_MATRIX_RESULT={result_flag}", flush=True)
    return 0 if result_flag == "PASS" else (1 if result_flag == "PARTIAL" else 2)


def main() -> int:
    return _run_worker(_parse_args())


if __name__ == "__main__":
    sys.exit(main())
