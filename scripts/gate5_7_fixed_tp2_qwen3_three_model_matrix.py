"""Gate 5.7 fixed-TP2 Qwen3 three-model capability matrix driver
(single-model per invocation).

Runs the Ascend fixed-TP2 six-case functional matrix (A-F) against
ONE Qwen3 dense model per torchrun invocation. The outer shell
loop runs this driver once for each of the three Gate-4/Gate-5
covered models:

    /mnt/nvme/models/Qwen3-0.6B
    /mnt/nvme/models/Qwen3-1.7B
    /mnt/nvme/models/Qwen3-4B

Splitting per-model into separate processes is required because
``DistributedInfo`` / TP-group state is pinned at first ``LLM()``
init and rejects re-init in the same process
(``RuntimeError: TP info has been set``). Each invocation emits
per-case JSONL and one ``GATE5.7_MODEL_RESULT=<PASS|PARTIAL|BLOCKED>``
footer on rank 0 (containing the model name). The outer shell
aggregates footers into ``GATE5.7_MATRIX_RESULT=...``.

Cases (locked at Gate 5.7):
    A. B=1 single request, max_new_tokens=8
    B. B=1 single request, max_new_tokens=16
    C. B=2 equal-length, max_new_tokens=8
    D. B=2 ragged prefill (unequal-length), max_new_tokens=8
    E. B=2 mixed-KV decode evidence (unequal prefills), max_new_tokens=8
    F. dynamic admission B: 1 -> 2 -> 1

Envelope (locked at Gate 5.7):
    Ascend 910B1, TP=2, eager, npu_fia, bf16, greedy
    memory_ratio=0.85, page_size=16, max_running_req=4
    cuda_graph_bs=[] (torch_npu has no CUDAGraph)
    use_pynccl=False (HCCL + gloo sidecar)

Functional-only:
    * NO timing statistics collected
    * PASS / PARTIAL / BLOCKED per-case status derived from
      output/token/allocator/evidence invariants
    * A/B: output token count == requested max_new_tokens
    * C:   prompt lengths equal, output [8,8]
    * D:   prompt lengths unequal, output [8,8]
    * E:   >= 1 decode snapshot with query_lengths == [1,1] AND
           unequal kv_lengths
    * F:   batch_timeline contains ordered subsequence [1,2,1]
    * All cases: rank0/rank1 output_texts / output_token_ids match
                 by uid (verified via log post-processing across
                 the two per-rank JSONL streams), available_tokens
                 _after == baseline, deferred_abort_uids == 0,
                 cache_integrity_ok == true

This script does not tune, does not sweep, does not touch
``python/minisgl/``.
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
# Metadata snapshot hook.
# ---------------------------------------------------------------------
@dataclass
class StepSnapshot:
    step_id: int
    batch_size: int
    active_uids: List[int]
    query_lengths: List[int]
    kv_lengths: List[int]


@dataclass
class HookState:
    active_gen_id: int = -1
    steps: List[StepSnapshot] = field(default_factory=list)


def _install_metadata_snapshot_hook(state: HookState) -> None:
    """Wrap ``AscendFIABackend.prepare_metadata`` once per process to
    record ``(batch_size, active_uids, query_seq_lens, kv_seq_lens)``
    for every forward pass. Script-only monkey-patch -- runtime is
    not modified.
    """
    from minisgl.attention.ascend_fia import AscendFIABackend, FIAMetadata

    original = AscendFIABackend.prepare_metadata

    def patched(self, batch):  # type: ignore[no-redef]
        original(self, batch)
        m = batch.attn_metadata
        if isinstance(m, FIAMetadata) and state.active_gen_id >= 0:
            reqs = getattr(batch, "padded_reqs", None) or getattr(
                batch, "requests", None
            ) or []
            active_uids: List[int] = []
            for idx, r in enumerate(reqs):
                uid = getattr(r, "uid", None)
                if uid is None:
                    uid = getattr(r, "request_id", None)
                if uid is None:
                    uid = idx
                try:
                    active_uids.append(int(uid))
                except Exception:
                    active_uids.append(idx)
            state.steps.append(
                StepSnapshot(
                    step_id=len(state.steps),
                    batch_size=int(m.batch_size),
                    active_uids=active_uids,
                    query_lengths=list(m.query_seq_lens),
                    kv_lengths=list(m.kv_seq_lens),
                )
            )

    AscendFIABackend.prepare_metadata = patched  # type: ignore[assignment]


# ---------------------------------------------------------------------
# Case F: shared state for dynamic admission B: 1 -> 2 -> 1.
# ---------------------------------------------------------------------
@dataclass
class AdmissionState:
    ready_to_admit_b: bool = False
    a_uid: int = 0
    b_uid: int = 1


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
# Model definitions.
# ---------------------------------------------------------------------
@dataclass
class Model:
    name: str
    path: str


DEFAULT_MODELS: List[Model] = [
    Model("Qwen3-0.6B", "/mnt/nvme/models/Qwen3-0.6B"),
    Model("Qwen3-1.7B", "/mnt/nvme/models/Qwen3-1.7B"),
    Model("Qwen3-4B", "/mnt/nvme/models/Qwen3-4B"),
]


# ---------------------------------------------------------------------
# Utility helpers.
# ---------------------------------------------------------------------
def _err(rank: int, msg: str) -> None:
    print(f"[gate5.7][rank={rank}] {msg}", flush=True)


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
# Per-case runners.
# ---------------------------------------------------------------------
def _run_static_case(
    llm,
    case: Case,
    hook_state: HookState,
    gen_id: int,
    sampling_params_cls,
) -> Tuple[List[StepSnapshot], List[Dict[str, Any]]]:
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
    llm._staged_b = None  # type: ignore[attr-defined]
    llm._b_revealed = True  # type: ignore[attr-defined]

    hook_state.steps.clear()
    hook_state.active_gen_id = gen_id

    from minisgl.llm.llm import RequestAllFinished
    try:
        llm.run_forever()
    except RequestAllFinished:
        pass
    hook_state.active_gen_id = -1

    results = []
    tok = llm.tokenizer
    for uid in sorted(llm.status_map.keys()):
        status = llm.status_map[uid]
        ids = list(status.output_ids)
        try:
            text = tok.decode(ids)
        except Exception:
            text = ""
        results.append({"uid": int(uid), "token_ids": ids, "text": text})

    return list(hook_state.steps), results


def _run_dynamic_admission_case(
    llm,
    case: Case,
    hook_state: HookState,
    adm_state: AdmissionState,
    gen_id: int,
    sampling_params_cls,
) -> Tuple[List[StepSnapshot], List[Dict[str, Any]]]:
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
    adm_state.a_uid = 0
    adm_state.b_uid = 1

    hook_state.steps.clear()
    hook_state.active_gen_id = gen_id

    from minisgl.llm.llm import RequestAllFinished
    try:
        llm.run_forever()
    except RequestAllFinished:
        pass
    hook_state.active_gen_id = -1

    results = []
    tok = llm.tokenizer
    for uid in sorted(llm.status_map.keys()):
        status = llm.status_map[uid]
        ids = list(status.output_ids)
        try:
            text = tok.decode(ids)
        except Exception:
            text = ""
        results.append({"uid": int(uid), "token_ids": ids, "text": text})

    return list(hook_state.steps), results


# ---------------------------------------------------------------------
# Case-level acceptance predicates.
# ---------------------------------------------------------------------
def _timeline_contains_1_2_1(timeline: List[int]) -> bool:
    saw_one = False
    saw_two_after_one = False
    for bs in timeline:
        if bs == 1 and not saw_one:
            saw_one = True
        elif bs == 2 and saw_one and not saw_two_after_one:
            saw_two_after_one = True
        elif bs == 1 and saw_two_after_one:
            return True
    return False


def _mixed_kv_evidence(steps: List[StepSnapshot]) -> Tuple[int, List[Dict[str, Any]]]:
    count = 0
    payload = []
    for s in steps:
        if s.query_lengths == [1, 1] and len(s.kv_lengths) == 2 \
                and s.kv_lengths[0] != s.kv_lengths[1]:
            count += 1
            payload.append(
                {
                    "step_id": s.step_id,
                    "kv_lengths": s.kv_lengths,
                }
            )
    return count, payload


def _decode_metadata_summary(steps: List[StepSnapshot]) -> Dict[str, Any]:
    decode_snapshots = [
        {
            "step_id": s.step_id,
            "batch_size": s.batch_size,
            "active_uids": s.active_uids,
            "query_lengths": s.query_lengths,
            "kv_lengths": s.kv_lengths,
        }
        for s in steps
        if s.query_lengths and all(q == 1 for q in s.query_lengths)
    ]
    return {
        "total_step_count": len(steps),
        "decode_step_count": len(decode_snapshots),
        "decode_step_snapshots": decode_snapshots,
    }


# ---------------------------------------------------------------------
# Argument parsing.
# ---------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate 5.7 fixed-TP2 Qwen3 three-model capability "
                    "matrix (single-model per invocation)",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Absolute path to a Qwen3 dense model directory.",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Human-readable model name for JSONL / footer tagging "
             "(e.g. Qwen3-0.6B).",
    )
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
            "[gate5.7] LOCAL_RANK / RANK / WORLD_SIZE must be set by "
            "torchrun --nproc_per_node=2",
            file=sys.stderr,
            flush=True,
        )
        return 2

    rank = int(rank_env)
    world_size = int(world_size_env)
    os.environ.setdefault("MINISGL_DISTRIBUTED_ADDR", "env://")

    if world_size != 2:
        _err(rank, f"Gate 5.7 pins world_size=2; got {world_size}")
        if rank == 0:
            print(
                f"GATE5.7_MODEL_RESULT model={args.model_name} BLOCKED",
                flush=True,
            )
        return 2

    hook_state = HookState()
    adm_state = AdmissionState()

    try:
        import torch
        from minisgl.core import SamplingParams
        from minisgl.distributed import DistributedInfo
        from minisgl.llm import LLM
    except BaseException as exc:
        _err(
            rank,
            f"minisgl import failed: {type(exc).__name__}: {exc}",
        )
        if rank == 0:
            print(
                f"GATE5.7_MODEL_RESULT model={args.model_name} BLOCKED",
                flush=True,
            )
        return 2

    tp_info = DistributedInfo(rank=rank, size=world_size)
    _install_metadata_snapshot_hook(hook_state)

    # StaggeredLLM: stages request B (case F only), reveals it once
    # the hook observes A alone decoding. For static cases the driver
    # sets ``_staged_b = None`` and ``_b_revealed = True`` so the
    # override is a no-op.
    class StaggeredLLM(LLM):
        def __init__(self_inner, *a, **kw):
            super().__init__(*a, **kw)
            self_inner._staged_b = None  # type: ignore[attr-defined]
            self_inner._b_revealed = True  # type: ignore[attr-defined]

        def offline_receive_msg(self_inner, blocking: bool = False):
            if (
                not adm_state.ready_to_admit_b
                and self_inner._staged_b is not None
            ):
                for s in hook_state.steps:
                    if s.batch_size == 1 and s.query_lengths == [1]:
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

    model_name = args.model_name
    model_path = args.model_path
    cases = _build_cases()
    model_result = "PASS"
    gen_id = 0

    if not os.path.isdir(model_path):
        payload = {
            "rank": rank,
            "world_size": world_size,
            "tp_size": world_size,
            "model_name": model_name,
            "model_path": model_path,
            "case_name": "MODEL_PATH_CHECK",
            "status": "BLOCKED",
            "failure_stage": "model_path_missing",
            "cold_start_attempt_id": args.cold_start_attempt_id,
            "memory_sync_retry_note": args.memory_sync_retry_note,
        }
        print(
            f"GATE5.7_JSONL rank={rank} {json.dumps(payload, ensure_ascii=False)}",
            flush=True,
        )
        if rank == 0:
            print(
                f"GATE5.7_MODEL_RESULT model={model_name} BLOCKED",
                flush=True,
            )
        return 2

    llm = None
    try:
        llm = StaggeredLLM(
            model_path=model_path,
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
            _err(rank, f"{model_name} post-load integrity failed")
            payload = {
                "rank": rank,
                "world_size": world_size,
                "tp_size": world_size,
                "model_name": model_name,
                "model_path": model_path,
                "case_name": "MODEL_LOAD",
                "status": "BLOCKED",
                "failure_stage": "post_load_integrity",
                "baseline_available_tokens": baseline_available_tokens,
                "baseline_free_pages": baseline_free_pages,
                "cold_start_attempt_id": args.cold_start_attempt_id,
                "memory_sync_retry_note": args.memory_sync_retry_note,
            }
            print(
                f"GATE5.7_JSONL rank={rank} {json.dumps(payload, ensure_ascii=False)}",
                flush=True,
            )
            try:
                llm.shutdown()
            except Exception:
                pass
            if rank == 0:
                print(
                    f"GATE5.7_MODEL_RESULT model={model_name} BLOCKED",
                    flush=True,
                )
            return 2
    except BaseException as exc:
        _err(
            rank,
            f"{model_name} init/load raised: {type(exc).__name__}: {exc}\n"
            + "".join(
                traceback.format_exception(
                    type(exc), exc, exc.__traceback__
                )
            )[-2000:],
        )
        payload = {
            "rank": rank,
            "world_size": world_size,
            "tp_size": world_size,
            "model_name": model_name,
            "model_path": model_path,
            "case_name": "MODEL_LOAD",
            "status": "BLOCKED",
            "failure_stage": "init_or_load",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "cold_start_attempt_id": args.cold_start_attempt_id,
            "memory_sync_retry_note": args.memory_sync_retry_note,
        }
        print(
            f"GATE5.7_JSONL rank={rank} {json.dumps(payload, ensure_ascii=False)}",
            flush=True,
        )
        if rank == 0:
            print(
                f"GATE5.7_MODEL_RESULT model={model_name} BLOCKED",
                flush=True,
            )
        return 2

    # -- Run cases A..F on this model --
    for case in cases:
        gen_id += 1
        case_status = "PASS"
        failure_stage: Optional[str] = None
        failure_reason: Optional[str] = None
        steps: List[StepSnapshot] = []
        results: List[Dict[str, Any]] = []

        try:
            prompt_lengths = [
                len(llm.tokenizer.encode(p)) for p in case.prompts
            ]
            if case.kind == "dynamic_admission":
                steps, results = _run_dynamic_admission_case(
                    llm, case, hook_state, adm_state, gen_id, SamplingParams
                )
            else:
                steps, results = _run_static_case(
                    llm, case, hook_state, gen_id, SamplingParams
                )
        except BaseException as exc:
            case_status = "FAIL"
            failure_stage = "run_forever"
            failure_reason = f"{type(exc).__name__}: {exc}"
            prompt_lengths = []

        after = _snapshot(llm.cache_manager)

        actual_tokens = [len(r["token_ids"]) for r in results]
        output_texts = [r["text"] for r in results]
        output_token_ids = [r["token_ids"] for r in results]

        batch_timeline = [s.batch_size for s in steps]
        decode_summary = _decode_metadata_summary(steps)
        mixed_kv_count, mixed_kv_examples = _mixed_kv_evidence(steps)
        timeline_ok = _timeline_contains_1_2_1(batch_timeline)

        allocator_ok = (
            after["available_tokens"] == baseline_available_tokens
            and len(llm.deferred_abort_uids) == 0
            and after["integrity_ok"]
        )

        # Case-specific acceptance.
        if case_status == "PASS":
            if case.name == "A_b1_maxnew8":
                if actual_tokens != [8]:
                    case_status = "FAIL"
                    failure_stage = "token_count"
                    failure_reason = (
                        f"expected [8], got {actual_tokens}"
                    )
            elif case.name == "B_b1_maxnew16":
                if actual_tokens != [16]:
                    case_status = "FAIL"
                    failure_stage = "token_count"
                    failure_reason = (
                        f"expected [16], got {actual_tokens}"
                    )
            elif case.name == "C_b2_equal_maxnew8":
                if (
                    len(prompt_lengths) != 2
                    or prompt_lengths[0] != prompt_lengths[1]
                ):
                    case_status = "FAIL"
                    failure_stage = "prompt_lengths"
                    failure_reason = (
                        f"C requires equal prompt lengths; got {prompt_lengths}"
                    )
                elif actual_tokens != [8, 8]:
                    case_status = "FAIL"
                    failure_stage = "token_count"
                    failure_reason = (
                        f"expected [8,8], got {actual_tokens}"
                    )
            elif case.name == "D_b2_ragged_maxnew8":
                if (
                    len(prompt_lengths) != 2
                    or prompt_lengths[0] == prompt_lengths[1]
                ):
                    case_status = "FAIL"
                    failure_stage = "prompt_lengths"
                    failure_reason = (
                        f"D requires unequal prompt lengths; got {prompt_lengths}"
                    )
                elif actual_tokens != [8, 8]:
                    case_status = "FAIL"
                    failure_stage = "token_count"
                    failure_reason = (
                        f"expected [8,8], got {actual_tokens}"
                    )
            elif case.name == "E_b2_mixed_kv_maxnew8":
                if actual_tokens != [8, 8]:
                    case_status = "FAIL"
                    failure_stage = "token_count"
                    failure_reason = (
                        f"expected [8,8], got {actual_tokens}"
                    )
                elif mixed_kv_count < 1:
                    case_status = "FAIL"
                    failure_stage = "mixed_kv_evidence"
                    failure_reason = (
                        "no decode step with query_lengths=[1,1] "
                        "and unequal kv_lengths"
                    )
            elif case.name == "F_dynamic_admission_b1_b2_b1":
                if actual_tokens != [8, 8]:
                    case_status = "FAIL"
                    failure_stage = "token_count"
                    failure_reason = (
                        f"expected [8,8], got {actual_tokens}"
                    )
                elif not timeline_ok:
                    case_status = "FAIL"
                    failure_stage = "batch_timeline"
                    failure_reason = (
                        f"batch_timeline lacks 1->2->1 subsequence: "
                        f"{batch_timeline}"
                    )

        # Allocator invariant check applies to every case.
        if case_status == "PASS" and not allocator_ok:
            case_status = "FAIL"
            failure_stage = "allocator_invariant"
            failure_reason = (
                f"avail={after['available_tokens']} baseline="
                f"{baseline_available_tokens} deferred="
                f"{len(llm.deferred_abort_uids)} integrity="
                f"{after['integrity_ok']}"
            )

        payload = {
            "rank": rank,
            "world_size": world_size,
            "tp_size": world_size,
            "model_name": model_name,
            "model_path": model_path,
            "case_name": case.name,
            "case_kind": case.kind,
            "prompt_token_lengths": prompt_lengths,
            "requested_max_new_tokens": case.max_new_tokens,
            "actual_output_tokens_per_request": actual_tokens,
            "output_texts": output_texts,
            "output_token_ids": output_token_ids,
            "batch_timeline": batch_timeline,
            "decode_metadata_summary": decode_summary,
            "mixed_kv_decode_step_count": mixed_kv_count,
            "mixed_kv_examples": mixed_kv_examples,
            "timeline_contains_1_2_1": timeline_ok,
            "baseline_available_tokens": baseline_available_tokens,
            "baseline_free_pages": baseline_free_pages,
            "available_tokens_after_case": after["available_tokens"],
            "free_pages_before_after": [
                baseline_free_pages,
                after["free_pages"],
            ],
            "deferred_abort_uids": len(llm.deferred_abort_uids),
            "cache_integrity_ok": after["integrity_ok"],
            "status": case_status,
            "failure_stage": failure_stage,
            "failure_reason": failure_reason,
            "cold_start_attempt_id": args.cold_start_attempt_id,
            "memory_sync_retry_note": args.memory_sync_retry_note,
        }
        print(
            f"GATE5.7_JSONL rank={rank} {json.dumps(payload, ensure_ascii=False)}",
            flush=True,
        )

        if case_status != "PASS":
            if case.name.startswith(("A_", "B_", "C_", "D_")):
                model_result = "BLOCKED"
            else:
                if model_result == "PASS":
                    model_result = "PARTIAL"

    try:
        llm.shutdown()
    except BaseException as exc:
        _err(rank, f"{model_name} shutdown raised: {exc!r}")
        if model_result == "PASS":
            model_result = "PARTIAL"

    if rank == 0:
        print(
            f"GATE5.7_MODEL_RESULT model={model_name} {model_result}",
            flush=True,
        )
    return 0 if model_result == "PASS" else (1 if model_result == "PARTIAL" else 2)


def main() -> int:
    return _run_worker(_parse_args())


if __name__ == "__main__":
    sys.exit(main())
