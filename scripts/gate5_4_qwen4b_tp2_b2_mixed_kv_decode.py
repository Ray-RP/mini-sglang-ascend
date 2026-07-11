"""Gate 5.4 Qwen3-4B TP=2 B=2 mixed-KV decode explicit evidence driver.

Runs a Qwen3-4B TP=2 B=2 unequal-length end-to-end on 2 x Ascend
910B1 and records per-decode-step FIA metadata to prove that the
runtime executes pure-decode steps with per-uid unequal KV lengths:
    A. TP=2 init
    B. Qwen3-4B TP=2 model load
    C. B=2 unequal-length prefill + decode
    D. mixed-KV decode metadata evidence

Envelope (locked at Gate 5.4):
    Ascend 910B1 x 2, TP=2, eager, npu_fia, bf16, greedy
    Qwen3-4B (/mnt/nvme/models/Qwen3-4B)
    max_new_tokens = 8, batch_size = 2, unequal-length prompts
    memory_ratio=0.85, page_size=16, max_running_req=4
    cuda_graph_bs=[] (torch_npu has no CUDAGraph)
    MINISGL_DISTRIBUTED_ADDR=env://, use_pynccl=False

Mixed-KV decode evidence:
    ``AscendFIABackend.prepare_metadata(batch)`` runs exactly once
    per forward pass to build ``FIAMetadata`` for the batch. The
    driver wraps that method (script-local monkey-patch) to record
    per-step ``batch_size``, ``query_seq_lens``, and ``kv_seq_lens``.
    Pure-decode steps are those with ``query_lengths == [1, 1]``;
    a mixed-KV decode step additionally has
    ``kv_lengths[0] != kv_lengths[1]``. Gate 5.4 acceptance requires
    at least one such step in the generate() window. This is the
    same script-only hook pattern used at Gates 4.4 / 4.11 / 4.14;
    the runtime source under ``python/minisgl/attention/ascend_fia.py``
    is NOT modified.

Structured log format (per rank):
    A single JSON object on stdout tagged ``GATE5.4_JSONL rank=<r> ...``
    with the fields listed in Gate 5.4 spec section 4.

Footer (rank 0 only):
    ``GATE5.4_RESULT=<PASS|PARTIAL|BLOCKED>``

Exit code (rank 0):
    0 on PASS, 1 on FAIL, 2 on BLOCKED. Rank 1 mirrors via the CPU
    barrier in Scheduler.shutdown.

This driver does not do dynamic admission, does not do B > 2, does
not touch TP > 2, does not do timing, does not benchmark, does not
compare against SGLang / vLLM / TGI, does not modify
python/minisgl/, does not modify tests.
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
    query_lengths: List[int]
    kv_lengths: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "batch_size": self.batch_size,
            "query_lengths": list(self.query_lengths),
            "kv_lengths": list(self.kv_lengths),
        }


@dataclass
class RankLog:
    rank: int
    world_size: int
    tp_size: int
    model_path: str
    model_name: str
    model_exists: Optional[bool] = None
    device: str = ""
    prompts: List[str] = field(default_factory=list)
    prompt_token_lengths: List[int] = field(default_factory=list)
    batch_size: int = 2
    baseline_available_tokens: Optional[int] = None
    baseline_free_pages: Optional[int] = None
    total_pages: Optional[int] = None
    init_status: str = "PENDING"
    load_status: str = "PENDING"
    prefill_status: str = "PENDING"
    decode_status: str = "PENDING"
    mixed_kv_status: str = "PENDING"
    actual_output_tokens_per_request: Optional[List[int]] = None
    output_texts: Optional[List[str]] = None
    output_token_ids: Optional[List[List[int]]] = None
    all_step_snapshots: Optional[List[Dict[str, Any]]] = None
    decode_step_snapshots: Optional[List[Dict[str, Any]]] = None
    mixed_kv_decode_step_count: Optional[int] = None
    available_tokens_after_case: Optional[int] = None
    free_pages_after_case: Optional[int] = None
    free_pages_before_after: Optional[List[int]] = None
    deferred_abort_uids: Optional[int] = None
    cache_integrity_ok: Optional[bool] = None
    generate_ms: Optional[float] = None
    status: str = "BLOCKED"
    failure_stage: Optional[str] = None
    failure_reason: Optional[str] = None
    cold_start_attempt_id: int = 1
    memory_sync_retry_note: str = ""

    def emit(self) -> None:
        payload: Dict[str, Any] = {
            "rank": self.rank,
            "world_size": self.world_size,
            "tp_size": self.tp_size,
            "model_name": self.model_name,
            "model_path": self.model_path,
            "model_exists": self.model_exists,
            "device": self.device,
            "prompts": self.prompts,
            "prompt_token_lengths": self.prompt_token_lengths,
            "batch_size": self.batch_size,
            "baseline_available_tokens": self.baseline_available_tokens,
            "baseline_free_pages": self.baseline_free_pages,
            "total_pages": self.total_pages,
            "init_status": self.init_status,
            "load_status": self.load_status,
            "prefill_status": self.prefill_status,
            "decode_status": self.decode_status,
            "mixed_kv_status": self.mixed_kv_status,
            "actual_output_tokens_per_request": self.actual_output_tokens_per_request,
            "output_texts": self.output_texts,
            "output_token_ids": self.output_token_ids,
            "all_step_snapshots": self.all_step_snapshots,
            "decode_step_snapshots": self.decode_step_snapshots,
            "mixed_kv_decode_step_count": self.mixed_kv_decode_step_count,
            "available_tokens_after_case": self.available_tokens_after_case,
            "free_pages_after_case": self.free_pages_after_case,
            "free_pages_before_after": self.free_pages_before_after,
            "deferred_abort_uids": self.deferred_abort_uids,
            "cache_integrity_ok": self.cache_integrity_ok,
            "generate_ms": self.generate_ms,
            "status": self.status,
            "failure_stage": self.failure_stage,
            "failure_reason": self.failure_reason,
            "cold_start_attempt_id": self.cold_start_attempt_id,
            "memory_sync_retry_note": self.memory_sync_retry_note,
        }
        print(
            f"GATE5.4_JSONL rank={self.rank} {json.dumps(payload, ensure_ascii=False)}",
            flush=True,
        )


def _err(rank: int, msg: str) -> None:
    print(f"[gate5.4][rank={rank}] {msg}", flush=True)


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


def _install_metadata_snapshot_hook(snapshots: List[StepSnapshot]) -> None:
    """Wrap ``AscendFIABackend.prepare_metadata`` to record per-step shape.

    Script-only monkey-patch — the runtime source under
    ``python/minisgl/attention/ascend_fia.py`` is not modified. Same
    pattern as Gates 4.4 / 4.11 / 4.14, now on Qwen3-4B.
    """
    from minisgl.attention.ascend_fia import AscendFIABackend, FIAMetadata

    original = AscendFIABackend.prepare_metadata

    def patched(self, batch):  # type: ignore[no-redef]
        original(self, batch)
        m = batch.attn_metadata
        if isinstance(m, FIAMetadata):
            snapshots.append(
                StepSnapshot(
                    step_id=len(snapshots),
                    batch_size=int(m.batch_size),
                    query_lengths=list(m.query_seq_lens),
                    kv_lengths=list(m.kv_seq_lens),
                )
            )

    AscendFIABackend.prepare_metadata = patched  # type: ignore[assignment]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate 5.4 Qwen3-4B TP=2 B=2 mixed-KV decode explicit evidence",
    )
    parser.add_argument(
        "--model-path",
        default="/mnt/nvme/models/Qwen3-4B",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen3-4B",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8,
        help="Fixed at 8 for Gate 5.4; exposed only for debugging",
    )
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
        "--prompt-short",
        default="Paris is",
        help=(
            "Short B=2 prompt; must tokenize to fewer tokens than "
            "--prompt-long on Qwen3-4B so kv_seq_lens diverge on "
            "every decode step."
        ),
    )
    parser.add_argument(
        "--prompt-long",
        default="The largest planet in our solar system by mass and volume is",
        help=(
            "Long B=2 prompt; must tokenize to more tokens than "
            "--prompt-short."
        ),
    )
    parser.add_argument(
        "--cold-start-attempt-id",
        type=int,
        default=1,
        help="Outer-shell attempt index threaded in for the JSONL record.",
    )
    parser.add_argument(
        "--memory-sync-retry-note",
        default="",
        help="Free-form note carrying the reason for retry attempts, if any.",
    )
    return parser.parse_args()


def _run_worker(args: argparse.Namespace) -> int:
    rank_env = os.environ.get("RANK")
    world_size_env = os.environ.get("WORLD_SIZE")
    local_rank_env = os.environ.get("LOCAL_RANK")
    if rank_env is None or world_size_env is None or local_rank_env is None:
        print(
            "[gate5.4] LOCAL_RANK / RANK / WORLD_SIZE must be set by the "
            "launcher (torchrun --nproc_per_node=2 ...)",
            file=sys.stderr,
            flush=True,
        )
        return 2

    rank = int(rank_env)
    world_size = int(world_size_env)
    local_rank = int(local_rank_env)

    os.environ.setdefault("MINISGL_DISTRIBUTED_ADDR", "env://")

    log = RankLog(
        rank=rank,
        world_size=world_size,
        tp_size=world_size,
        model_path=args.model_path,
        model_name=args.model_name,
        prompts=[args.prompt_short, args.prompt_long],
        cold_start_attempt_id=args.cold_start_attempt_id,
        memory_sync_retry_note=args.memory_sync_retry_note,
    )
    log.model_exists = os.path.isdir(args.model_path)

    if world_size != 2:
        log.failure_stage = "launch"
        log.failure_reason = f"Gate 5.4 pins world_size=2; got {world_size}"
        log.status = "BLOCKED"
        log.emit()
        if rank == 0:
            print("GATE5.4_RESULT=BLOCKED", flush=True)
        return 2

    if not log.model_exists:
        log.failure_stage = "model_path"
        log.failure_reason = f"model_path does not exist: {args.model_path}"
        log.status = "BLOCKED"
        log.emit()
        if rank == 0:
            print("GATE5.4_RESULT=BLOCKED", flush=True)
        return 2

    # ------------------------------------------------------------------
    # A. TP=2 init + B. Qwen3-4B TP=2 model load (share one LLM boot)
    # ------------------------------------------------------------------
    llm = None
    snapshots: List[StepSnapshot] = []
    try:
        import torch
        from minisgl.distributed import DistributedInfo
        from minisgl.llm import LLM

        tp_info = DistributedInfo(rank=rank, size=world_size)
        log.device = f"npu:{local_rank}"

        _install_metadata_snapshot_hook(snapshots)

        llm = LLM(
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
            log.failure_reason = "check_integrity() raised after weight load"
            log.emit()
            if rank == 0:
                print("GATE5.4_RESULT=BLOCKED", flush=True)
            return 1

    except BaseException as exc:
        log.failure_stage = "init" if log.init_status == "PENDING" else "load"
        log.failure_reason = (
            f"{type(exc).__name__}: {exc}\n"
            + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:]
        )
        log.status = "BLOCKED" if log.failure_stage == "init" else "FAIL"
        log.emit()
        if rank == 0:
            print("GATE5.4_RESULT=BLOCKED", flush=True)
        return 2 if log.status == "BLOCKED" else 1

    # Discard any pre-generate snapshots so the mixed-KV evidence only
    # reflects the driver's own B=2 batch.
    snapshots_pre_generate = len(snapshots)

    # ------------------------------------------------------------------
    # C + D. TP=2 B=2 unequal-length generate() + decode-step snapshots
    # ------------------------------------------------------------------
    try:
        from minisgl.core import SamplingParams

        sampling_params = SamplingParams(
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            max_tokens=args.max_new_tokens,
            ignore_eos=True,
        )

        prompts = [args.prompt_short, args.prompt_long]
        tokenized_lengths = [len(llm.tokenizer.encode(p)) for p in prompts]
        log.prompt_token_lengths = tokenized_lengths
        log.batch_size = len(prompts)
        if tokenized_lengths[0] == tokenized_lengths[1]:
            raise RuntimeError(
                f"Gate 5.4 requires unequal-length prompts; got "
                f"{tokenized_lengths} for prompts={prompts!r}"
            )

        free_pages_before = len(llm.cache_manager.free_slots)

        t0 = time.perf_counter()
        results = llm.generate(prompts, sampling_params)
        t1 = time.perf_counter()
        log.generate_ms = (t1 - t0) * 1000.0
        _err(rank, f"generate() returned in {log.generate_ms:.2f} ms")

        if not isinstance(results, list) or len(results) != len(prompts):
            raise RuntimeError(f"generate() returned malformed result: {results!r}")

        actual_tokens_per_req = [len(r["token_ids"]) for r in results]
        if any(n != args.max_new_tokens for n in actual_tokens_per_req):
            raise RuntimeError(
                f"expected {args.max_new_tokens} tokens per request, "
                f"got {actual_tokens_per_req}"
            )

        log.actual_output_tokens_per_request = actual_tokens_per_req
        log.output_texts = [r.get("text") for r in results]
        log.output_token_ids = [list(r["token_ids"]) for r in results]
        log.prefill_status = "PASS"
        log.decode_status = "PASS"

        free_pages_after = len(llm.cache_manager.free_slots)
        log.free_pages_before_after = [free_pages_before, free_pages_after]

        # Slice snapshots that belong to this generate() call.
        generate_snapshots = snapshots[snapshots_pre_generate:]
        for i, s in enumerate(generate_snapshots):
            s.step_id = i
        log.all_step_snapshots = [s.to_dict() for s in generate_snapshots]

        # Mixed-KV decode evidence: steps where query_lengths == [1, 1]
        # (pure decode over both uids) AND kv_lengths differ per uid.
        decode_snaps = [
            s for s in generate_snapshots
            if s.batch_size == 2 and s.query_lengths == [1, 1]
        ]
        mixed_kv_snaps = [
            s for s in decode_snaps if s.kv_lengths[0] != s.kv_lengths[1]
        ]
        log.decode_step_snapshots = [s.to_dict() for s in decode_snaps]
        log.mixed_kv_decode_step_count = len(mixed_kv_snaps)

        if len(mixed_kv_snaps) == 0:
            log.mixed_kv_status = "FAIL"
            raise RuntimeError(
                "Gate 5.4 requires >=1 decode step with query_lengths=[1,1] "
                "and kv_lengths[0] != kv_lengths[1]; captured "
                f"{len(decode_snaps)} decode steps but none show unequal KV. "
                f"decode_step_snapshots={log.decode_step_snapshots!r}"
            )
        log.mixed_kv_status = "PASS"

    except BaseException as exc:
        if log.prefill_status == "PENDING":
            log.failure_stage = "prefill"
        elif log.decode_status == "PENDING":
            log.failure_stage = "decode"
        elif log.mixed_kv_status != "PASS":
            log.failure_stage = "mixed_kv_evidence"
        else:
            log.failure_stage = "post_generate"
        log.failure_reason = (
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
        if rank == 0:
            print("GATE5.4_RESULT=PARTIAL", flush=True)
        return 1

    # ------------------------------------------------------------------
    # Post-case allocator invariant + symmetric shutdown
    # ------------------------------------------------------------------
    try:
        after = _snapshot(llm.cache_manager)
        log.available_tokens_after_case = after["available_tokens"]
        log.free_pages_after_case = after["free_pages"]
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
            log.failure_reason = (
                f"available_tokens_after={after['available_tokens']!r} "
                f"vs baseline={log.baseline_available_tokens!r}; "
                f"deferred_abort_uids={log.deferred_abort_uids!r}; "
                f"cache_integrity_ok={after['integrity_ok']!r}"
            )
    except BaseException as exc:
        log.failure_stage = "post_case_invariant"
        log.failure_reason = (
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
            log.failure_reason = f"{type(exc).__name__}: {exc}"

    log.emit()

    if rank == 0:
        if log.status == "PASS":
            print("GATE5.4_RESULT=PASS", flush=True)
        elif log.status == "FAIL":
            print("GATE5.4_RESULT=PARTIAL", flush=True)
        else:
            print("GATE5.4_RESULT=BLOCKED", flush=True)

    return 0 if log.status == "PASS" else 1


def main() -> int:
    args = _parse_args()
    return _run_worker(args)


if __name__ == "__main__":
    sys.exit(main())
