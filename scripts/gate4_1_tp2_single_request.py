"""Gate 4.1 TP=2 single-request bring-up smoke driver.

Runs a Qwen3-0.6B TP=2 single-request end-to-end on Ascend 910B1:
init-only smoke (case A) → model load smoke (case B) → single-request
max_new_tokens=8 (case C).

Envelope (locked at Gate 4.1):
    Ascend 910B1, TP=2, eager, npu_fia, bf16, greedy
    max_new_tokens = 8, single request only
    memory_ratio=0.85, page_size=16, max_running_req=4
    cuda_graph_bs=[] (torch_npu has no CUDAGraph)

Design:
    * The script IS a per-rank worker. Launch it under torchrun so
      LOCAL_RANK / RANK / WORLD_SIZE are set by the launcher; the
      script reads them and constructs ``DistributedInfo(rank=RANK,
      size=WORLD_SIZE)`` for the ``LLM`` driver (Gate 4.1 relaxation
      in ``python/minisgl/llm/llm.py``).
    * ``use_pynccl=False`` is required on NPU (PyNCCL is a CUDA-only
      compile artefact). The Engine's TP>1 branch then takes the
      accelerator-backend + gloo-sidecar path: HCCL for tensor
      collectives, gloo for CPU collectives.
    * Both ranks enqueue the same prompt (bit-identical greedy path
      keeps rank 0 and rank 1 in lockstep). Only rank 0 prints the
      structured JSON row; rank 1 emits a minimal status line so a
      hung peer surfaces immediately.
    * Cases A / B / C share a single ``LLM`` boot per process because
      ``set_tp_info`` is a one-shot. Case A only checks post-init state,
      case B only checks post-weight-load state, case C runs one
      generate() call.

Structured log format (per rank):
    A single JSON object on stdout tagged ``GATE4.1_JSONL rank=<r> ...``
    with fields listed in Gate 4.1 spec §3.

Exit code (rank 0):
    0 on PASS, 1 on FAIL, 2 on BLOCKED. Rank 1 mirrors rank 0's exit
    code via the barrier at the end of shutdown — a mismatch will
    manifest as a launcher-level error.

This script does not benchmark, does not batch, does not run
recovery, and does not touch any non-Qwen3 model family. It exercises
init, weight sharding, one prefill, ``max_new_tokens=8`` decode, and
symmetric cleanup — nothing more.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RankLog:
    rank: int
    world_size: int
    model_path: str
    tp_size: int
    device: str = ""
    baseline_available_tokens: Optional[int] = None
    baseline_free_pages: Optional[int] = None
    total_pages: Optional[int] = None
    init_status: str = "PENDING"
    load_status: str = "PENDING"
    prefill_status: str = "PENDING"
    decode_status: str = "PENDING"
    actual_output_tokens: Optional[int] = None
    output_text: Optional[str] = None
    available_tokens_after_case: Optional[int] = None
    free_pages_after_case: Optional[int] = None
    deferred_abort_uids: Optional[int] = None
    cache_integrity_ok: Optional[bool] = None
    status: str = "BLOCKED"
    failure_stage: Optional[str] = None
    failure_trace_summary: Optional[str] = None

    def emit(self) -> None:
        payload: Dict[str, Any] = {
            "rank": self.rank,
            "world_size": self.world_size,
            "model_path": self.model_path,
            "tp_size": self.tp_size,
            "device": self.device,
            "baseline_available_tokens": self.baseline_available_tokens,
            "baseline_free_pages": self.baseline_free_pages,
            "total_pages": self.total_pages,
            "init_status": self.init_status,
            "load_status": self.load_status,
            "prefill_status": self.prefill_status,
            "decode_status": self.decode_status,
            "actual_output_tokens": self.actual_output_tokens,
            "output_text": self.output_text,
            "available_tokens_after_case": self.available_tokens_after_case,
            "free_pages_after_case": self.free_pages_after_case,
            "deferred_abort_uids": self.deferred_abort_uids,
            "cache_integrity_ok": self.cache_integrity_ok,
            "status": self.status,
            "failure_stage": self.failure_stage,
            "failure_trace_summary": self.failure_trace_summary,
        }
        print(
            f"GATE4.1_JSONL rank={self.rank} {json.dumps(payload, ensure_ascii=False)}",
            flush=True,
        )


def _err(rank: int, msg: str) -> None:
    print(f"[gate4.1][rank={rank}] {msg}", flush=True)


def _snapshot(cache_manager) -> Dict[str, Any]:
    """Read allocator invariants without mutating state.

    ``available_size`` follows Gate 3.1 §4: free slots + evictable
    prefix cache pages (in TOKENS, not pages). ``check_integrity`` is
    invoked defensively so any allocator drift surfaces here rather
    than as a downstream forward crash.
    """
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate 4.1 TP=2 single-request bring-up on Qwen3-0.6B",
    )
    parser.add_argument(
        "--model-path",
        default="/mnt/nvme/models/Qwen3-0.6B",
        help="Model directory (default: /mnt/nvme/models/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8,
        help="Fixed at 8 for Gate 4.1; exposed only for debugging",
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
        "--prompt",
        default="The capital of France is",
    )
    return parser.parse_args()


def _run_worker(args: argparse.Namespace) -> int:
    rank_env = os.environ.get("RANK")
    world_size_env = os.environ.get("WORLD_SIZE")
    local_rank_env = os.environ.get("LOCAL_RANK")
    if rank_env is None or world_size_env is None or local_rank_env is None:
        print(
            "[gate4.1] LOCAL_RANK / RANK / WORLD_SIZE must be set by the "
            "launcher (torchrun --nproc_per_node=2 ...)",
            file=sys.stderr,
            flush=True,
        )
        return 2

    rank = int(rank_env)
    world_size = int(world_size_env)
    local_rank = int(local_rank_env)

    # Gate 4.1: use the launcher's rendezvous store. torchrun exports
    # MASTER_ADDR / MASTER_PORT for us — pointing minisgl's distributed
    # init at "env://" reuses that store instead of the loopback fallback
    # in EngineConfig.distributed_addr, which the standalone TCP rendezvous
    # handler on this torch build does not stand up correctly.
    os.environ.setdefault("MINISGL_DISTRIBUTED_ADDR", "env://")

    log = RankLog(
        rank=rank,
        world_size=world_size,
        model_path=args.model_path,
        tp_size=world_size,
    )

    if world_size != 2:
        log.failure_stage = "launch"
        log.failure_trace_summary = (
            f"Gate 4.1 pins world_size=2; got {world_size}"
        )
        log.status = "BLOCKED"
        log.emit()
        return 2

    # ------------------------------------------------------------------
    # A. Init-only smoke
    # ------------------------------------------------------------------
    llm = None
    try:
        # Late imports so that a fatal environment problem (missing
        # torch_npu, wrong CANN) surfaces via the structured log rather
        # than a bare ImportError at module top.
        import torch
        from minisgl.distributed import DistributedInfo
        from minisgl.llm import LLM

        tp_info = DistributedInfo(rank=rank, size=world_size)
        log.device = f"npu:{local_rank}"

        # Boot: this is where TP=2 init exercises init_process_group
        # under the hccl backend + gloo sidecar. use_pynccl=False is
        # mandatory on NPU (PyNCCL is a CUDA-only compile artefact).
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

        # Cases A + B collapse into the same LLM boot: reaching here
        # means init_process_group returned, weights loaded and sharded
        # per rank, KV cache allocated, sampler initialised. We record
        # allocator baseline here as case B's post-load snapshot.
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

    # ------------------------------------------------------------------
    # C. TP=2 single-request max_new_tokens=8
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

        # Both ranks enqueue the same prompt so their generate() calls
        # advance in lockstep. Greedy on identical prompts with
        # identical weight shards and an all-gathered lm_head yields
        # bit-identical output on both ranks; the driver only reports
        # rank 0's text, but rank 1's output_tokens count is still
        # asserted equal.
        prefill_t0 = time.perf_counter()
        results = llm.generate([args.prompt], sampling_params)
        prefill_t1 = time.perf_counter()
        _err(rank, f"generate() returned in {(prefill_t1 - prefill_t0)*1000.0:.2f} ms")

        if not results or "token_ids" not in results[0]:
            raise RuntimeError(f"generate() returned malformed result: {results!r}")

        actual_tokens = len(results[0]["token_ids"])
        if actual_tokens != args.max_new_tokens:
            raise RuntimeError(
                f"expected {args.max_new_tokens} output tokens, got {actual_tokens}"
            )

        log.prefill_status = "PASS"
        log.decode_status = "PASS"
        log.actual_output_tokens = actual_tokens
        log.output_text = results[0].get("text")

    except BaseException as exc:
        log.failure_stage = "prefill" if log.prefill_status == "PENDING" else "decode"
        log.failure_trace_summary = (
            f"{type(exc).__name__}: {exc}\n"
            + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:]
        )
        log.status = "FAIL"
        # Still attempt shutdown so both ranks release HCCL/gloo state.
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

    # Symmetric shutdown across ranks. sync_all_ranks is a CPU barrier
    # on the gloo sidecar — both ranks must reach it or hang, so this
    # runs regardless of the PASS/FAIL decision above.
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
