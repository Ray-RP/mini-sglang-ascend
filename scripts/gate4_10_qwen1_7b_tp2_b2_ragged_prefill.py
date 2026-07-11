"""Gate 4.10 Qwen3-1.7B TP=2 B=2 ragged prefill smoke driver.

Runs a Qwen3-1.7B TP=2 B=2 ragged-prefill (unequal-length prompts)
end-to-end on 2 x Ascend 910B1:
    A. TP=2 init
    B. Qwen3-1.7B TP=2 model load
    C. B=2 ragged prefill + decode, max_new_tokens=8

Envelope (locked at Gate 4.10):
    Ascend 910B1, TP=2, eager, npu_fia, bf16, greedy
    Qwen3-1.7B (/mnt/nvme/models/Qwen3-1.7B)
    max_new_tokens = 8, batch_size = 2, unequal-length prompts
    memory_ratio=0.85, page_size=16, max_running_req=4
    cuda_graph_bs=[] (torch_npu has no CUDAGraph)

FIA branch selection:
    Two prompts with DIFFERENT tokenized lengths. Both uids arrive
    with cached_len == 0 (fresh boot, no radix prefix, single
    generate() call per process). This drives the FIA "ragged
    prefill with cached_len == 0" branch — the same branch proven
    on Qwen3-0.6B at Gate 4.3, now on Qwen3-1.7B. The Gate 2.2f
    documented "ragged + non-zero cached_len + extend_len > 1"
    unsupported combination is deliberately NOT exercised.

Design:
    * Same per-rank worker shape as Gate 4.1 / 4.3 / 4.8 / 4.9.
      torchrun x2, each rank reads LOCAL_RANK / RANK / WORLD_SIZE,
      use_pynccl=False.
    * Both ranks enqueue the SAME two prompts in the SAME order,
      chosen so their tokenized lengths on Qwen3-1.7B are unequal.
      The driver logs each prompt's tokenized length and asserts
      inequality up-front so the run cannot silently regress into
      an equal-length batch.
    * Greedy on all-gathered logits yields bit-identical per-uid
      output tokens per rank.
    * Post-case allocator invariants checked per rank.
    * ``cold_start_attempt_id`` is threaded through the JSONL to
      reflect the launch's attempt-in-a-shell-loop index; the
      driver itself does NOT retry — the outer shell loop is the
      Gate 4.10-authorised recovery for the transient
      ``_sync_get_memory()`` imbalance documented at Gate 4.9.

Structured log format (per rank):
    A single JSON object on stdout tagged ``GATE4.10_JSONL rank=<r> ...``
    with the fields listed in Gate 4.10 spec section 4.

Exit code (rank 0):
    0 on PASS, 1 on FAIL, 2 on BLOCKED. Rank 1 mirrors via the CPU
    barrier in Scheduler.shutdown.

This script does not do mixed-KV decode explicit evidence, does not
do dynamic admission, does not do B > 2, does not do timing, does
not touch TP=4, does not benchmark, does not modify python/minisgl/,
does not modify tests.
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
class RankLog:
    rank: int
    world_size: int
    tp_size: int
    model_path: str
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
    actual_output_tokens_per_request: Optional[List[int]] = None
    output_texts: Optional[List[str]] = None
    output_token_ids: Optional[List[List[int]]] = None
    available_tokens_after_case: Optional[int] = None
    free_pages_after_case: Optional[int] = None
    deferred_abort_uids: Optional[int] = None
    cache_integrity_ok: Optional[bool] = None
    generate_ms: Optional[float] = None
    cold_start_attempt_id: Optional[int] = None
    memory_sync_retry_note: Optional[str] = None
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
            "actual_output_tokens_per_request": self.actual_output_tokens_per_request,
            "output_texts": self.output_texts,
            "output_token_ids": self.output_token_ids,
            "available_tokens_after_case": self.available_tokens_after_case,
            "free_pages_after_case": self.free_pages_after_case,
            "deferred_abort_uids": self.deferred_abort_uids,
            "cache_integrity_ok": self.cache_integrity_ok,
            "generate_ms": self.generate_ms,
            "cold_start_attempt_id": self.cold_start_attempt_id,
            "memory_sync_retry_note": self.memory_sync_retry_note,
            "status": self.status,
            "failure_stage": self.failure_stage,
            "failure_trace_summary": self.failure_trace_summary,
        }
        print(
            f"GATE4.10_JSONL rank={self.rank} {json.dumps(payload, ensure_ascii=False)}",
            flush=True,
        )


def _err(rank: int, msg: str) -> None:
    print(f"[gate4.10][rank={rank}] {msg}", flush=True)


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate 4.10 Qwen3-1.7B TP=2 B=2 ragged prefill",
    )
    parser.add_argument(
        "--model-path",
        default="/mnt/nvme/models/Qwen3-1.7B",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8,
        help="Fixed at 8 for Gate 4.10; exposed only for debugging",
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
            "--prompt-long on Qwen3-1.7B."
        ),
    )
    parser.add_argument(
        "--prompt-long",
        default=(
            "The largest planet in our solar system by mass and volume is"
        ),
        help=(
            "Long B=2 prompt; must tokenize to more tokens than "
            "--prompt-short so the batch stays on the FIA ragged "
            "(cached_len == 0) branch."
        ),
    )
    parser.add_argument(
        "--cold-start-attempt-id",
        type=int,
        default=1,
        help=(
            "Externally-supplied attempt-in-a-shell-loop index. The "
            "driver itself does NOT retry; the outer shell loop "
            "increments this when re-launching after a transient "
            "_sync_get_memory() imbalance. Recorded in the JSONL "
            "for the verdict."
        ),
    )
    parser.add_argument(
        "--memory-sync-retry-note",
        default=None,
        help=(
            "Optional free-form note describing what the outer "
            "shell loop observed on prior attempts (e.g., 'attempt 1 "
            "hit min=X,max=Y imbalance'). Recorded in the JSONL "
            "for the verdict."
        ),
    )
    return parser.parse_args()


def _run_worker(args: argparse.Namespace) -> int:
    rank_env = os.environ.get("RANK")
    world_size_env = os.environ.get("WORLD_SIZE")
    local_rank_env = os.environ.get("LOCAL_RANK")
    if rank_env is None or world_size_env is None or local_rank_env is None:
        print(
            "[gate4.10] LOCAL_RANK / RANK / WORLD_SIZE must be set by the "
            "launcher (torchrun --nproc_per_node=2 ...)",
            file=sys.stderr,
            flush=True,
        )
        return 2

    rank = int(rank_env)
    world_size = int(world_size_env)
    local_rank = int(local_rank_env)

    # Gate 4.1 minimum fix carried forward: reuse torchrun's rendezvous
    # store via env:// instead of the loopback TCP fallback in
    # EngineConfig.distributed_addr.
    os.environ.setdefault("MINISGL_DISTRIBUTED_ADDR", "env://")

    log = RankLog(
        rank=rank,
        world_size=world_size,
        tp_size=world_size,
        model_path=args.model_path,
        prompts=[args.prompt_short, args.prompt_long],
        cold_start_attempt_id=args.cold_start_attempt_id,
        memory_sync_retry_note=args.memory_sync_retry_note,
    )

    if world_size != 2:
        log.failure_stage = "launch"
        log.failure_trace_summary = (
            f"Gate 4.10 pins world_size=2; got {world_size}"
        )
        log.status = "BLOCKED"
        log.emit()
        return 2

    # ------------------------------------------------------------------
    # A. Init-only smoke + B. Model load smoke (share one LLM boot)
    # ------------------------------------------------------------------
    llm = None
    try:
        import torch
        from minisgl.distributed import DistributedInfo
        from minisgl.llm import LLM

        tp_info = DistributedInfo(rank=rank, size=world_size)
        log.device = f"npu:{local_rank}"

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
    # C. TP=2 B=2 ragged prefill (cached_len == 0 branch) + decode
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
                f"Gate 4.10 requires unequal-length prompts; got "
                f"{tokenized_lengths} for prompts={prompts!r}"
            )

        # Only one generate() per process — no second batch would
        # ever hit the Gate 2.2f unsupported ragged + non-zero
        # cached_len + extend_len > 1 combination.
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

    except BaseException as exc:
        log.failure_stage = "prefill" if log.prefill_status == "PENDING" else "decode"
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
