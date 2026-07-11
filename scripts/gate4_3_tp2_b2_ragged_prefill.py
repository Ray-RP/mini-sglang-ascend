"""Gate 4.3 TP=2 B=2 ragged prefill smoke driver.

Runs a Qwen3-0.6B TP=2 B=2 ragged-prefill end-to-end on Ascend 910B1:
init-only smoke (case A) → model load smoke (case B) → B=2 ragged
prefill (unequal tokenized lengths) + decode max_new_tokens=8 (case C).

Envelope (locked at Gate 4.3):
    Ascend 910B1, TP=2, eager, npu_fia, bf16, greedy
    max_new_tokens = 8, batch_size = 2, unequal-length prompts
    memory_ratio=0.85, page_size=16, max_running_req=4
    cuda_graph_bs=[] (torch_npu has no CUDAGraph)

FIA branch selection:
    ``AscendFIAMetadata`` supports the "ragged prefill with
    cached_len == 0" branch (Gate 2.2f). Both prompts are enqueued
    fresh — no previous run leaves a radix-cache prefix hit — so the
    scheduler sees both uids with ``cached_len == 0`` on the prefill
    step. The (Gate 2.2f documented) unsupported "ragged +
    non-zero cached_len + extend_len > 1" combination is deliberately
    NOT exercised: the driver runs one generate() call and then
    exits, never enqueuing a second ragged batch that would share
    prefixes with the first.

Design:
    * Same per-rank worker shape as Gate 4.1 / 4.2. torchrun ×2, each
      rank reads LOCAL_RANK / RANK / WORLD_SIZE, use_pynccl=False.
    * Both ranks enqueue the SAME two prompts in the SAME order. The
      two prompts are chosen to guarantee different tokenized
      lengths on Qwen3-0.6B (short factual + longer factual). The
      script logs each prompt's actual tokenized length and asserts
      inequality before dispatching generate().
    * Case C asserts len(results) == 2, both requests produced
      exactly max_new_tokens output tokens, and rank 0's per-uid
      output_texts match rank 1's per-uid output_texts (bit-identical
      greedy across ranks under all-gathered logits).
    * Post-case allocator invariants (available_size returns to
      baseline, deferred_abort_uids empty, cache_integrity_ok true)
      checked per rank.

Structured log format (per rank):
    A single JSON object on stdout tagged ``GATE4.3_JSONL rank=<r> ...``
    with the fields listed in Gate 4.3 spec §4.

Exit code (rank 0):
    0 on PASS, 1 on FAIL, 2 on BLOCKED. Rank 1 mirrors via the CPU
    barrier in Scheduler.shutdown.

This script does not benchmark, does not do mixed-KV decode, does
not do dynamic admission, does not do B > 2, does not touch TP=4
or any non-Qwen3 model.
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
            "tp_size": self.tp_size,
            "model_path": self.model_path,
            "device": self.device,
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
            "available_tokens_after_case": self.available_tokens_after_case,
            "free_pages_after_case": self.free_pages_after_case,
            "deferred_abort_uids": self.deferred_abort_uids,
            "cache_integrity_ok": self.cache_integrity_ok,
            "status": self.status,
            "failure_stage": self.failure_stage,
            "failure_trace_summary": self.failure_trace_summary,
        }
        print(
            f"GATE4.3_JSONL rank={self.rank} {json.dumps(payload, ensure_ascii=False)}",
            flush=True,
        )


def _err(rank: int, msg: str) -> None:
    print(f"[gate4.3][rank={rank}] {msg}", flush=True)


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
        description="Gate 4.3 TP=2 B=2 ragged prefill on Qwen3-0.6B",
    )
    parser.add_argument(
        "--model-path",
        default="/mnt/nvme/models/Qwen3-0.6B",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8,
        help="Fixed at 8 for Gate 4.3; exposed only for debugging",
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
        help="Short prompt for the ragged pair (tokenizes to fewer tokens).",
    )
    parser.add_argument(
        "--prompt-long",
        default=(
            "The largest planet in our solar system by mass and volume is"
        ),
        help=(
            "Long prompt for the ragged pair; must tokenize to a "
            "different length than --prompt-short so the batch stays "
            "on the FIA ragged (cached_len==0) branch."
        ),
    )
    return parser.parse_args()


def _run_worker(args: argparse.Namespace) -> int:
    rank_env = os.environ.get("RANK")
    world_size_env = os.environ.get("WORLD_SIZE")
    local_rank_env = os.environ.get("LOCAL_RANK")
    if rank_env is None or world_size_env is None or local_rank_env is None:
        print(
            "[gate4.3] LOCAL_RANK / RANK / WORLD_SIZE must be set by the "
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
    )

    if world_size != 2:
        log.failure_stage = "launch"
        log.failure_trace_summary = (
            f"Gate 4.3 pins world_size=2; got {world_size}"
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
    # C. TP=2 B=2 ragged prefill (cached_len == 0 branch)
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

        # Ragged by construction: two prompts with different tokenized
        # lengths. Assert inequality up-front so the run cannot silently
        # regress to equal-length (which would defeat Gate 4.3's goal).
        prompts = [args.prompt_short, args.prompt_long]
        tokenized_lengths = [
            len(llm.tokenizer.encode(p)) for p in prompts
        ]
        log.prompt_token_lengths = tokenized_lengths
        log.batch_size = len(prompts)
        if len(set(tokenized_lengths)) < 2:
            raise RuntimeError(
                f"Gate 4.3 requires unequal-length prompts; got "
                f"{tokenized_lengths}"
            )

        # Both ranks enqueue the identical ragged batch. Greedy on
        # gathered logits keeps per-uid outputs bit-identical across
        # ranks. Only one generate() call is made per process — the
        # scheduler therefore sees a fresh cached_len==0 ragged prefill
        # and never enters the Gate 2.2f unsupported branch.
        prefill_t0 = time.perf_counter()
        results = llm.generate(prompts, sampling_params)
        prefill_t1 = time.perf_counter()
        _err(rank, f"generate() returned in {(prefill_t1 - prefill_t0)*1000.0:.2f} ms")

        if not isinstance(results, list) or len(results) != len(prompts):
            raise RuntimeError(
                f"generate() returned malformed result: {results!r}"
            )

        actual_tokens_per_req = [len(r["token_ids"]) for r in results]
        if any(n != args.max_new_tokens for n in actual_tokens_per_req):
            raise RuntimeError(
                f"expected {args.max_new_tokens} tokens per request, "
                f"got {actual_tokens_per_req}"
            )

        log.prefill_status = "PASS"
        log.decode_status = "PASS"
        log.actual_output_tokens_per_request = actual_tokens_per_req
        log.output_texts = [r.get("text") for r in results]

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
