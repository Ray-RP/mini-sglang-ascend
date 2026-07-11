"""Gate 5.1 Qwen3-4B TP=2 B=1 single-request bring-up smoke driver.

Runs a Qwen3-4B TP=2 B=1 end-to-end on 2 x Ascend 910B1:
    A. TP=2 init
    B. Qwen3-4B TP=2 model load
    C. B=1 single request, max_new_tokens=8

Envelope (locked at Gate 5.1):
    Ascend 910B1 x 2, TP=2, eager, npu_fia, bf16, greedy
    Qwen3-4B (/mnt/nvme/models/Qwen3-4B)
    max_new_tokens = 8, batch_size = 1
    memory_ratio=0.85, page_size=16, max_running_req=4
    cuda_graph_bs=[] (torch_npu has no CUDAGraph)
    MINISGL_DISTRIBUTED_ADDR=env://, use_pynccl=False

This driver deliberately restricts scope to Cases A / B / C only:
    * No B > 1, no ragged prefill, no mixed-KV decode, no dynamic
      admission.
    * No timing statistics, no repeats, no warmup.
    * No modification to python/minisgl/ or tests/.
    * No TP=4 / TP=8, no elasticity, no non-Qwen3 model.

Structured log format (per rank):
    A single JSON object on stdout tagged ``GATE5.1_JSONL rank=<r> ...``
    with the fields listed in Gate 5.1 spec section 4.

Footer (rank 0 only):
    ``GATE5.1_RESULT=<PASS|PARTIAL|BLOCKED>``

Exit code (rank 0):
    0 on PASS, 1 on FAIL, 2 on BLOCKED. Rank 1 mirrors via the CPU
    barrier in Scheduler.shutdown.
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
    tp_size: int
    model_path: str
    model_name: str
    model_exists: Optional[bool] = None
    device: str = ""
    model_config_summary: Optional[Dict[str, Any]] = None
    prompt: str = ""
    prompt_token_length: Optional[int] = None
    batch_size: int = 1
    baseline_available_tokens: Optional[int] = None
    baseline_free_pages: Optional[int] = None
    total_pages: Optional[int] = None
    init_status: str = "PENDING"
    load_status: str = "PENDING"
    prefill_status: str = "PENDING"
    decode_status: str = "PENDING"
    actual_output_tokens: Optional[int] = None
    output_text: Optional[str] = None
    output_token_ids: Optional[List[int]] = None
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
            "model_config_summary": self.model_config_summary,
            "prompt": self.prompt,
            "prompt_token_length": self.prompt_token_length,
            "batch_size": self.batch_size,
            "baseline_available_tokens": self.baseline_available_tokens,
            "baseline_free_pages": self.baseline_free_pages,
            "total_pages": self.total_pages,
            "init_status": self.init_status,
            "load_status": self.load_status,
            "prefill_status": self.prefill_status,
            "decode_status": self.decode_status,
            "actual_output_tokens": self.actual_output_tokens,
            "output_text": self.output_text,
            "output_token_ids": self.output_token_ids,
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
            f"GATE5.1_JSONL rank={self.rank} {json.dumps(payload, ensure_ascii=False)}",
            flush=True,
        )


def _err(rank: int, msg: str) -> None:
    print(f"[gate5.1][rank={rank}] {msg}", flush=True)


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


def _extract_model_config_summary(llm) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    hf_cfg = None
    for attr in ("hf_config", "model_config", "config"):
        candidate = getattr(llm, attr, None)
        if candidate is not None:
            hf_cfg = candidate
            break
    if hf_cfg is None:
        for outer in ("model_runner", "engine", "runner"):
            container = getattr(llm, outer, None)
            if container is not None:
                for attr in ("hf_config", "model_config", "config"):
                    candidate = getattr(container, attr, None)
                    if candidate is not None:
                        hf_cfg = candidate
                        break
            if hf_cfg is not None:
                break
    if hf_cfg is None:
        return summary
    for key in (
        "model_type",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "max_position_embeddings",
        "vocab_size",
        "head_dim",
        "tie_word_embeddings",
        "torch_dtype",
    ):
        try:
            value = getattr(hf_cfg, key, None)
            if value is not None:
                summary[key] = str(value) if key == "torch_dtype" else value
        except Exception:
            continue
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate 5.1 Qwen3-4B TP=2 B=1 single-request bring-up",
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
        help="Fixed at 8 for Gate 5.1; exposed only for debugging",
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
        default="Paris is",
        help="Single B=1 prompt; identical on both ranks so greedy outputs match.",
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
            "[gate5.1] LOCAL_RANK / RANK / WORLD_SIZE must be set by the "
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
        prompt=args.prompt,
        cold_start_attempt_id=args.cold_start_attempt_id,
        memory_sync_retry_note=args.memory_sync_retry_note,
    )
    log.model_exists = os.path.isdir(args.model_path)

    if world_size != 2:
        log.failure_stage = "launch"
        log.failure_reason = f"Gate 5.1 pins world_size=2; got {world_size}"
        log.status = "BLOCKED"
        log.emit()
        if rank == 0:
            print("GATE5.1_RESULT=BLOCKED", flush=True)
        return 2

    if not log.model_exists:
        log.failure_stage = "model_path"
        log.failure_reason = f"model_path does not exist: {args.model_path}"
        log.status = "BLOCKED"
        log.emit()
        if rank == 0:
            print("GATE5.1_RESULT=BLOCKED", flush=True)
        return 2

    # ------------------------------------------------------------------
    # A. TP=2 init + B. Qwen3-4B TP=2 model load (share one LLM boot)
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
        log.model_config_summary = _extract_model_config_summary(llm)

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
                print("GATE5.1_RESULT=BLOCKED", flush=True)
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
            print("GATE5.1_RESULT=BLOCKED", flush=True)
        return 2 if log.status == "BLOCKED" else 1

    # ------------------------------------------------------------------
    # C. TP=2 B=1 single-request generate()
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

        prompts = [args.prompt]
        try:
            log.prompt_token_length = len(llm.tokenizer.encode(args.prompt))
        except Exception:
            log.prompt_token_length = None
        log.batch_size = 1

        free_pages_before = len(llm.cache_manager.free_slots)

        t0 = time.perf_counter()
        results = llm.generate(prompts, sampling_params)
        t1 = time.perf_counter()
        log.generate_ms = (t1 - t0) * 1000.0
        _err(rank, f"generate() returned in {log.generate_ms:.2f} ms")

        if not isinstance(results, list) or len(results) != 1:
            raise RuntimeError(f"generate() returned malformed result: {results!r}")

        r0 = results[0]
        token_ids = r0["token_ids"]
        log.actual_output_tokens = len(token_ids)
        log.output_token_ids = list(token_ids)
        log.output_text = r0.get("text")

        if log.actual_output_tokens != args.max_new_tokens:
            raise RuntimeError(
                f"expected {args.max_new_tokens} tokens, got {log.actual_output_tokens} "
                f"(token_ids={token_ids!r})"
            )

        log.prefill_status = "PASS"
        log.decode_status = "PASS"

        free_pages_after = len(llm.cache_manager.free_slots)
        log.free_pages_before_after = [free_pages_before, free_pages_after]

    except BaseException as exc:
        log.failure_stage = "prefill" if log.prefill_status == "PENDING" else "decode"
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
            print("GATE5.1_RESULT=PARTIAL", flush=True)
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
            print("GATE5.1_RESULT=PASS", flush=True)
        elif log.status == "FAIL":
            print("GATE5.1_RESULT=PARTIAL", flush=True)
        else:
            print("GATE5.1_RESULT=BLOCKED", flush=True)

    return 0 if log.status == "PASS" else 1


def main() -> int:
    args = _parse_args()
    return _run_worker(args)


if __name__ == "__main__":
    sys.exit(main())
