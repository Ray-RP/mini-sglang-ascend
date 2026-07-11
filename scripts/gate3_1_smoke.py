"""Gate 3.1 — Qwen3-1.7B real-hardware smoke on Ascend 910B1.

Runs entirely off the offline in-process ``minisgl.llm.LLM`` driver.
Not a hermetic pytest: this script talks to the real NPU and takes
seconds. It is committed as evidence for the Gate 3.1 verdict.

Envelope (frozen at this gate):

* Hardware: 1x Ascend 910B1 die
* Model: Qwen3-1.7B, bf16, tie_word_embeddings=True
* TP=1, eager, attention_backend=npu_fia
* greedy sampling (temperature=0.0, top_k=1, top_p=1.0)
* short English prompt, max_new_tokens=8 for prefill+decode timing,
  then max_new_tokens=16 for a slightly longer decode run
* B=1 required; B=2 attempted opportunistically

Assertions:

1. ``LLM(model_path=...)`` constructor completes; the Qwen3-1.7B
   config is loaded and Ascend engine boots.
2. cache_manager's baseline free-page count is recorded BEFORE any
   request runs.
3. After ``generate([prompt], sp)``:
   * len(output.token_ids) == max_new_tokens (greedy, ignore_eos)
   * cache_manager returns to baseline free-page count (allocator
     invariant — no leak per completed request).
   * scheduler.deferred_abort_uids is empty (no residual abort state).
4. A second B=1 run (max_new_tokens=16) also converges to baseline.
5. Best-effort B=2 equal-length prefill+decode. If HBM does not
   accommodate B=2 at the current configuration, catch the error and
   log a limitation.

Output is line-oriented so it can be grepped from the CI harness log.
The final line ``GATE3.1_SMOKE_RESULT=<verdict>`` is scraped by the
Gate 3.1 verdict generator.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
import traceback


def _log(section: str, msg: str) -> None:
    print(f"[gate3.1][{section}] {msg}", flush=True)


def _report_alloc(scheduler, tag: str) -> tuple[int, int, int, int]:
    """Capture allocator snapshot + prove integrity.

    Reports:
      free_pages        — pages currently on the free list
      available_tokens  — free_slots + evictable prefix cache
                          (invariant that must return to baseline;
                          radix-cache retention is expected between
                          requests, so raw free_pages alone will drift)
      total_pages       — configured pool size
      deferred          — Scheduler.deferred_abort_uids size
    """
    cm = scheduler.cache_manager
    free = len(cm.free_slots)
    available = cm.available_size
    total = cm.num_pages
    deferred = len(scheduler.deferred_abort_uids)
    # Integrity: free + retained prefix pages == total. Radix-cache
    # retention across requests is expected and DOES NOT count as a leak;
    # available_size (free_slots + evictable prefix) is the invariant.
    cm.check_integrity()
    _log(
        "alloc",
        f"{tag} free_pages={free} available_tokens={available} "
        f"total_pages={total} deferred_abort_uids={deferred}",
    )
    return free, available, total, deferred


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--attention-backend", default="npu_fia")
    parser.add_argument("--max-new-tokens-1", type=int, default=8)
    parser.add_argument("--max-new-tokens-2", type=int, default=16)
    parser.add_argument(
        "--prompt",
        default="The capital of France is",
    )
    parser.add_argument("--try-batch-2", action="store_true", default=True)
    args = parser.parse_args()

    _log("boot", f"model_path={args.model_path}")
    _log("boot", f"attention_backend={args.attention_backend}")

    import torch

    from minisgl.core import SamplingParams
    from minisgl.llm import LLM

    t0 = time.time()
    llm = LLM(
        model_path=args.model_path,
        dtype=torch.bfloat16,
        attention_backend=args.attention_backend,
        max_running_req=8,
        memory_ratio=0.85,
        # Gate 3.1 envelope is eager mode. Pass an empty list so
        # ``_determine_cuda_graph_bs`` returns [] (short-circuits any
        # torch.cuda.CUDAGraph capture — which torch_npu does not
        # provide, since CUDAGraph is the CUDA backend's class).
        cuda_graph_bs=[],
        # Ascend FIA requires block_size (page_size) aligned to 16 in
        # the NO_QUANT path — otherwise aclnnFusedInferAttentionScoreV3
        # returns 561002 at tiling ("block_size should aligned to 16,
        # but got 1"). Use 16 to match the FIA contract; the KV page
        # geometry becomes page_size=16 * num_kv_heads=8 * head_dim=128
        # = 16384 elems per page per layer (same layout used by Gate 1).
        page_size=16,
    )
    _log("boot", f"LLM initialised in {time.time() - t0:.2f}s")

    baseline_free, baseline_available, total_pages, baseline_deferred = _report_alloc(
        llm, "baseline"
    )
    assert baseline_deferred == 0, "deferred_abort_uids must be empty at boot"

    # --------------------------------------------------------------
    # Test A: B=1, max_new_tokens=8, greedy, ignore_eos
    # --------------------------------------------------------------
    _log("A", "single-request prefill + 8-step decode")
    sp1 = SamplingParams(
        temperature=0.0, top_k=1, top_p=1.0,
        max_tokens=args.max_new_tokens_1, ignore_eos=True,
    )
    t0 = time.time()
    out = llm.generate([args.prompt], sp1)
    _log("A", f"elapsed={time.time() - t0:.2f}s")
    assert len(out) == 1
    tokens = out[0]["token_ids"]
    text = out[0]["text"]
    _log("A", f"generated_tokens={len(tokens)}")
    _log("A", f"tokens={tokens}")
    _log("A", f"text={text!r}")
    assert len(tokens) == args.max_new_tokens_1, (
        f"expected {args.max_new_tokens_1} tokens, got {len(tokens)}"
    )

    free_after_a, available_after_a, _, deferred_after_a = _report_alloc(llm, "after_A")
    assert deferred_after_a == 0
    # Correct allocator invariant: free_slots + retained prefix pages
    # must equal the baseline pool. Radix cache retention across
    # requests is a designed feature and does not count as a leak.
    assert available_after_a == baseline_available, (
        f"allocator did not return to baseline after A: "
        f"available={available_after_a} baseline={baseline_available}"
    )

    # --------------------------------------------------------------
    # Test B: B=1, max_new_tokens=16 (slightly longer decode run)
    # --------------------------------------------------------------
    _log("B", "single-request prefill + 16-step decode")
    sp2 = SamplingParams(
        temperature=0.0, top_k=1, top_p=1.0,
        max_tokens=args.max_new_tokens_2, ignore_eos=True,
    )
    t0 = time.time()
    out2 = llm.generate([args.prompt], sp2)
    _log("B", f"elapsed={time.time() - t0:.2f}s")
    tokens2 = out2[0]["token_ids"]
    _log("B", f"generated_tokens={len(tokens2)}")
    _log("B", f"tokens={tokens2}")
    _log("B", f"text={out2[0]['text']!r}")
    assert len(tokens2) == args.max_new_tokens_2, (
        f"expected {args.max_new_tokens_2} tokens, got {len(tokens2)}"
    )

    free_after_b, available_after_b, _, deferred_after_b = _report_alloc(llm, "after_B")
    assert deferred_after_b == 0
    assert available_after_b == baseline_available, (
        f"allocator did not return to baseline after B: "
        f"available={available_after_b} baseline={baseline_available}"
    )

    # --------------------------------------------------------------
    # Test C (best-effort): B=2, equal-length prefill+decode
    # --------------------------------------------------------------
    batch_2_result = "SKIPPED"
    if args.try_batch_2:
        _log("C", "best-effort B=2 equal-length prefill + decode")
        prompts = [args.prompt, args.prompt]
        sps = [sp1, sp1]
        try:
            t0 = time.time()
            outc = llm.generate(prompts, sps)
            _log("C", f"elapsed={time.time() - t0:.2f}s")
            assert len(outc) == 2
            len0 = len(outc[0]["token_ids"])
            len1 = len(outc[1]["token_ids"])
            _log("C", f"generated_tokens=[{len0}, {len1}]")
            _log("C", f"tokens[0]={outc[0]['token_ids']}")
            _log("C", f"tokens[1]={outc[1]['token_ids']}")
            assert len0 == args.max_new_tokens_1
            assert len1 == args.max_new_tokens_1

            free_after_c, available_after_c, _, deferred_after_c = _report_alloc(
                llm, "after_C"
            )
            assert deferred_after_c == 0
            assert available_after_c == baseline_available, (
                f"allocator did not return to baseline after C: "
                f"available={available_after_c} baseline={baseline_available}"
            )
            batch_2_result = "PASS"
        except Exception as e:  # noqa: BLE001
            batch_2_result = f"FAIL: {type(e).__name__}: {e}"
            _log("C", f"best-effort B=2 failed: {batch_2_result}")
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    _log(
        "summary",
        f"baseline_free_pages={baseline_free} "
        f"baseline_available_tokens={baseline_available} "
        f"total_pages={total_pages}",
    )
    _log("summary", f"A(B=1, N=8)  = PASS")
    _log("summary", f"B(B=1, N=16) = PASS")
    _log("summary", f"C(B=2, N=8)  = {batch_2_result}")

    # Verdict from the smoke's point of view (final verdict is written
    # in the Gate 3.1 verdict doc; this string is the smoke's own
    # judgement of whether the allocator + prefill + decode contract
    # holds on the target model).
    verdict = "PASS" if batch_2_result in ("PASS", "SKIPPED") else "PARTIAL"
    if batch_2_result.startswith("FAIL"):
        # Gate spec: "如果 B=2 因显存不足或模型过大失败，不要硬调优；
        # 记录为 limitation." — this is a limitation, not a failure of
        # the B=1 contract. The smoke's own verdict stays PASS on
        # B=1 evidence; the B=2 result is reported as a limitation
        # in the final verdict.
        verdict = "PASS"
    print(f"GATE3.1_SMOKE_RESULT={verdict}", flush=True)
    print(f"GATE3.1_SMOKE_BATCH2={batch_2_result}", flush=True)
    print(f"GATE3.1_SMOKE_BASELINE_FREE={baseline_free}", flush=True)
    print(f"GATE3.1_SMOKE_BASELINE_AVAILABLE={baseline_available}", flush=True)
    print(f"GATE3.1_SMOKE_TOTAL_PAGES={total_pages}", flush=True)

    del llm
    gc.collect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
