"""Gate 3.3 — TP=1 Ascend capability matrix smoke on Ascend 910B1.

Not a hermetic pytest: talks to the real NPU. Committed as evidence for
the Gate 3.3 verdict. This gate is a **capability boundary snapshot**,
not a performance benchmark. It runs a fixed matrix of request shapes
against every in-scope Qwen3 model that already ships on the Ascend
host and records structured evidence per case.

Envelope (identical to Gate 3.1 / 3.2, frozen):

    Hardware:          Ascend 910B1 (1 die)
    Parallelism:       TP=1
    Execution:         eager (cuda_graph_bs=[])
    Attention backend: npu_fia
    Sampling:          greedy (temperature=0.0, top_k=1, top_p=1.0,
                       ignore_eos=True)

Models (only local paths already present under /mnt/nvme/models/):

    Qwen3-0.6B         baseline from Gate 1/2.x
    Qwen3-1.7B         Gate 3.1 / 3.2 target

No new architecture family, no quantized variant, no TP > 1, no
Qwen3-4B, no MoE. See §5 of the verdict for the exclusion list.

Case matrix (per model):

    A. B=1  max_new_tokens=8    prefill + 8-step decode
    B. B=1  max_new_tokens=16   prefill + 16-step decode
    C. B=2  max_new_tokens=8    equal-length prefill + decode
    D. B=2  max_new_tokens=8    ragged prefill (short + long prompt)
    E. B=2  max_new_tokens=8    mixed-KV decode after ragged prefill
                                (distinct prompt content vs. D — the two
                                 requests carry different cached_len at
                                 every decode step)

Per-case JSONL row (§4 of the verdict) captures:

    model_name, model_path, case_name,
    prompt_lengths_chars, requested_max_new_tokens,
    actual_output_tokens_per_request,
    batch_size_timeline, baseline_available_tokens,
    available_tokens_after_case, deferred_abort_uids,
    cache_integrity_ok, status, failure_reason

PASS predicate (per case):

    len(output) == B and lens == requested and
    available_tokens_after_case == baseline_available_tokens and
    deferred_abort_uids == 0 and cache_integrity_ok

Process model:

    ``minisgl.distributed.info.set_tp_info`` is a process-global
    singleton — instantiating a second ``LLM`` in the same process
    raises ``RuntimeError: TP info has been set``. Therefore each model
    runs in its own **child** subprocess. The parent invocation spawns
    one child per model in ``--models`` and aggregates JSONL rows into
    the final matrix output.

Footer lines (scraped by the verdict generator):

    GATE3.3_MODEL_<name>=PASS|PARTIAL|FAIL
    GATE3.3_MATRIX_RESULT=PASS|PARTIAL|BLOCKED
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from typing import List


def _log(section: str, msg: str) -> None:
    print(f"[gate3.3][{section}] {msg}", flush=True)


# ---------------------------------------------------------------- helpers

def _snapshot_alloc(scheduler) -> dict:
    """Capture the allocator snapshot used by the PASS predicate.

    Correct paged-cache invariant (see Gate 3.1 verdict §4):
      ``available_size`` = ``free_slots + evictable prefix pages``.
    Radix-cache retention across requests is a designed feature; the
    invariant that must return to baseline is ``available_size``.
    """
    cm = scheduler.cache_manager
    return {
        "free_pages": len(cm.free_slots),
        "available_tokens": cm.available_size,
        "total_pages": cm.num_pages,
        "deferred_abort_uids": len(scheduler.deferred_abort_uids),
    }


def _check_alloc_and_integrity(scheduler, tag: str) -> dict:
    snap = _snapshot_alloc(scheduler)
    ok = True
    try:
        scheduler.cache_manager.check_integrity()
    except Exception as e:  # noqa: BLE001
        ok = False
        _log("alloc", f"{tag} check_integrity FAILED: {type(e).__name__}: {e}")
    snap["check_integrity_ok"] = ok
    _log(
        "alloc",
        f"{tag} free_pages={snap['free_pages']} "
        f"available_tokens={snap['available_tokens']} "
        f"total_pages={snap['total_pages']} "
        f"deferred_abort_uids={snap['deferred_abort_uids']} "
        f"integrity_ok={ok}",
    )
    return snap


def _model_name_from_path(path: str) -> str:
    return os.path.basename(os.path.normpath(path))


def _pass_predicate(
    output_len: int,
    lens: List[int],
    requested: List[int],
    baseline: dict,
    after: dict,
) -> bool:
    return (
        output_len == len(requested)
        and lens == requested
        and after["available_tokens"] == baseline["available_tokens"]
        and after["deferred_abort_uids"] == 0
        and after.get("check_integrity_ok", False)
    )


# ---------------------------------------------------------------- single-model driver

def _build_llm(args):
    """Boot the offline LLM under the Gate 3.1/3.2/3.3 frozen envelope."""
    import torch
    from minisgl.llm import LLM

    t0 = time.time()
    llm = LLM(
        model_path=args.model_path,
        dtype=torch.bfloat16,
        attention_backend=args.attention_backend,
        max_running_req=8,
        memory_ratio=0.85,
        # Eager mode — torch_npu has no CUDAGraph capture.
        cuda_graph_bs=[],
        # FIA NO_QUANT tiling requires page_size % 16 == 0.
        page_size=16,
    )
    _log("boot", f"LLM initialised in {time.time() - t0:.2f}s")
    return llm


def _generate(llm, prompts, sps):
    """Wrapper around llm.generate that returns dict-shaped outputs."""
    return llm.generate(prompts, sps if isinstance(sps, list) else [sps] * len(prompts))


def _run_case(
    llm,
    model_name: str,
    model_path: str,
    case_name: str,
    description: str,
    prompts: List[str],
    requested_max_new_tokens: List[int],
    batch_size_timeline: List[int],
    jsonl_rows: List[dict],
) -> str:
    """Common per-case runner. Returns PASS/FAIL and appends a JSONL row."""
    from minisgl.core import SamplingParams

    tag = f"{case_name}[{model_name}]"
    baseline = _check_alloc_and_integrity(llm, f"{tag}_baseline")

    sps = [
        SamplingParams(
            temperature=0.0, top_k=1, top_p=1.0,
            max_tokens=n, ignore_eos=True,
        )
        for n in requested_max_new_tokens
    ]

    failure_reason = None
    lens: List[int] = []
    reply_texts: List[str] = []
    try:
        t0 = time.time()
        out = _generate(llm, prompts, sps)
        elapsed = time.time() - t0
        _log(tag, f"elapsed={elapsed:.2f}s")
        for i, o in enumerate(out):
            _log(tag, f"req{i} tokens={o['token_ids']}")
            _log(tag, f"req{i} text={o['text']!r}")
        lens = [len(o["token_ids"]) for o in out]
        reply_texts = [o["text"] for o in out]
        output_len = len(out)
    except Exception as e:  # noqa: BLE001
        failure_reason = f"{type(e).__name__}: {e}"
        _log(tag, f"generate raised: {failure_reason}")
        traceback.print_exc()
        output_len = 0

    after = _check_alloc_and_integrity(llm, f"{tag}_after")
    pass_ok = failure_reason is None and _pass_predicate(
        output_len, lens, requested_max_new_tokens, baseline, after
    )
    verdict = "PASS" if pass_ok else "FAIL"
    if not pass_ok and failure_reason is None:
        # Build a structured failure reason for the JSONL row.
        reasons = []
        if output_len != len(requested_max_new_tokens):
            reasons.append(
                f"output_count={output_len} expected={len(requested_max_new_tokens)}"
            )
        if lens != requested_max_new_tokens:
            reasons.append(f"lens={lens} expected={requested_max_new_tokens}")
        if after["available_tokens"] != baseline["available_tokens"]:
            reasons.append(
                f"available_tokens {baseline['available_tokens']} -> "
                f"{after['available_tokens']}"
            )
        if after["deferred_abort_uids"] != 0:
            reasons.append(f"deferred_abort_uids={after['deferred_abort_uids']}")
        if not after.get("check_integrity_ok", False):
            reasons.append("check_integrity_failed")
        failure_reason = "; ".join(reasons)

    row = {
        "model_name": model_name,
        "model_path": model_path,
        "case_name": case_name,
        "description": description,
        "prompt_lengths_chars": [len(p) for p in prompts],
        "requested_max_new_tokens": requested_max_new_tokens,
        "actual_output_tokens_per_request": lens,
        "batch_size_timeline": batch_size_timeline,
        "baseline_available_tokens": baseline["available_tokens"],
        "available_tokens_after_case": after["available_tokens"],
        "baseline_free_pages": baseline["free_pages"],
        "free_pages_after_case": after["free_pages"],
        "total_pages": baseline["total_pages"],
        "deferred_abort_uids": after["deferred_abort_uids"],
        "cache_integrity_ok": after.get("check_integrity_ok", False),
        "reply_texts_equal": (
            len(reply_texts) == 2 and reply_texts[0] == reply_texts[1]
        ),
        "status": verdict,
        "failure_reason": failure_reason,
    }
    jsonl_rows.append(row)
    _log(tag, f"verdict={verdict}")
    return verdict


def run_single_model(args) -> int:
    """Run cases A-E against a single model in this process and print JSONL."""
    model_name = _model_name_from_path(args.model_path)
    _log("boot", f"model_name={model_name}")
    _log("boot", f"model_path={args.model_path}")
    _log("boot", f"attention_backend={args.attention_backend}")

    jsonl_rows: List[dict] = []
    verdicts: dict = {}

    llm = None
    try:
        llm = _build_llm(args)

        # --------------------------------------------- Case A — B=1 N=8
        verdicts["A"] = _run_case(
            llm,
            model_name=model_name,
            model_path=args.model_path,
            case_name="A",
            description="B=1 single request, prefill + 8-step decode",
            prompts=[args.prompt_short],
            requested_max_new_tokens=[8],
            batch_size_timeline=[1],
            jsonl_rows=jsonl_rows,
        )

        # --------------------------------------------- Case B — B=1 N=16
        verdicts["B"] = _run_case(
            llm,
            model_name=model_name,
            model_path=args.model_path,
            case_name="B",
            description="B=1 single request, prefill + 16-step decode",
            prompts=[args.prompt_short],
            requested_max_new_tokens=[16],
            batch_size_timeline=[1],
            jsonl_rows=jsonl_rows,
        )

        # --------------------------------------------- Case C — B=2 equal-length
        verdicts["C"] = _run_case(
            llm,
            model_name=model_name,
            model_path=args.model_path,
            case_name="C",
            description="B=2 equal-length batching, prefill + 8-step decode",
            prompts=[args.prompt_short, args.prompt_short],
            requested_max_new_tokens=[8, 8],
            batch_size_timeline=[2],
            jsonl_rows=jsonl_rows,
        )

        # --------------------------------------------- Case D — B=2 ragged prefill
        short_prompt = "Hi."
        long_prompt = (
            "The Ascend 910B1 accelerator has 64 gigabytes of high bandwidth "
            "memory and is used for large language model inference."
        )
        verdicts["D"] = _run_case(
            llm,
            model_name=model_name,
            model_path=args.model_path,
            case_name="D",
            description="B=2 ragged prefill (short + long prompt), N=8",
            prompts=[short_prompt, long_prompt],
            requested_max_new_tokens=[8, 8],
            batch_size_timeline=[2],
            jsonl_rows=jsonl_rows,
        )

        # --------------------------------------------- Case E — B=2 mixed-KV decode
        # Distinct prompt content from D. Both prompts are ragged so the two
        # decoding requests carry different cached_len at every decode step
        # (this is "mixed-KV" during the shared decode phase). N=8 for both
        # keeps the case boundary at exactly the required max_new_tokens=8.
        e_short = "Hello."
        e_long = (
            "In one short paragraph, describe the color of a clear afternoon "
            "sky over the ocean, using only common English words."
        )
        verdicts["E"] = _run_case(
            llm,
            model_name=model_name,
            model_path=args.model_path,
            case_name="E",
            description="B=2 mixed-KV decode after ragged prefill, N=8",
            prompts=[e_short, e_long],
            requested_max_new_tokens=[8, 8],
            batch_size_timeline=[2],
            jsonl_rows=jsonl_rows,
        )

    finally:
        if llm is not None:
            del llm
        gc.collect()

    all_pass = all(v == "PASS" for v in verdicts.values())
    model_verdict = "PASS" if all_pass else "FAIL"

    _log(
        "summary",
        f"model={model_name} "
        f"A={verdicts.get('A')} B={verdicts.get('B')} "
        f"C={verdicts.get('C')} D={verdicts.get('D')} "
        f"E={verdicts.get('E')}",
    )

    # Emit JSONL rows via a scrape-friendly prefix so the parent can
    # pick them up from the child stdout.
    for row in jsonl_rows:
        print("GATE3.3_JSONL " + json.dumps(row, ensure_ascii=False), flush=True)

    if args.jsonl_out:
        with open(args.jsonl_out, "w", encoding="utf-8") as f:
            for row in jsonl_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    for case, verdict in verdicts.items():
        print(f"GATE3.3_MODEL_{model_name}_CASE_{case}={verdict}", flush=True)
    print(f"GATE3.3_MODEL_{model_name}={model_verdict}", flush=True)
    return 0 if all_pass else 1


# ---------------------------------------------------------------- parent orchestrator

def run_matrix(args) -> int:
    """Parent invocation. Spawns one child per model. Aggregates JSONL rows."""
    model_paths = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_paths:
        _log("boot", "no --models paths provided; nothing to do")
        print("GATE3.3_MATRIX_RESULT=BLOCKED", flush=True)
        return 2

    _log("boot", f"model_paths={model_paths}")
    _log("boot", f"attention_backend={args.attention_backend}")
    _log("boot", f"jsonl_out={args.jsonl_out or '<stdout only>'}")

    aggregated_rows: List[dict] = []
    model_verdicts: dict = {}
    model_case_verdicts: dict = {}

    env = os.environ.copy()

    for model_path in model_paths:
        model_name = _model_name_from_path(model_path)
        _log("matrix", f"--- START model={model_name} ---")

        child_cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--single",
            "--model-path", model_path,
            "--attention-backend", args.attention_backend,
            "--prompt-short", args.prompt_short,
        ]
        _log("matrix", f"child_cmd={' '.join(child_cmd)}")

        t0 = time.time()
        proc = subprocess.run(
            child_cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
        elapsed = time.time() - t0
        _log("matrix", f"child model={model_name} rc={proc.returncode} "
                       f"elapsed={elapsed:.2f}s")

        # Passthrough child output for debug + evidence.
        for line in proc.stdout.splitlines():
            print(line, flush=True)

        cases_for_this_model: dict = {}
        for line in proc.stdout.splitlines():
            if line.startswith("GATE3.3_JSONL "):
                try:
                    row = json.loads(line[len("GATE3.3_JSONL "):])
                except json.JSONDecodeError:
                    continue
                aggregated_rows.append(row)
            elif line.startswith(f"GATE3.3_MODEL_{model_name}_CASE_"):
                # GATE3.3_MODEL_<name>_CASE_<letter>=<verdict>
                key = line.split("=", 1)[0]
                letter = key.rsplit("_CASE_", 1)[1]
                cases_for_this_model[letter] = line.split("=", 1)[1]
            elif line.startswith(f"GATE3.3_MODEL_{model_name}="):
                model_verdicts[model_name] = line.split("=", 1)[1]

        model_case_verdicts[model_name] = cases_for_this_model
        _log("matrix", f"--- END model={model_name} ---")

    # Persist aggregated JSONL if requested.
    if args.jsonl_out:
        with open(args.jsonl_out, "w", encoding="utf-8") as f:
            for row in aggregated_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        _log("matrix", f"wrote {len(aggregated_rows)} JSONL rows -> {args.jsonl_out}")

    # Overall matrix verdict:
    #   PASS     — every model passed every required case A-E.
    #   PARTIAL  — at least one model fully PASSed; another has a
    #              documented case-level failure (recorded in the row's
    #              ``failure_reason`` field).
    #   BLOCKED  — no model even reached the B=1 baseline (case A / B
    #              missing for every model). The parent also returns
    #              BLOCKED if the child failed to launch (no scraped
    #              GATE3.3_MODEL_* line).
    passing_models = [m for m, v in model_verdicts.items() if v == "PASS"]
    if len(passing_models) == len(model_paths) and passing_models:
        overall = "PASS"
    elif not model_verdicts:
        overall = "BLOCKED"
    elif len(passing_models) >= 1:
        overall = "PARTIAL"
    else:
        # Every model failed. Distinguish BLOCKED (case A missing everywhere)
        # from PARTIAL-not-really by checking if A/B ever ran successfully.
        any_baseline_pass = any(
            cases.get("A") == "PASS" or cases.get("B") == "PASS"
            for cases in model_case_verdicts.values()
        )
        overall = "PARTIAL" if any_baseline_pass else "BLOCKED"

    # Structured footer for the verdict generator.
    for model_name, verdict in model_verdicts.items():
        cases = model_case_verdicts.get(model_name, {})
        _log(
            "summary",
            f"model={model_name} verdict={verdict} cases={cases}",
        )
        for letter in ("A", "B", "C", "D", "E"):
            print(
                f"GATE3.3_MODEL_{model_name}_CASE_{letter}={cases.get(letter, 'MISSING')}",
                flush=True,
            )
        print(f"GATE3.3_MODEL_{model_name}={verdict}", flush=True)

    print(f"GATE3.3_MATRIX_RESULT={overall}", flush=True)
    return 0 if overall == "PASS" else (1 if overall == "PARTIAL" else 2)


# ------------------------------------------------------------------ main

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--single",
        action="store_true",
        help="Child mode: run cases A-E against a single --model-path.",
    )
    parser.add_argument(
        "--model-path",
        default="",
        help="(child mode) Absolute path to one local model directory.",
    )
    parser.add_argument(
        "--models",
        default="",
        help=(
            "(parent mode) Comma-separated list of model paths. The parent "
            "spawns one child subprocess per model because "
            "``minisgl.distributed.info.set_tp_info`` is a process-global "
            "singleton — a second LLM boot in the same process raises "
            "``RuntimeError: TP info has been set``."
        ),
    )
    parser.add_argument("--attention-backend", default="npu_fia")
    parser.add_argument(
        "--prompt-short",
        default="The capital of France is",
    )
    parser.add_argument(
        "--jsonl-out",
        default="",
        help="Optional path to write aggregated JSONL rows.",
    )
    args = parser.parse_args()

    if args.single:
        if not args.model_path:
            print("--single requires --model-path", file=sys.stderr, flush=True)
            return 2
        return run_single_model(args)

    if not args.models:
        print("--models is required in parent mode", file=sys.stderr, flush=True)
        return 2
    return run_matrix(args)


if __name__ == "__main__":
    sys.exit(main())
