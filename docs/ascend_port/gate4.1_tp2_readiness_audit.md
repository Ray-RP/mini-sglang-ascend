# Gate 4.1 Readiness Audit — TP=2 single-request bring-up path

**Gate ID:** 4.1 (TP=2 Ascend single-request bring-up on Qwen3-0.6B)
**Kind:** read-only source audit — no code touched at this step
**Branch:** `gate4.1-tp2-single-request-bringup`
**Base commit:** `c651d91` (tip of `ascend-port`, Gate 3.4 merge)
**Date:** 2026-07-11
**Scope:** identify whether Mini-SGLang-Ascend at `c651d91` already
contains a viable TP=2 runtime path and where a Qwen3-0.6B TP=2
single-request bring-up will first block, without proposing large
refactors.

This audit exists to answer three questions defined by the Gate 4.1
opening:

1. 当前代码是否已经有 TP=2 runtime path？
2. Gate 1 的 HCCL init 证据覆盖到哪里？
3. TP=2 卡住在 init、weight loading、forward、attention、sampler 还是 cleanup？

---

## 1. Distributed init path

### 1.1 `set_tp_info` — process-global singleton

`python/minisgl/distributed/info.py:21-25` exposes a one-shot
`set_tp_info(rank, size)`; the second call raises
`RuntimeError: TP info has been set`. This is the same singleton that
forced Gate 3.3 / 3.4 to spawn one child subprocess per model.

For TP=2 the driver must therefore be launched as **two separate
processes**, one per rank. A `torchrun --nproc_per_node=2` (or a plain
`RANK=… LOCAL_RANK=… WORLD_SIZE=2 python …` pair) is the natural
carrier — each process independently calls `set_tp_info(rank, 2)`.

### 1.2 `initialize_distributed_from_env` — env-var driven

`python/minisgl/distributed/runtime.py:100-160` reads
`LOCAL_RANK` / `RANK` / `WORLD_SIZE`, calls `bind_local_device` (which
imports `torch_npu` lazily on `npu`, then `torch.npu.set_device(local_rank)`),
then `dist.init_process_group(backend=backend)` with backend selected
by device (`hccl` on NPU). It performs no collective; the group is
lazily used by the first accelerator collective invocation.

The engine (below) does **not** call this helper — it calls
`bind_local_device` directly and drives `init_process_group` from its
own `_init_communication`. `initialize_distributed_from_env` is
consumed only by the standalone HCCL smoke (`tests/ascend/hccl_smoke.py`)
today. Both call sites end up in the same `dist.init_process_group`,
so the runtime helper is not on the Gate 4.1 critical path.

### 1.3 `Engine._init_communication` — TP>1 branch on NPU

`python/minisgl/engine/engine.py:148-191`:

```python
if self.device_type == "npu" and config.tp_info.size == 1:
    return None                               # Gate 1 fast-path: skip torch.dist entirely

if config.tp_info.size == 1 or config.use_pynccl:
    torch.distributed.init_process_group(
        backend="gloo",
        rank=config.tp_info.rank,
        world_size=config.tp_info.size,
        timeout=timedelta(seconds=config.distributed_timeout),
        init_method=config.distributed_addr,
    )
    tp_cpu_group = torch.distributed.group.WORLD
    ...
    enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
else:
    accel_backend = get_distributed_backend(self.device_type)   # "hccl" on NPU
    torch.distributed.init_process_group(
        backend=accel_backend, rank=..., world_size=..., init_method=config.distributed_addr,
    )
    tp_cpu_group = torch.distributed.new_group(backend="gloo")
```

Two implications for TP=2 on NPU:

* The `use_pynccl` default in `EngineConfig` is **True** (`engine/config.py:29`).
  Under `use_pynccl=True` the primary group is `gloo` and PyNCCL is
  layered on top via `enable_pynccl_distributed` — which imports
  `minisgl.kernel.init_pynccl`. The PyNCCL wrapper is a CUDA-only
  compile artefact (Gate 1 verdict called this out explicitly); on
  Ascend it is unavailable. **TP=2 on NPU must therefore be driven
  with `use_pynccl=False`** so the engine takes the `else` branch and
  initialises the HCCL group directly, with a `gloo` sidecar for CPU
  collectives (all_reduce of min/max free memory, ack broadcasts,
  scheduler barriers).
* `distributed_addr` is hard-coded to `tcp://127.0.0.1:2333`
  (`engine/config.py:54`). `init_process_group` treats
  `init_method="tcp://127.0.0.1:2333"` as authoritative and does not
  read `MASTER_ADDR`/`MASTER_PORT` env vars in that case. So a
  `torchrun` launch does not need `MASTER_ADDR` — the loopback URI is
  used. A second concurrent job on the same host would collide on
  port 2333; the driver must serialise runs (matches every prior gate).

### 1.4 Rank-1 shutdown symmetry

`Engine.shutdown` (line 289-298) calls `destroy_process_group` when
`tp_cpu_group is not None`. Under TP=2 both ranks own their group and
must both reach shutdown — an early exception on one rank without
`destroy_process_group` on the peer leaves HCCL and gloo in an
undefined state. `Scheduler.shutdown` (line 191-199) calls
`synchronize_device` → `sync_all_ranks` (CPU barrier) → drain →
engine.shutdown. That ordering is symmetric across ranks.

---

## 2. Offline `LLM` driver — TP=1 lockdown

`python/minisgl/llm/llm.py:29-40` hardcodes:

```python
config = SchedulerConfig(
    model_path=model_path,
    tp_info=DistributedInfo(0, 1),        # <-- TP=1 only
    dtype=dtype,
    offline_mode=True,
    **kwargs,
)
```

This is the **single blocker** at the driver layer: even in a
torchrun-launched rank-1 process the offline `LLM` would set itself
to `(rank=0, size=1)`, which:

* prevents `Engine._init_communication` from entering the HCCL branch,
* makes `_shard_tensor` a no-op (whole tensor per rank),
* makes `VocabParallelEmbedding` / `LinearColumnParallel` /
  `LinearRowParallel` / `Qwen3Attn` all take the `tp_size == 1`
  fast-paths (no all_reduce, no all_gather), and
* the two processes end up as two independent TP=1 replicas with no
  HCCL comm — not the intended TP=2 shape.

Gate 4.1's minimum fix (per §4 of the opening — "rank/world_size 参数
传递") is therefore: **let `LLM.__init__` accept `tp_info` as a keyword
argument and forward it to `SchedulerConfig`**, defaulting to
`DistributedInfo(0, 1)` to keep every existing offline TP=1 call site
unchanged.

The `SchedulerIOMixin.__init__` offline-mode short-circuit
(`scheduler/io.py:30-33`) returns early **before** setting up the
rank-0→rank-N ZMQ broadcast, so an offline TP=2 driver cannot use
that broadcast to fan out prompts. Instead **both ranks must
independently enqueue the same prompt list** — which is naturally the
case if both processes run `LLM.generate([prompt], sp)` in lock-step
with identical arguments (`generate` builds `pending_requests`
locally). This is the exact pattern used by SGLang's and vLLM's
in-process TP drivers.

Rank-1 output emission: `LLM.offline_send_result` fills
`status_map[msg.uid].output_ids`. Only rank 0 emits from the sampler
(the `Sampler` returns a `next_tokens_gpu` computed from the fused
logits which have already been all-gathered by `ParallelLMHead`),
but under offline mode both ranks run the same forward and same
sampler on the same all-gathered logits — so both processes'
`status_map` end up with identical `output_ids`. The driver need only
report rank 0's `status_map`.

---

## 3. Weight sharding path — TP=2 ready

`python/minisgl/models/weight.py:75-124` (`load_weight`):

* pulls `tp_info = get_tp_info()` and iterates safetensors files
  streaming.
* `_shard_tensor` (line 34-52):
  * Q / K / V / gate / up: split along dim 0 by rank; K/V respects
    `num_kv_heads < tp_size` replication (chunk-then-index-by-head).
  * O / down: split along dim 1 by rank (row-parallel input).
  * `lm_head` / `embed_tokens`: vocab-split along dim 0 by rank.
  * Everything else: replicated (no split).

For Qwen3-0.6B (`num_kv_heads=8`), TP=2 means:

* Q: dim-0 split into 2 shards (16 heads → 8 heads/rank).
* K, V: `num_kv_heads=8 >= tp_size=2`, chunk-into-2 along dim 0 (8 heads → 4 heads/rank).
* O: dim-1 split into 2 shards.
* MLP gate/up: dim-0 split into 2 shards (`intermediate_size=3072` → 1536/rank).
* MLP down: dim-1 split into 2 shards.
* Embedding & lm_head: vocab (151936) split into 2 shards.

All these divisions are clean integers — no padding required, no
`div_ceil` odd tails. **Weight sharding is not a blocker for
Qwen3-0.6B TP=2.**

---

## 4. TP-aware layers — audit summary

Grep for `get_tp_info` under `python/minisgl/layers/`:

| Layer file | TP behaviour |
|---|---|
| `layers/linear.py` | `LinearColumnParallel` (dim-0 output split), `LinearRowParallel` (dim-1 input split + all_reduce). `_tp_size = tp_info.size`; guards `if self._tp_size > 1` before the collective. |
| `layers/embedding.py` | `VocabParallelEmbedding` (masked partial lookup + all_reduce). `ParallelLMHead` extends embedding, all_gather output across TP. |
| `layers/attention.py` | `num_qo_heads = div_even(num_qo_heads, tp_size)` and `num_kv_heads = div_even(num_kv_heads, tp_size, allow_replicate=True)`. |
| `layers/moe.py` | Not exercised (Qwen3-0.6B is dense). |

None of these fault on `tp_size == 1` (all take the fast-path) and
none of them assert `tp_size == 1`. **TP=2 forward path is exercised
purely by construction.**

---

## 5. Attention backend (`npu_fia`) — TP-agnostic

`python/minisgl/attention/ascend_fia.py` operates on per-rank
`num_qo_heads` and `num_kv_heads` handed in by `Qwen3Attn`. It has no
`get_tp_info` reference. The docstring block (lines 10-24) enumerates
supported and refused batch shapes:

* Supported: B≥1 decode, B≥1 equal-length prefill, B≥1 ragged prefill
  with `cached_len == 0`.
* Refused (`NotImplementedError`): ragged batches with any non-zero
  `cached_len`; decode with per-request `cached_len` variance.

Gate 4.1 only fires a **single request** (B=1), so the FIA path never
hits ragged mixed-cached_len. `AscendFIAMetadata` is not a Gate 4.1
blocker.

---

## 6. KV cache allocation per rank

`Engine._determine_num_pages` (line 202-222):

```python
cache_per_page = (
    2  # key + value
    * head_dim
    * div_even(num_kv_heads, tp_info.size, allow_replicate=True)   # <-- per-rank KV heads
    * page_size
    * dtype.itemsize
    * num_layers
)
```

Under TP=2 with `num_kv_heads=8`, per-page KV footprint is halved on
each rank — so per-rank KV cache grows to ~2× the token capacity a
TP=1 rank would allocate at the same memory ratio. This is the
expected behaviour; the `min_free_memory` / `max_free_memory`
imbalance check (line 233-247, 2 GiB tolerance) guards against a
misconfigured rank.

The `_sync_get_memory` all_reduce uses the `gloo` sidecar
(`tp_cpu_group`) — not HCCL — so it does not require a live NPU
context to arrive at the min/max. Correct by construction.

---

## 7. Sampler / logits rank ownership

`ParallelLMHead.forward` all-gathers the per-rank logits along the
vocab dim and returns the fully replicated logits tensor on every
rank (`layers/embedding.py:116-126`). `Sampler.sample` (called on line
264 of `engine.py`) then runs identically on all ranks and produces
identical `next_tokens_gpu` because greedy on identical logits is
deterministic.

Because both ranks compute the same next-token independently, no
extra rank-0→rank-1 broadcast is required for the greedy path
tracked by Gate 4.1.

---

## 8. Gate 1 HCCL evidence — coverage boundary

`tests/ascend/hccl_smoke.py` (Gate 1) proves:

* `initialize_distributed_from_env` succeeds at `WORLD_SIZE=8`.
* Backend resolves to `hccl`, device to `npu`.
* `all_reduce` sum matches `world_size*(world_size+1)/2`.
* `broadcast(src=0)` delivers the sentinel payload to every rank.
* `all_gather` returns `[0..7]`.
* `barrier()` succeeds.
* `destroy_process_group()` succeeds.

Gate 1's verdict itself notes as an explicit non-goal (line 56-57):

> *TP > 1 collective execution on NPU (HCCL smoke test only asserts
> init; no forward/decode across ranks).*

So the HCCL primitive path is proven at world_size=8, but:

* no engine forward / no model weight load / no scheduler tick,
* no `Engine._init_communication` under the TP>1 branch is exercised
  in any regression test,
* no `_sync_get_memory` gloo-sidecar all_reduce is exercised end-to-end,
* no PyNCCL / no CPU sidecar sanity at TP=2.

Gate 4.1 must therefore stand up its own end-to-end proof.

---

## 9. Predicted first-blocker chain (Gate 4.1 hypothesis)

Ordered by earliest expected failure under a naive `torchrun
--nproc_per_node=2 python -c "from minisgl.llm import LLM; llm =
LLM('/mnt/nvme/models/Qwen3-0.6B', ...); llm.generate([prompt], sp)"`:

1. **Driver: `LLM.__init__` hardcodes `DistributedInfo(0, 1)`**
   (`llm/llm.py:32`). Both processes come up as independent TP=1
   replicas. No HCCL group is created.
   *Minimum fix (Gate 4.1 authorised):* let `LLM.__init__` accept
   `tp_info` (or `tp_rank` + `tp_size`) as a keyword argument. Default
   `(0, 1)` to preserve every existing call site. **This is the only
   `python/minisgl/` change contemplated by this gate.**
2. **Config: `use_pynccl` defaults to `True`** (`engine/config.py:29`).
   The Ascend path must set `use_pynccl=False` so
   `_init_communication` takes the `hccl + gloo sidecar` branch.
   *Handled by the bring-up script — passed as a kwarg to `LLM`, no
   engine change.*
3. **Env: `torchrun` supplies `LOCAL_RANK`/`RANK`/`WORLD_SIZE`.**
   `Engine.__init__` reads `config.tp_info.rank` directly (not the
   env). The launcher must therefore also read
   `int(os.environ["RANK"])` into the `tp_info` it hands to `LLM`.
   *Handled by the bring-up script.*
4. If (1)-(3) are addressed:
   * `bind_local_device("npu", local_rank)` binds each process to
     its own die (rank 0 → npu:0, rank 1 → npu:1).
   * `init_process_group(backend="hccl", rank=…, world_size=2,
     init_method="tcp://127.0.0.1:2333")` runs. **This is where Gate
     1's HCCL init proof stops covering — from here on every layer
     is being exercised end-to-end for the first time on the port.**
   * `_sync_get_memory` runs a `torch.distributed.all_reduce` on the
     gloo sidecar with `torch.int64` tensor.
   * `load_weight` streams safetensors, sharding into per-rank
     tensors.
   * `Qwen3Model.forward` runs `VocabParallelEmbedding` (all_reduce),
     28 decoder layers (each with QKV column-split, O row-split +
     all_reduce, gate/up column-split, down row-split + all_reduce,
     RMSNorm), and `ParallelLMHead` all_gather.
   * `Sampler.sample` runs greedy on the gathered logits.
   * Scheduler tick loops on decode until `max_new_tokens=8` completes.

Most likely places for a first real failure once (1)-(3) are fixed:

| Stage | Likely symptom | Root cause hypothesis |
|---|---|---|
| `dist.init_process_group(backend="hccl", init_method=…)` | store-side handshake timeout | `distributed_timeout` = 60 s is generous; primary risk is not this. |
| `enable_pynccl_distributed` in the `use_pynccl=True` branch | `ImportError` on `minisgl.kernel` | avoided by passing `use_pynccl=False`. |
| `VocabParallelEmbedding` all_reduce | HCCL first-collective handshake | first cross-rank collective under this port; possible hang if the gloo sidecar was not created before HCCL first-use. Guarded by `_init_communication`. |
| `ParallelLMHead` all_gather | HCCL contiguity assertion | tensors are created via `x.new_empty(shape)`; contiguous by construction. |
| Sampler determinism | greedy identity mismatch across ranks | logits are all-gathered before sample — should be bit-identical. |
| `Scheduler.shutdown` | rank-1 hang on `sync_all_ranks` barrier | both ranks must call shutdown; a bring-up script that returns from `generate` on rank 0 but leaves rank 1 blocked inside the scheduler will deadlock. Solvable by driving both ranks through the same code path. |

None of these predicted failure sites require anything larger than a
small parameter-plumbing fix. **The audit conclusion is that TP=2
Qwen3-0.6B single-request bring-up is achievable at this gate with a
single-line default relaxation in `LLM.__init__`.**

---

## 10. Direct answers to the three audit questions

**Q1. 当前代码是否已经有 TP=2 runtime path？**

Below the driver layer, **yes**:

* `Engine._init_communication` has a working TP>1 branch that on NPU
  initialises the `hccl` accelerator group and a `gloo` sidecar for
  CPU collectives.
* `load_weight` performs correct per-rank tensor sharding (Q/K/V dim
  0, O dim 1, gate/up dim 0, down dim 1, vocab dim 0).
* All TP-aware layers (`VocabParallelEmbedding`, `ParallelLMHead`,
  `LinearColumnParallel`, `LinearRowParallel`, `Qwen3Attn`,
  `RMSNormFused`) already fan into `all_reduce` / `all_gather`
  through `DistributedCommunicator`.
* `Engine._sync_get_memory` guards TP-rank memory imbalance.
* `Engine.shutdown` symmetrically destroys the process group.

At the driver layer, **no** — `LLM.__init__` hardcodes
`DistributedInfo(0, 1)`. Every existing offline call site is TP=1
because the class does not surface a TP dial.

**Q2. Gate 1 的 HCCL init 证据覆盖到哪里？**

It covers only `tests/ascend/hccl_smoke.py`:
`initialize_distributed_from_env()` at WORLD_SIZE=8 succeeds, primitive
collectives (all_reduce sum, broadcast, all_gather, barrier) return
the expected values on the raw `hccl` group, and
`destroy_process_group()` succeeds. Gate 1 verdict explicitly
disclaims TP > 1 forward / decode. HCCL init at TP=2 has never been
exercised in this repo; nor has a full model forward at any TP > 1.

**Q3. TP=2 卡住在 init、weight loading、forward、attention、sampler 还是 cleanup？**

Based on the source-only audit, with the single driver fix in
`LLM.__init__`:

* **init**: high confidence to pass — the exact
  `init_process_group(backend="hccl", init_method="tcp://127.0.0.1:2333",
  rank, world_size)` invocation is the same shape Gate 1 already
  validated, minus the WORLD_SIZE parameter.
* **weight loading**: high confidence to pass — Qwen3-0.6B's tensor
  shapes divide evenly for TP=2 and the sharding rules are already
  correct.
* **forward**: medium-high confidence — the code path is exercised
  by construction but no repo test has driven a full HCCL forward
  yet, so this is the earliest layer where a genuinely-new symptom
  is possible. Likeliest new symptom, if any: HCCL first-collective
  handshake or a per-rank device-buffer contiguity assertion in
  `all_gather`.
* **attention (`ascend_fia`)**: single-request B=1 does not touch any
  of the FIA NotImplemented boundaries.
* **sampler**: greedy on gathered logits is deterministic across
  ranks.
* **cleanup**: symmetric `synchronize_device → sync_all_ranks (CPU
  barrier) → drain → engine.shutdown` is correct by construction as
  long as both ranks reach `LLM` finalisation.

The audit therefore does not predict a BLOCKED outcome. Real
execution on 2× Ascend 910B1 will confirm the prediction; if a
verdict-worthy failure appears, it will most likely land in `forward`
and be documented as PARTIAL per the gate opening.
