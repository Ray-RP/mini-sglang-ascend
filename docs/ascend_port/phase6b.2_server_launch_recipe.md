# Phase 6B.2 Ascend Fixed-TP2 Server Launch Recipe

**Kind:** Documentation-only record that locks the minimal CLI + env
recipe for the Qwen3-0.6B fixed-TP2 server bring-up on Ascend. This
file introduces no runtime, script, test, tag, GitHub Release, or
`CHANGELOG.md` change. It does not launch the server, call `torchrun`,
curl any endpoint, or print credentials. Its sole purpose is to freeze
the launch flags so that the follow-up bring-up gate has an
unambiguous starting point.

Scope: fixed TP=2, eager, `npu_fia`, bf16, greedy — same envelope as
the v0.2.0a1 technical preview and the Phase 6B.1 inventory
([`phase6b.1_server_api_path_inventory.md`](./phase6b.1_server_api_path_inventory.md)).

---

## 1. Target

Qwen3-0.6B fixed TP=2 server bring-up.

## 2. Model

```
/mnt/nvme/models/Qwen3-0.6B
```

## 3. Required flags

```
--tp-size 2
--attention-backend npu_fia
--disable-pynccl
--cuda-graph-max-bs 0
--page-size 16
```

### Preflight

* `prompt_toolkit` must be installed because `api_server.py` imports
  it at module load. `pyproject.toml` declares it as a base runtime
  dependency; environments provisioned before that declaration was
  merged must install it explicitly (`pip install prompt_toolkit`)
  before the recipe will start. This was observed at Phase 6B.3
  bring-up on the Ascend host and is recorded here in Phase 6B.4.

Rationale (from Phase 6B.1 inventory):

* `--tp-size 2` — fixed TP=2 envelope.
* `--attention-backend npu_fia` — the only backend proven under the
  v0.2.0a1 six-case functional matrix on NPU.
* `--disable-pynccl` — `EngineConfig.use_pynccl` defaults to `True`,
  which loads a CUDA-only pynccl module inside
  `Engine._init_communication`. Without this flag the server crashes
  on NPU before HCCL init. With it, the code falls through to
  `get_distributed_backend(device_type)` which maps `npu → hccl`
  and adds a gloo sidecar.
* `--cuda-graph-max-bs 0` — the v0.2.0a1 envelope is eager
  (`cuda_graph_bs=[]`). This flag keeps the server inside that
  envelope.
* `--page-size 16` — the `npu_fia` backend routes to the CANN
  `aclnnFusedInferAttentionScoreV3` kernel. Its BF16 no-quant path
  requires `block_size` (paged-KV page size) aligned to `16`, per
  `CheckFeatureNoquantBlockSize`
  (`fused_infer_attention_score_tiling_check_feature.cpp:159`).
  `ServerArgs.page_size` defaults to `1`, which the kernel rejects
  with error `561002`
  (`In NO_QUANT situation, block_size should aligned to 16, but got 1`).
  This flag is required for the Ascend FIA path; the constraint was
  observed at Phase 6B.6 and locked in Phase 6B.7.

## 4. Required checks (to perform during real bring-up, not in this gate)

* **Server uses `mp.Process`, not `torchrun`.** `python/minisgl/server/launch.py`
  spawns `world_size` scheduler ranks via `mp.Process`. `torchrun`-style
  env vars (`RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR`,
  `MASTER_PORT`) are populated by the parent, not by an external
  launcher. The bring-up gate must confirm each child rank sees the
  expected `DistributedInfo(rank, world_size)`.
* **Verify rank / device pinning during real bring-up.** No
  per-child `ASCEND_RT_VISIBLE_DEVICES` split or explicit
  `torch.npu.set_device(local_rank)` is visible from the Phase 6B.1
  inventory reads. If both ranks end up on NPU 0, HCCL init will
  fail. Bring-up must record which physical NPUs each rank binds to.
* **Verify `distributed_addr` behaviour.** `ServerArgs.distributed_addr`
  overrides `EngineConfig.distributed_addr` and is set to
  `tcp://127.0.0.1:{server_port+1}`. `MINISGL_DISTRIBUTED_ADDR=env://`
  is **not** the mechanism in server mode. Bring-up must confirm both
  ranks rendezvous on the server-port-relative TCP address without
  needing an env-var override.
* **Verify no CUDA / NCCL fallback.** With `--disable-pynccl`, the
  code path should reach the NPU/HCCL branch of
  `_init_communication` and never touch pynccl. Bring-up must log
  that neither pynccl nor NCCL was loaded and that the resulting
  process group backend is HCCL (with a gloo sidecar).

## 5. Routes to test later (not in this gate)

* `GET /v1` — shallow health probe.
* `GET /v1/models` — model list.
* `POST /generate` — SSE streaming only.
* `POST /v1/chat/completions` with `stream=false` — non-streaming
  OpenAI-compatible completion.
* `POST /v1/chat/completions` with `stream=true` — streaming
  OpenAI-compatible completion, including client-disconnect abort-ack
  behaviour (Gate 2.3f).

## 6. Redacted launch skeleton

Placeholders only. `<PORT>` is chosen by the human operator at
bring-up time. Do **not** substitute a real port in this file.

```
python -m minisgl.server.launch \
  --model-path /mnt/nvme/models/Qwen3-0.6B \
  --tp-size 2 \
  --attention-backend npu_fia \
  --disable-pynccl \
  --cuda-graph-max-bs 0 \
  --page-size 16 \
  --host 0.0.0.0 \
  --port <PORT>
```

This skeleton is the frozen launch recipe for the Phase 6B.2
Qwen3-0.6B fixed-TP2 server bring-up. The follow-up bring-up gate
(Phase 6B.3 or equivalent) is the only party authorised to execute
it, and is responsible for recording the observed rank/device
pinning, `distributed_addr` behaviour, and absence of CUDA/NCCL
fallback per the checks in §4.
