from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, List

import torch
from minisgl.distributed import DistributedInfo
from minisgl.utils import cached_load_hf_config

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


# Gate 4.1: allow a launcher (torchrun, torch.distributed.run) to supply the
# rendezvous URI via an env var so TP>1 bring-ups can reuse the launcher's
# store instead of the loopback TCP fallback. TP=1 behaviour is unchanged
# because the env var is not set on any existing call site.
_DISTRIBUTED_ADDR_ENV = "MINISGL_DISTRIBUTED_ADDR"


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    max_running_req: int = 256
    attention_backend: str = "auto"
    moe_backend: str = "auto"
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    page_size: int = 1
    memory_ratio: float = 0.9
    distributed_timeout: float = 60.0
    use_dummy_weight: bool = False
    use_pynccl: bool = True
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages

    @cached_property
    def hf_config(self):
        return cached_load_hf_config(self.model_path)

    @cached_property
    def model_config(self) -> ModelConfig:
        from minisgl.models import ModelConfig

        return ModelConfig.from_hf(self.hf_config)

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        # Gate 4.1: honour an env-supplied rendezvous URI (e.g. ``env://`` when
        # launched under torchrun, so the launcher's own TCPStore is reused).
        # Falls back to the historical loopback URI when the env var is unset,
        # preserving every existing TP=1 offline call site (Gate 2.1 → 3.4).
        override = os.environ.get(_DISTRIBUTED_ADDR_ENV)
        if override:
            return override
        return "tcp://127.0.0.1:2333"
