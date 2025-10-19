from __future__ import annotations

from collections import OrderedDict
from typing import Dict

try:
    import torch  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("请先安装 PyTorch，命令: pip install torch") from exc


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = OrderedDict()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.detach().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            assert name in self.shadow
            self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        backup: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            backup[name] = param.data.detach().clone()
            param.data.copy_(self.shadow[name])
        return backup

    @torch.no_grad()
    def restore(self, model: torch.nn.Module, backup: Dict[str, torch.Tensor]) -> None:
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])
