import torch
import torch.nn as nn
from max import nn as max_nn
from max.dtype import DType
from max.graph import DeviceRef, TensorType, Weight


class OldLayerScale(nn.Module):
    def __init__(self, channels: int, init: float):
        super().__init__()
        self.scale = nn.Parameter(torch.full((channels,), init))

    def forward(self, x: torch.Tensor):
        return self.scale * x


class LayerScale(max_nn.Module):
    def __init__(self, channels: int, init: float):
        super().__init__()
        self.channels = channels
        self.scale = Weight("scale", DType.float32, (channels,), DeviceRef.CPU())
        self.compiled_forward = None

    def forward_max(self, x: torch.Tensor):
        current = self.scale * x
        return current

    def __call__(self, x: torch.Tensor):
        if self.compiled_forward is None and isinstance(x, torch.Tensor):
            self.compiled_forward = self.compile(
                TensorType(
                    DType.float32, ("batch", self.channels, "other_dim"), device=DeviceRef.CPU()
                )
            )

        if isinstance(x, torch.Tensor):
            return self.compiled_forward(x)

        current = self.forward_max(x)
        return current
