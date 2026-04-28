"""Mimi-flavoured streaming transformer + projected transformer.

Mirrors the legacy module structure (`StreamingTransformerLayer`,
`StreamingTransformer`, `ProjectedTransformer`) so existing checkpoints load
without renames.
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn import Module
from typing_extensions import Self

from pocket_tts.modules.layer_scale import LayerScale
from pocket_tts.modules.rope import RotaryEmbedding
from pocket_tts.modules.transformer import StreamingMultiheadAttention, _LinearNoBias
from pocket_tts.utils.config import FlowLMTransformerConfig


class _LayerNorm(Module):
    """LayerNorm matching torch's default semantics (with affine, eps default 1e-5)."""

    def __init__(self, channels: int, eps: float = 1e-5, dtype: DType = DType.float32) -> None:
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.weight = Weight("weight", dtype, [channels], device=DeviceRef.CPU())
        self.bias = Weight("bias", dtype, [channels], device=DeviceRef.CPU())

    def __call__(self, x: TensorValue) -> TensorValue:
        return ops.cast(
            ops.layer_norm(
                ops.cast(x, DType.float32),
                gamma=ops.cast(self.weight, DType.float32),
                beta=ops.cast(self.bias, DType.float32),
                epsilon=self.eps,
            ),
            x.dtype,
        )


class _Identity(Module):
    def __call__(self, x: TensorValue) -> TensorValue:
        return x


class StreamingTransformerLayer(Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        context: int | None,
        rope: RotaryEmbedding,
        layer_scale: float | None = None,
        dtype: DType = DType.float32,
        device: DeviceRef | None = None,
    ):
        super().__init__()
        self._device = device or DeviceRef.CPU()
        self.self_attn = StreamingMultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            rope=rope,
            context=context,
            dtype=dtype,
            device=self._device,
        )
        self.norm1 = _LayerNorm(d_model, eps=1e-5, dtype=dtype)
        self.norm2 = _LayerNorm(d_model, eps=1e-5, dtype=dtype)
        self.linear1 = _LinearNoBias(d_model, dim_feedforward, dtype=dtype, device=self._device)
        self.linear2 = _LinearNoBias(dim_feedforward, d_model, dtype=dtype, device=self._device)
        if layer_scale is None:
            self.layer_scale_1 = _Identity()
            self.layer_scale_2 = _Identity()
        else:
            self.layer_scale_1 = LayerScale(d_model, layer_scale, dtype=dtype)
            self.layer_scale_2 = LayerScale(d_model, layer_scale, dtype=dtype)

    def _ff_block(self, x: TensorValue) -> TensorValue:
        x_orig = x
        x = self.norm2(x)
        update = self.linear2(ops.gelu(self.linear1(x)))
        return x_orig + self.layer_scale_2(update)

    def _sa_block(self, x: TensorValue, model_state) -> TensorValue:
        x_orig = x
        x = self.norm1(x)
        update = self.self_attn(x, model_state)
        return x_orig + self.layer_scale_1(update)

    def __call__(self, x: TensorValue, model_state) -> TensorValue:
        x = self._sa_block(x, model_state)
        x = self._ff_block(x)
        return x


class _LayerList(Module):
    """Equivalent of nn.ModuleList for a fixed list of homogenous layers."""

    def __init__(self, layers: list):
        super().__init__()
        self._layers = layers
        for i, layer in enumerate(layers):
            setattr(self, str(i), layer)

    def __iter__(self):
        return iter(self._layers)

    def __len__(self):
        return len(self._layers)

    def __call__(self, *args, **kwargs):  # pragma: no cover
        raise RuntimeError("Iterate over _LayerList; do not call it.")


class StreamingTransformer(Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        layer_scale: float | None = None,
        dim_feedforward: int = 2048,
        context: int | None = None,
        max_period: float = 10_000.0,
        dtype: DType = DType.float32,
        device: DeviceRef | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.max_period = max_period
        self.rope = RotaryEmbedding(max_period=max_period)
        self.layers = _LayerList(
            [
                StreamingTransformerLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    context=context,
                    rope=self.rope,
                    layer_scale=layer_scale,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(num_layers)
            ]
        )

    @classmethod
    def from_pydantic_config(
        cls,
        config: FlowLMTransformerConfig,
        dtype: DType = DType.float32,
        device: DeviceRef | None = None,
    ) -> Self:
        dim_feedforward = int(config.d_model * config.hidden_scale)
        return cls(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dim_feedforward=dim_feedforward,
            max_period=float(config.max_period),
            dtype=dtype,
            device=device,
        )

    def __call__(self, x: TensorValue, model_state) -> TensorValue:
        for layer in self.layers:
            x = layer(x, model_state)
        return x


class ProjectedTransformer(Module):
    def __init__(
        self,
        input_dimension: int,
        output_dimensions: tuple[int, ...] | list[int],
        d_model: int,
        num_heads: int,
        num_layers: int,
        layer_scale: float | None,
        context: int,
        max_period: float,
        dim_feedforward: int,
        dtype: DType = DType.float32,
        device: DeviceRef | None = None,
    ):
        super().__init__()
        self._device = device or DeviceRef.CPU()
        self.input_dimension = input_dimension
        self.output_dimensions = tuple(output_dimensions)
        self.transformer = StreamingTransformer(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            layer_scale=layer_scale,
            context=context,
            max_period=max_period,
            dim_feedforward=dim_feedforward,
            dtype=dtype,
            device=self._device,
        )
        if d_model != input_dimension:
            self.input_proj = _LinearNoBias(
                input_dimension, d_model, dtype=dtype, device=self._device
            )
        else:
            self.input_proj = None

        # output_projs is a ModuleList in the legacy code; keep the same FQNs.
        proj_layers: list = []
        for out_dim in self.output_dimensions:
            if d_model == out_dim:
                proj_layers.append(_Identity())
            else:
                proj_layers.append(
                    _LinearNoBias(d_model, out_dim, dtype=dtype, device=self._device)
                )
        self.output_projs = _LayerList(proj_layers)  # type: ignore[arg-type]

    def __call__(self, x: TensorValue, model_state) -> tuple[TensorValue, ...]:
        # Input is (B, C, T); transformer expects (B, T, C).
        x = ops.transpose(x, 1, 2)
        if self.input_proj is not None:
            x = self.input_proj(x)
        z = self.transformer(x, model_state)
        ys = []
        for output_proj in self.output_projs:
            y = output_proj(z)
            y = ops.transpose(y, 1, 2)
            ys.append(y)
        return tuple(ys)
