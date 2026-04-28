"""Flow language model on a transformer backbone, MAX implementation."""

from __future__ import annotations

import logging

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn import Module
from typing_extensions import Self

from pocket_tts.conditioners.text import LUTConditioner
from pocket_tts.modules.mimi_transformer import StreamingTransformer, _LayerNorm
from pocket_tts.modules.mlp import SimpleMLPAdaLN
from pocket_tts.modules.transformer import _LinearNoBias
from pocket_tts.utils.config import FlowLMConfig

logger = logging.getLogger(__name__)


class _LinearWithBias(Module):
    """Linear with bias, weight named `weight`, bias named `bias`."""

    def __init__(self, in_dim: int, out_dim: int, dtype: DType, device: DeviceRef) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = Weight("weight", dtype, [out_dim, in_dim], device=device)
        self.bias = Weight("bias", dtype, [out_dim], device=device)

    def __call__(self, x: TensorValue) -> TensorValue:
        return x @ ops.transpose(self.weight, 0, 1) + self.bias


class FlowLMModel(Module):
    """Transformer-based flow language model on multiple streams of latents.

    This module composes the conditioner, transformer backbone and flow MLP
    just like the legacy implementation. Method names mirror the original
    so the orchestrator code can be ported with minimal changes.
    """

    def __init__(
        self,
        conditioner: LUTConditioner,
        flow_net: SimpleMLPAdaLN,
        transformer: StreamingTransformer,
        dim: int = 128,
        ldim: int = 64,
        text_padding_weight: float = 1.0,
        dtype: DType = DType.float32,
        device: DeviceRef | None = None,
        insert_bos_before_voice: bool = False,
    ):
        super().__init__()
        self.conditioner = conditioner
        self.ldim = ldim
        self.dim = dim
        self.text_padding_weight = text_padding_weight
        self.dtype = dtype
        self._device = device or DeviceRef.CPU()
        self.insert_bos_before_voice = insert_bos_before_voice

        self.flow_net = flow_net
        # `emb_std` and `emb_mean` are torch buffers in the legacy code; here
        # they are simple Weights so the safetensors load registers them.
        self.emb_std = Weight("emb_std", dtype, [ldim], device=self._device)
        self.emb_mean = Weight("emb_mean", dtype, [ldim], device=self._device)
        self.bos_emb = Weight("bos_emb", dtype, [ldim], device=self._device)

        if insert_bos_before_voice:
            self.bos_before_voice = Weight(
                "bos_before_voice", dtype, [1, 1, dim], device=self._device
            )

        # Initialised at load time (legacy: `tts_model.flow_lm.speaker_proj_weight`).
        self.speaker_proj_weight = Weight(
            "speaker_proj_weight", dtype, [dim, dim], device=self._device
        )

        self.input_linear = _LinearNoBias(ldim, dim, dtype=dtype, device=self._device)
        self.transformer = transformer
        self.out_norm = _LayerNorm(dim, eps=1e-5, dtype=dtype)
        self.out_eos = _LinearWithBias(dim, 1, dtype=dtype, device=self._device)

    def __call__(self, *args, **kwargs):  # pragma: no cover
        raise RuntimeError("FlowLMModel is not directly callable; use backbone_run_* helpers.")

    @classmethod
    def from_pydantic_config(
        cls,
        config: FlowLMConfig,
        latent_dim: int,
        insert_bos_before_voice: bool,
        speaker_proj_in_dim: int,
        dtype: DType = DType.float32,
        device: DeviceRef | None = None,
    ) -> Self:
        d_model = config.transformer.d_model
        flow_mlp = SimpleMLPAdaLN.from_pydantic_config(
            config, latent_dim, d_model, dtype=dtype, device=device
        )
        conditioner = LUTConditioner(
            n_bins=config.lookup_table.n_bins,
            tokenizer_path=str(config.lookup_table.tokenizer_path),
            dim=config.lookup_table.dim,
            output_dim=d_model,
            dtype=dtype,
            device=device,
        )
        transformer = StreamingTransformer.from_pydantic_config(
            config.transformer, dtype=dtype, device=device
        )
        # speaker_proj_weight in the legacy code maps from
        # `mimi.inner_dim or mimi.seanet.dimension` -> `flow_lm.transformer.d_model`.
        # We reshape it on construction.
        instance = cls(
            flow_net=flow_mlp,
            transformer=transformer,
            conditioner=conditioner,
            dim=d_model,
            ldim=latent_dim,
            dtype=dtype,
            device=device,
            insert_bos_before_voice=insert_bos_before_voice,
        )
        # Resize speaker_proj_weight to (d_model, speaker_proj_in_dim).
        instance.speaker_proj_weight = Weight(
            "speaker_proj_weight", dtype, [d_model, speaker_proj_in_dim], device=instance._device
        )
        return instance

    # ------------------------------------------------------------------ #
    # Graph-level helpers used by the compiled graphs.
    # ------------------------------------------------------------------ #

    def backbone_run_prompt_text(self, text_tokens: TensorValue, model_state) -> None:
        """Run the transformer over text-token embeddings, updating kv cache.

        Returns nothing — only the kv-cache update inside `model_state` matters.
        """
        text_emb = self.conditioner.embed(text_tokens)
        # Pass through the transformer to update KV state.
        _ = self.transformer(text_emb, model_state)

    def backbone_run_prompt_audio(self, audio_cond: TensorValue, model_state) -> None:
        _ = self.transformer(audio_cond, model_state)

    def backbone_run_gen_step(
        self, backbone_input: TensorValue, model_state, noise: TensorValue, num_steps: int = 1
    ) -> tuple[TensorValue, TensorValue]:
        """One autoregressive step.

        Args:
            backbone_input: (1, 1, ldim) latent (NaN positions are replaced
                with `bos_emb`).
            model_state: kv-cache state for the transformer.
            noise: pre-sampled (1, ldim) Gaussian noise. Caller scales by
                temperature/clamp before passing in.
            num_steps: Number of LSD decode iterations (matches the legacy
                ``lsd_decode_steps`` parameter).

        Returns:
            (next_latent, eos_logit) — ``eos_logit`` is the raw logit; the
            caller applies the threshold.
        """
        is_nan = ops.is_nan(backbone_input)
        bos_b = ops.broadcast_to(self.bos_emb, shape=backbone_input.shape)
        sequence = ops.where(is_nan, bos_b, backbone_input)
        input_ = self.input_linear(sequence)
        transformer_out = self.transformer(input_, model_state)
        # The legacy backbone applies `out_norm` to the transformer output
        # *before* slicing to the last token; mirror that here.
        transformer_out = self.out_norm(transformer_out)
        transformer_out = ops.cast(transformer_out, DType.float32)
        # transformer_out shape: (1, 1, dim) — squeeze the sequence dim.
        transformer_last = ops.squeeze(transformer_out, 1)  # (1, dim)
        eos_logit = self.out_eos(transformer_last)  # (1, 1)
        next_latent = lsd_decode(self.flow_net, transformer_last, noise, num_steps=num_steps)
        return next_latent, eos_logit


def lsd_decode(
    flow_net: SimpleMLPAdaLN, cond: TensorValue, x_0: TensorValue, num_steps: int = 1
) -> TensorValue:
    """Lagrangian Self Distillation decode loop (graph-built; static ``num_steps``).

    Mirrors the legacy ``lsd_decode`` semantics:

        for i in range(num_steps):
            s, t = i / num_steps, (i + 1) / num_steps
            current += v_t(s, t, current) / num_steps
    """
    device = x_0.device
    current = x_0
    for i in range(num_steps):
        s_val = float(i) / num_steps
        t_val = float(i + 1) / num_steps
        s = ops.broadcast_to(ops.constant(s_val, x_0.dtype, device), shape=(x_0.shape[0], 1))
        t = ops.broadcast_to(ops.constant(t_val, x_0.dtype, device), shape=(x_0.shape[0], 1))
        flow_dir = flow_net(cond, s, t, current)
        scaled = flow_dir * ops.constant(1.0 / num_steps, x_0.dtype, device)
        current = current + scaled
    return current
