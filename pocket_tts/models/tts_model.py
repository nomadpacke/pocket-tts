"""Top-level TTSModel orchestrator using compiled MAX graphs.

This module preserves the public API exposed by the legacy torch
implementation (``TTSModel.load_model``, ``generate_audio``,
``generate_audio_stream``, ``get_state_for_audio_prompt``) while running
all heavy compute in compiled MAX graphs.

Compiled graphs:
- ``flow_lm_text_prompt(text_tokens, *kv_state) -> *kv_state``
- ``flow_lm_audio_prompt(audio_cond, *kv_state) -> *kv_state``
- ``flow_lm_gen_step(backbone_in, *kv_state, noise) -> (next_latent, eos_logit, *kv_state)``
- ``mimi_encode(audio) -> latent``
- ``mimi_decode(latent, *mimi_state) -> (audio_chunk, *mimi_state)``

State is held as a flat dict ``{module_fqn: {key: numpy_array}}`` and
flattened to/from a tuple of arrays around each compiled-graph call.
"""

from __future__ import annotations

import copy
import logging
import math
import os
import queue
import statistics
import threading
import time
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

import numpy as np
import safetensors
import safetensors.numpy
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Dim, Graph, TensorType
from max.nn.layer.layer import recursive_named_layers
from typing_extensions import Self

from pocket_tts.conditioners.base import TokenizedText
from pocket_tts.data.audio import audio_read
from pocket_tts.data.audio_utils import convert_audio
from pocket_tts.default_parameters import (
    DEFAULT_EOS_THRESHOLD,
    DEFAULT_LANGUAGE,
    DEFAULT_LSD_DECODE_STEPS,
    DEFAULT_NOISE_CLAMP,
    DEFAULT_TEMPERATURE,
    MAX_TOKEN_PER_CHUNK,
)
from pocket_tts.models.flow_lm import FlowLMModel
from pocket_tts.models.mimi import MimiModel
from pocket_tts.modules.conv import pad_for_conv1d_numpy
from pocket_tts.modules.dummy_quantizer import DummyQuantizer
from pocket_tts.modules.mimi_transformer import ProjectedTransformer
from pocket_tts.modules.seanet import SEANetDecoder, SEANetEncoder
from pocket_tts.modules.stateful_module import (
    StatefulModule,
    assign_module_absolute_names,
    increment_steps,
    init_states,
)
from pocket_tts.utils.config import CONFIGS_DIR, Config, load_config
from pocket_tts.utils.utils import (
    _ORIGINS_OF_PREDEFINED_VOICES,
    display_execution_time,
    download_if_necessary,
    get_predefined_voice,
    size_of_dict,
)
from pocket_tts.utils.weights_loading import (
    get_flow_lm_state_dict,
    get_mimi_state_dict,
    load_top_level_safetensors,
)

logger = logging.getLogger(__name__)

VOICE_CLONING_UNSUPPORTED = (
    f"We could not download the weights for the model with voice cloning, "
    f"but you're trying to use voice cloning. "
    f"Without voice cloning, you can use our catalog of voices "
    f"{list(_ORIGINS_OF_PREDEFINED_VOICES.keys())}. "
    f"If you want access to the model with voice cloning, go to "
    f"https://huggingface.co/kyutai/pocket-tts and accept the terms, "
    f"then make sure you're logged in locally with `uvx hf auth login`."
)


# ----------------------------------------------------------------------------
# State flattening / unflattening
# ----------------------------------------------------------------------------


def _state_schema(model) -> list[tuple[str, str]]:
    """Return a stable ordering of (module_fqn, state_key) for `model`."""
    schema: list[tuple[str, str]] = []
    for name, module in recursive_named_layers(model):
        if not isinstance(module, StatefulModule):
            continue
        sample = module.init_state(batch_size=1, sequence_length=0)
        for key in sample.keys():
            schema.append((name, key))
    return schema


def _state_to_tuple(schema, state):
    return tuple(state[fqn][key] for fqn, key in schema)


def _tuple_to_state(schema, tensors):
    out: dict = {}
    for (fqn, key), value in zip(schema, tensors):
        out.setdefault(fqn, {})[key] = value
    return out


# ----------------------------------------------------------------------------
# Graph builders
# ----------------------------------------------------------------------------


def _state_input_types(
    schema: list[tuple[str, str]],
    init_state: dict[str, dict[str, np.ndarray]],
    seq_dynamic_keys: dict[str, set[str]] | None = None,
    device: DeviceRef | None = None,
) -> list[TensorType]:
    """Build TensorType list for graph inputs corresponding to state.

    `seq_dynamic_keys` maps fqn -> set of state keys whose first non-batch
    dim should be made dynamic (e.g. KV-cache time dim). Keys belonging to
    the same FQN share a single Dim so the type checker can prove that K and
    V time dims stay in sync.
    """
    seq_dynamic_keys = seq_dynamic_keys or {}
    device = device or DeviceRef.CPU()
    types: list[TensorType] = []
    fqn_dim_cache: dict[str, Dim] = {}
    for fqn, key in schema:
        sample = init_state[fqn][key]
        shape: list = list(sample.shape)
        if key in seq_dynamic_keys.get(fqn, set()):
            if fqn not in fqn_dim_cache:
                sanitized = (fqn + "_t").replace(".", "_")
                fqn_dim_cache[fqn] = Dim(sanitized)
            shape[1] = fqn_dim_cache[fqn]
        dtype = DType.from_numpy(sample.dtype)
        types.append(TensorType(dtype, shape=shape, device=device))
    return types


def _build_kv_state_dynamic_keys(model) -> dict[str, set[str]]:
    """Mark `k`/`v` keys of every StreamingMultiheadAttention as dynamic
    along their time dim (axis 1)."""
    from pocket_tts.modules.transformer import StreamingMultiheadAttention

    out: dict[str, set[str]] = {}
    for name, module in recursive_named_layers(model):
        if isinstance(module, StreamingMultiheadAttention):
            out[name] = {"k", "v"}
    return out


# ----------------------------------------------------------------------------
# TTS model
# ----------------------------------------------------------------------------


class TTSModel:
    _TOKENS_PER_SECOND_ESTIMATE = 3.0
    _GEN_SECONDS_PADDING = 2.0

    def __init__(
        self,
        flow_lm: FlowLMModel,
        mimi: MimiModel,
        temp: float,
        lsd_decode_steps: int,
        noise_clamp: float | None,
        eos_threshold: float,
        config: Config,
        origin: Path | None = None,
        pad_with_spaces_for_short_inputs: bool = False,
        model_recommended_frames_after_eos: int | None = None,
        remove_semicolons: bool = False,
    ):
        self.flow_lm = flow_lm
        self.mimi = mimi
        self.temp = float(temp)
        self.lsd_decode_steps = int(lsd_decode_steps)
        self.noise_clamp = noise_clamp
        self.eos_threshold = float(eos_threshold)
        self.config = config
        self.has_voice_cloning = True
        self.origin = origin
        self.pad_with_spaces_for_short_inputs = pad_with_spaces_for_short_inputs
        self.model_recommended_frames_after_eos = model_recommended_frames_after_eos
        self.remove_semicolons = remove_semicolons

        # --- Resolve module FQN names so state lookups work. -----------------
        assign_module_absolute_names(self.flow_lm)
        assign_module_absolute_names(self.mimi)

        # --- Schemas (deterministic state ordering for graph IO). ------------
        self._flow_lm_schema = _state_schema(self.flow_lm)
        self._mimi_schema = _state_schema(self.mimi)

        # --- Lazy-built compiled graphs. -------------------------------------
        self._session: InferenceSession | None = None
        self._flow_lm_text_prompt = None
        self._flow_lm_audio_prompt = None
        self._flow_lm_gen_step = None
        self._mimi_encode = None
        self._mimi_decode = None

        if self.lsd_decode_steps < 1:
            raise ValueError(f"lsd_decode_steps must be >= 1; got {self.lsd_decode_steps}")

        # Pre-cached weights registries (numpy arrays) for graph compilation.
        self._flow_lm_weights = self.flow_lm.state_dict()
        self._mimi_weights = self.mimi.state_dict()

        # The encoder pad-multiple, used in audio prompt encoding.
        self._mimi_frame_size = self.mimi.frame_size

    # ------------------------------------------------------------------ #
    # Properties matching the legacy public API.
    # ------------------------------------------------------------------ #

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def sample_rate(self) -> int:
        return self.config.mimi.sample_rate

    # ------------------------------------------------------------------ #
    # Construction.
    # ------------------------------------------------------------------ #

    @classmethod
    def _from_pydantic_config(
        cls,
        config: Config,
        temp,
        lsd_decode_steps,
        noise_clamp: float | None,
        eos_threshold,
        origin: Path | None,
    ) -> Self:
        device = DeviceRef.CPU()
        dtype = DType.float32

        flow_lm = FlowLMModel.from_pydantic_config(
            config.flow_lm,
            latent_dim=config.mimi.quantizer.dimension,
            insert_bos_before_voice=config.flow_lm.insert_bos_before_voice,
            speaker_proj_in_dim=config.mimi.inner_dim or config.mimi.seanet.dimension,
            dtype=dtype,
            device=device,
        )

        mimi_config = config.mimi.model_dump()
        encoder = SEANetEncoder(**mimi_config["seanet"], dtype=dtype, device=device)
        decoder = SEANetDecoder(**mimi_config["seanet"], dtype=dtype, device=device)
        encoder_transformer = ProjectedTransformer(
            **mimi_config["transformer"], dtype=dtype, device=device
        )
        decoder_transformer = ProjectedTransformer(
            **mimi_config["transformer"], dtype=dtype, device=device
        )
        quantizer = DummyQuantizer(
            dimension=mimi_config["quantizer"]["dimension"],
            output_dimension=mimi_config["quantizer"]["output_dimension"],
            dtype=dtype,
            device=device,
        )
        mimi = MimiModel(
            encoder=encoder,
            decoder=decoder,
            quantizer=quantizer,
            channels=mimi_config["channels"],
            sample_rate=mimi_config["sample_rate"],
            frame_rate=mimi_config["frame_rate"],
            encoder_frame_rate=mimi_config["sample_rate"] / encoder.hop_length,
            inner_dim=mimi_config["inner_dim"],
            outer_dim=mimi_config["outer_dim"],
            encoder_transformer=encoder_transformer,
            decoder_transformer=decoder_transformer,
            dtype=dtype,
            device=device,
        )

        return cls(
            flow_lm=flow_lm,
            mimi=mimi,
            temp=temp,
            lsd_decode_steps=lsd_decode_steps,
            noise_clamp=noise_clamp,
            eos_threshold=eos_threshold,
            config=config,
            origin=origin,
            pad_with_spaces_for_short_inputs=config.pad_with_spaces_for_short_inputs,
            model_recommended_frames_after_eos=config.model_recommended_frames_after_eos,
            remove_semicolons=config.remove_semicolons,
        )

    @classmethod
    def _from_pydantic_config_with_weights(
        cls,
        config: Config,
        temp,
        lsd_decode_steps,
        noise_clamp: float | None,
        eos_threshold,
        origin: Path | None = None,
    ) -> Self:
        tts = cls._from_pydantic_config(
            config, temp, lsd_decode_steps, noise_clamp, eos_threshold, origin=origin
        )

        # Load FlowLM weights.
        if config.flow_lm.weights_path is not None:
            if config.mimi.weights_path is None:
                raise ValueError(
                    "If you specify flow_lm.weights_path you should specify mimi.weights_path"
                )
            logger.info("Loading FlowLM weights from %s", config.flow_lm.weights_path)
            state_dict_flowlm = get_flow_lm_state_dict(
                download_if_necessary(config.flow_lm.weights_path)
            )
            tts.flow_lm.load_state_dict(state_dict_flowlm, strict=True)

        # Load Mimi weights.
        if config.mimi.weights_path is not None:
            if config.flow_lm.weights_path is None:
                raise ValueError(
                    "If you specify mimi.weights_path you should specify flow_lm.weights_path"
                )
            logger.info("Loading Mimi weights from %s", config.mimi.weights_path)
            mimi_state = get_mimi_state_dict(download_if_necessary(config.mimi.weights_path))
            tts.mimi.load_state_dict(mimi_state, strict=True)

        # Load combined TTSModel weights (flow_lm.* and mimi.*).
        if config.weights_path is not None:
            logger.info("Loading TTSModel weights from %s", config.weights_path)
            try:
                weights_file = download_if_necessary(config.weights_path)
            except Exception:
                tts.has_voice_cloning = False
                weights_file = download_if_necessary(config.weights_path_without_voice_cloning)
            top_state = load_top_level_safetensors(weights_file)
            flow_lm_state = {}
            mimi_state = {}
            for key, value in top_state.items():
                # `time_embed.*.freqs` were torch buffers in the legacy code;
                # we recompute them inline in the graph so drop them here.
                if ".freqs" in key:
                    continue
                if key.startswith("flow_lm."):
                    flow_lm_state[key[len("flow_lm.") :]] = value
                elif key.startswith("mimi."):
                    mimi_state[key[len("mimi.") :]] = value
                else:
                    raise ValueError(f"Unexpected weight key: {key}")
            tts.flow_lm.load_state_dict(flow_lm_state, strict=True)
            tts.mimi.load_state_dict(mimi_state, strict=True)

        if config.flow_lm.weights_path is None and config.weights_path is None:
            logger.warning(
                "No weights_path specified for FlowLM or TTSModel, model is uninitialized!"
            )

        # Refresh cached weight registries.
        tts._flow_lm_weights = tts.flow_lm.state_dict()
        tts._mimi_weights = tts.mimi.state_dict()

        size_in_mb = (
            size_of_dict(tts._flow_lm_weights) + size_of_dict(tts._mimi_weights)
        ) // 1_000_000
        logger.info("TTS Model loaded successfully. Its size is %d MB", size_in_mb)

        return tts

    @classmethod
    def load_model(
        cls,
        language: str | None = None,
        config: str | Path | None = None,
        temp: float | int = DEFAULT_TEMPERATURE,
        lsd_decode_steps: int = DEFAULT_LSD_DECODE_STEPS,
        noise_clamp: float | int | None = DEFAULT_NOISE_CLAMP,
        eos_threshold: float = DEFAULT_EOS_THRESHOLD,
        quantize: bool = False,
    ) -> Self:
        if config is not None and language is not None:
            raise ValueError(
                "Cannot specify both config and language, please choose one or the other."
            )
        if config is None and language is None:
            language = DEFAULT_LANGUAGE
        if language is not None:
            if language == "french":
                raise ValueError(
                    "The french model is not ready yet, please use 'french_24l' instead."
                )
            config = CONFIGS_DIR / f"{language}.yaml"
        config = Path(config)
        if config.suffix not in (".yaml", ".yml"):
            raise ValueError("Config should be a path to a YAML file ending with .yaml")
        config_path = Path(config)
        config = load_config(config_path)
        logger.info("Loading model from config at %s...", config_path)

        tts = cls._from_pydantic_config_with_weights(
            config, temp, lsd_decode_steps, noise_clamp, eos_threshold, origin=config_path
        )

        if quantize:
            from pocket_tts.quantization import RECOMMENDED_CONFIG, apply_dynamic_int8

            apply_dynamic_int8(tts.flow_lm, RECOMMENDED_CONFIG)

        return tts

    # ------------------------------------------------------------------ #
    # Graph compilation.
    # ------------------------------------------------------------------ #

    def _ensure_session(self) -> InferenceSession:
        if self._session is None:
            self._session = InferenceSession(devices=[CPU()])
        return self._session

    def _ensure_compiled(self) -> None:
        if self._flow_lm_text_prompt is None:
            self._compile_all()

    @staticmethod
    def _reset_weight_caches(*models) -> None:
        """Clear the cached `_mlir_value` on every Weight in `models`.

        :class:`max.graph.Weight._mlir_value` is a ``cached_property`` that
        binds the weight to the graph in which it was first used. Reusing the
        same module across multiple compiled graphs requires clearing this
        cache between builds.
        """
        from max.graph import Weight

        for model in models:
            for _, module in recursive_named_layers(model):
                for attr_name in list(getattr(module, "_layer_weights", {}).keys()):
                    weight = module._layer_weights.get(attr_name)
                    if isinstance(weight, Weight):
                        weight.__dict__.pop("_mlir_value", None)

    def _compile_all(self) -> None:
        sess = self._ensure_session()

        device = DeviceRef.CPU()
        flow_lm = self.flow_lm
        mimi = self.mimi

        zero_state_flow = init_states(flow_lm, batch_size=1, sequence_length=0)
        zero_state_mimi = init_states(mimi, batch_size=1, sequence_length=0)

        flow_dynamic_keys = _build_kv_state_dynamic_keys(flow_lm)
        mimi_dynamic_keys = _build_kv_state_dynamic_keys(mimi)

        flow_state_in_types = _state_input_types(
            self._flow_lm_schema, zero_state_flow, flow_dynamic_keys, device
        )
        mimi_state_in_types = _state_input_types(
            self._mimi_schema, zero_state_mimi, mimi_dynamic_keys, device
        )

        # --- flow_lm_text_prompt graph -----------------------------------
        text_input_type = TensorType(DType.int64, shape=[1, Dim("text_t")], device=device)

        def _text_prompt_fn(text_tokens, *state_inputs):
            state = _tuple_to_state(self._flow_lm_schema, state_inputs)
            flow_lm.backbone_run_prompt_text(text_tokens, state)
            return _state_to_tuple(self._flow_lm_schema, state)

        self._reset_weight_caches(flow_lm, mimi)
        text_graph = Graph(
            "flow_lm_text_prompt",
            _text_prompt_fn,
            input_types=[text_input_type, *flow_state_in_types],
        )
        with display_execution_time("Compile flow_lm_text_prompt"):
            self._flow_lm_text_prompt = sess.load(
                text_graph, weights_registry=self._flow_lm_weights
            )

        # --- flow_lm_audio_prompt graph -----------------------------------
        audio_cond_type = TensorType(
            DType.float32, shape=[1, Dim("audio_t"), self.flow_lm.dim], device=device
        )

        def _audio_prompt_fn(audio_cond, *state_inputs):
            state = _tuple_to_state(self._flow_lm_schema, state_inputs)
            flow_lm.backbone_run_prompt_audio(audio_cond, state)
            return _state_to_tuple(self._flow_lm_schema, state)

        self._reset_weight_caches(flow_lm, mimi)
        audio_graph = Graph(
            "flow_lm_audio_prompt",
            _audio_prompt_fn,
            input_types=[audio_cond_type, *flow_state_in_types],
        )
        with display_execution_time("Compile flow_lm_audio_prompt"):
            self._flow_lm_audio_prompt = sess.load(
                audio_graph, weights_registry=self._flow_lm_weights
            )

        # --- flow_lm_gen_step graph ---------------------------------------
        gen_input_type = TensorType(DType.float32, shape=[1, 1, self.flow_lm.ldim], device=device)
        noise_type = TensorType(DType.float32, shape=[1, self.flow_lm.ldim], device=device)

        gen_num_steps = self.lsd_decode_steps

        def _gen_step_fn(backbone_input, *args):
            *state_inputs, noise = args
            state = _tuple_to_state(self._flow_lm_schema, state_inputs)
            next_latent, eos_logit = flow_lm.backbone_run_gen_step(
                backbone_input, state, noise, num_steps=gen_num_steps
            )
            return (next_latent, eos_logit, *_state_to_tuple(self._flow_lm_schema, state))

        self._reset_weight_caches(flow_lm, mimi)
        gen_graph = Graph(
            "flow_lm_gen_step",
            _gen_step_fn,
            input_types=[gen_input_type, *flow_state_in_types, noise_type],
        )
        with display_execution_time("Compile flow_lm_gen_step"):
            self._flow_lm_gen_step = sess.load(gen_graph, weights_registry=self._flow_lm_weights)

        # --- mimi_encode graph --------------------------------------------
        # encoder works on the full audio at once; no streaming state.
        audio_in_type = TensorType(
            DType.float32, shape=[1, mimi.channels, Dim("aud_t")], device=device
        )

        def _mimi_encode_fn(audio):
            return mimi.encode_to_latent(audio)

        self._reset_weight_caches(flow_lm, mimi)
        mimi_enc_graph = Graph("mimi_encode", _mimi_encode_fn, input_types=[audio_in_type])
        with display_execution_time("Compile mimi_encode"):
            self._mimi_encode = sess.load(mimi_enc_graph, weights_registry=self._mimi_weights)

        # --- mimi_decode graph --------------------------------------------
        latent_in_type = TensorType(
            DType.float32,
            shape=[1, self.config.mimi.quantizer.dimension, Dim("lat_t")],
            device=device,
        )

        def _mimi_decode_fn(latent, *state_inputs):
            state = _tuple_to_state(self._mimi_schema, state_inputs)
            # Quantizer projects latent (1, q_dim, T) -> (1, dim, T).
            quantized = mimi.quantizer(latent)
            audio = mimi.decode_from_latent(quantized, state)
            return (audio, *_state_to_tuple(self._mimi_schema, state))

        self._reset_weight_caches(flow_lm, mimi)
        mimi_dec_graph = Graph(
            "mimi_decode", _mimi_decode_fn, input_types=[latent_in_type, *mimi_state_in_types]
        )
        with display_execution_time("Compile mimi_decode"):
            self._mimi_decode = sess.load(mimi_dec_graph, weights_registry=self._mimi_weights)

    # ------------------------------------------------------------------ #
    # Compiled-graph wrappers.
    # ------------------------------------------------------------------ #

    def _run_flow_lm_text_prompt(
        self, text_tokens: np.ndarray, state: dict[str, dict[str, np.ndarray]]
    ) -> dict[str, dict[str, np.ndarray]]:
        self._ensure_compiled()
        flat_state = _state_to_tuple(self._flow_lm_schema, state)
        outputs = self._flow_lm_text_prompt(text_tokens, *flat_state)
        new_state = _tuple_to_state(self._flow_lm_schema, tuple(t.to_numpy() for t in outputs))
        increment_steps(self.flow_lm, new_state, increment=int(text_tokens.shape[1]))
        return new_state

    def _run_flow_lm_audio_prompt(
        self, audio_cond: np.ndarray, state: dict[str, dict[str, np.ndarray]]
    ) -> dict[str, dict[str, np.ndarray]]:
        self._ensure_compiled()
        flat_state = _state_to_tuple(self._flow_lm_schema, state)
        outputs = self._flow_lm_audio_prompt(audio_cond, *flat_state)
        new_state = _tuple_to_state(self._flow_lm_schema, tuple(t.to_numpy() for t in outputs))
        increment_steps(self.flow_lm, new_state, increment=int(audio_cond.shape[1]))
        return new_state

    def _run_flow_lm_gen_step(
        self, backbone_in: np.ndarray, state: dict[str, dict[str, np.ndarray]], noise: np.ndarray
    ) -> tuple[np.ndarray, bool, dict[str, dict[str, np.ndarray]]]:
        self._ensure_compiled()
        flat_state = _state_to_tuple(self._flow_lm_schema, state)
        outputs = self._flow_lm_gen_step(backbone_in, *flat_state, noise)
        next_latent = outputs[0].to_numpy()
        eos_logit = outputs[1].to_numpy()
        new_state_flat = tuple(t.to_numpy() for t in outputs[2:])
        new_state = _tuple_to_state(self._flow_lm_schema, new_state_flat)
        increment_steps(self.flow_lm, new_state, increment=int(backbone_in.shape[1]))
        is_eos = bool(eos_logit.flatten()[0] > self.eos_threshold)
        return next_latent, is_eos, new_state

    def _run_mimi_encode(self, audio: np.ndarray) -> np.ndarray:
        self._ensure_compiled()
        outputs = self._mimi_encode(audio)
        return outputs[0].to_numpy()

    def _run_mimi_decode(
        self, latent: np.ndarray, state: dict[str, dict[str, np.ndarray]]
    ) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
        self._ensure_compiled()
        flat_state = _state_to_tuple(self._mimi_schema, state)
        outputs = self._mimi_decode(latent, *flat_state)
        audio = outputs[0].to_numpy()
        new_state_flat = tuple(t.to_numpy() for t in outputs[1:])
        new_state = _tuple_to_state(self._mimi_schema, new_state_flat)
        return audio, new_state

    # ------------------------------------------------------------------ #
    # Audio-prompt encoding (voice state).
    # ------------------------------------------------------------------ #

    def _encode_audio(self, audio: np.ndarray) -> np.ndarray:
        """Encode an audio prompt into a (B, T_frames, dim) conditioning."""
        # Pad to a multiple of frame_size for the encoder.
        x = pad_for_conv1d_numpy(audio, self._mimi_frame_size, self._mimi_frame_size)
        encoded = self._run_mimi_encode(x)  # (1, q_dim, T_frames)
        # Move the channel dim last and project through speaker_proj_weight.
        latents = np.transpose(encoded, (0, 2, 1)).astype(np.float32)
        # F.linear(latents, speaker_proj_weight) = latents @ W.T
        spw = np.asarray(self._flow_lm_weights["speaker_proj_weight"])
        return latents @ spw.T

    # ------------------------------------------------------------------ #
    # Public API: state for audio prompt.
    # ------------------------------------------------------------------ #

    def _flow_lm_current_end(self, model_state: dict) -> int:
        for module_state in model_state.values():
            offset = module_state.get("offset")
            if offset is not None:
                return int(np.asarray(offset).reshape(-1)[0])
        raise ValueError("Could not find offset in model state; please open an issue.")

    @lru_cache(maxsize=2)
    def _cached_get_state_for_audio_prompt(
        self, audio_conditioning, truncate: bool = False
    ) -> dict:
        return self.get_state_for_audio_prompt(audio_conditioning, truncate)

    def get_state_for_audio_prompt(
        self, audio_conditioning: Path | str | np.ndarray, truncate: bool = False
    ) -> dict[str, dict[str, np.ndarray]]:
        if isinstance(audio_conditioning, (str, Path)) and str(audio_conditioning).endswith(
            ".safetensors"
        ):
            if isinstance(audio_conditioning, str):
                audio_conditioning = download_if_necessary(audio_conditioning)
            return _import_model_state(audio_conditioning, schema=self._flow_lm_schema)
        elif (
            isinstance(audio_conditioning, str)
            and audio_conditioning in _ORIGINS_OF_PREDEFINED_VOICES
        ):
            if self.origin is None or not self.origin.is_relative_to(CONFIGS_DIR):
                raise ValueError(
                    f"Cannot use predefined voices when the model "
                    f"is not loaded from a config associated with a language. "
                    f"Here the origin is {self.origin}"
                )
            return _import_model_state(
                download_if_necessary(
                    get_predefined_voice(language=self.origin.stem, name=audio_conditioning)
                ),
                schema=self._flow_lm_schema,
            )

        if not self.has_voice_cloning and isinstance(audio_conditioning, (str, Path)):
            raise ValueError(VOICE_CLONING_UNSUPPORTED)

        if isinstance(audio_conditioning, str):
            audio_conditioning = download_if_necessary(audio_conditioning)

        if isinstance(audio_conditioning, Path):
            audio, conditioning_sample_rate = audio_read(audio_conditioning)

            if truncate:
                max_samples = int(30 * conditioning_sample_rate)
                if audio.shape[-1] > max_samples:
                    audio = audio[..., :max_samples]
                    logger.info("Audio truncated to first 30 seconds (%d samples)", max_samples)

            audio_conditioning = convert_audio(
                audio, conditioning_sample_rate, self.config.mimi.sample_rate, 1
            )

        # audio_conditioning shape: (1, T) -> (1, 1, T)
        if audio_conditioning.ndim == 2:
            audio_conditioning = audio_conditioning[:, None, :]

        with display_execution_time("Encoding audio prompt"):
            prompt = self._encode_audio(audio_conditioning.astype(np.float32))

        if self.flow_lm.insert_bos_before_voice:
            bos = self._flow_lm_weights["bos_before_voice"]
            bos = np.asarray(bos, dtype=np.float32)
            prompt = np.concatenate([bos, prompt], axis=1)

        model_state = init_states(self.flow_lm, batch_size=1, sequence_length=prompt.shape[1])

        with display_execution_time("Prompting audio"):
            model_state = self._run_flow_lm_audio_prompt(prompt, model_state)

        logger.info(
            "Size of the model state for audio prompt: %d MB",
            size_of_dict(model_state) // 1_000_000,
        )
        return model_state

    # ------------------------------------------------------------------ #
    # Generation.
    # ------------------------------------------------------------------ #

    def _estimate_max_gen_len(self, token_count: int) -> int:
        gen_len_sec = token_count / self._TOKENS_PER_SECOND_ESTIMATE + self._GEN_SECONDS_PADDING
        frame_rate = self.config.mimi.frame_rate
        return math.ceil(gen_len_sec * frame_rate)

    def _decode_audio_worker(
        self,
        latents_queue: queue.Queue,
        result_queue: queue.Queue,
        mimi_steps_per_latent: int,
        flow_emb_std: np.ndarray,
        flow_emb_mean: np.ndarray,
    ):
        try:
            mimi_state = init_states(self.mimi, batch_size=1, sequence_length=0)
            while True:
                latent = latents_queue.get()
                if latent is None:
                    break
                # Apply flow normalisation.
                mimi_decoding_input = latent * flow_emb_std + flow_emb_mean
                # latent: (1, 1, ldim) -> (1, ldim, 1)
                transposed = np.transpose(mimi_decoding_input, (0, 2, 1)).astype(np.float32)
                t = time.monotonic()
                audio_frame, mimi_state = self._run_mimi_decode(transposed, mimi_state)
                # increment mimi state offsets if any.
                increment_steps(self.mimi, mimi_state, increment=mimi_steps_per_latent)
                audio_frame_duration = audio_frame.shape[2] / self.config.mimi.sample_rate
                logger.debug(
                    " " * 30 + "Decoded %d ms of audio with mimi in %d ms",
                    int(audio_frame_duration * 1000),
                    int((time.monotonic() - t) * 1000),
                )
                result_queue.put(("chunk", audio_frame))
                latents_queue.task_done()
            result_queue.put(("done", None))
        except Exception as e:  # pragma: no cover
            result_queue.put(("error", e))

    def generate_audio(
        self,
        model_state: dict,
        text_to_generate: str,
        max_tokens: int = MAX_TOKEN_PER_CHUNK,
        frames_after_eos: int | None = None,
        copy_state: bool = True,
    ) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for chunk in self.generate_audio_stream(
            model_state=model_state,
            text_to_generate=text_to_generate,
            frames_after_eos=frames_after_eos,
            copy_state=copy_state,
            max_tokens=max_tokens,
        ):
            chunks.append(chunk)
        return np.concatenate(chunks, axis=0)

    def generate_audio_stream(
        self,
        model_state: dict,
        text_to_generate: str,
        max_tokens: int = MAX_TOKEN_PER_CHUNK,
        frames_after_eos: int | None = None,
        copy_state: bool = True,
    ) -> Iterator[np.ndarray]:
        if frames_after_eos is None:
            frames_after_eos = self.model_recommended_frames_after_eos

        chunks = split_into_best_sentences(
            self.flow_lm.conditioner.tokenizer,
            text_to_generate,
            max_tokens,
            self.pad_with_spaces_for_short_inputs,
            remove_semicolons=self.remove_semicolons,
        )

        for chunk in chunks:
            text_to_generate, frames_after_eos_guess = prepare_text_prompt(
                chunk, self.pad_with_spaces_for_short_inputs, self.remove_semicolons
            )
            frames_after_eos_guess += 2
            effective_frames = (
                frames_after_eos if frames_after_eos is not None else frames_after_eos_guess
            )
            yield from self._generate_audio_stream_short_text(
                model_state=model_state,
                text_to_generate=chunk,
                frames_after_eos=effective_frames,
                copy_state=copy_state,
            )

    def _generate_audio_stream_short_text(
        self, model_state: dict, text_to_generate: str, frames_after_eos: int, copy_state: bool
    ) -> Iterator[np.ndarray]:
        if copy_state:
            model_state = copy.deepcopy(model_state)

        prepared = self.flow_lm.conditioner.prepare(text_to_generate)
        token_count = prepared.tokens.shape[1]
        max_gen_len = self._estimate_max_gen_len(token_count)
        mimi_steps_per_latent = int(self.mimi.encoder_frame_rate / self.mimi.frame_rate)

        flow_emb_std = self._flow_lm_weights["emb_std"]
        flow_emb_mean = self._flow_lm_weights["emb_mean"]

        latents_queue: queue.Queue = queue.Queue()
        result_queue: queue.Queue = queue.Queue()

        decoder_thread = threading.Thread(
            target=self._decode_audio_worker,
            args=(latents_queue, result_queue, mimi_steps_per_latent, flow_emb_std, flow_emb_mean),
            daemon=True,
        )
        logger.info("starting timer now!")
        t_generating = time.monotonic()
        decoder_thread.start()

        self._generate(
            model_state=model_state,
            prepared=prepared,
            max_gen_len=max_gen_len,
            frames_after_eos=frames_after_eos,
            latents_queue=latents_queue,
            result_queue=result_queue,
        )

        total_generated_samples = 0
        while True:
            result = result_queue.get()
            if result[0] == "chunk":
                audio_chunk = result[1]
                total_generated_samples += audio_chunk.shape[-1]
                yield audio_chunk[0, 0]  # remove batch+channel
            elif result[0] == "done":
                break
            elif result[0] == "error":
                with display_execution_time("Waiting for mimi decoder to finish"):
                    decoder_thread.join()
                raise result[1]

        with display_execution_time("Waiting for mimi decoder to finish"):
            decoder_thread.join()

        duration_generated_audio = int(
            total_generated_samples * 1000 / self.config.mimi.sample_rate
        )
        generation_time = int((time.monotonic() - t_generating) * 1000)
        if generation_time > 0:
            real_time_factor = duration_generated_audio / generation_time
        else:
            real_time_factor = 0.0
        logger.info(
            "Generated: %d ms of audio in %d ms so %.2fx faster than real-time",
            duration_generated_audio,
            generation_time,
            real_time_factor,
        )

    def _generate(
        self,
        model_state: dict,
        prepared: TokenizedText,
        max_gen_len: int,
        frames_after_eos: int,
        latents_queue: queue.Queue,
        result_queue: queue.Queue,
    ):
        with display_execution_time("Prompting text"):
            model_state.update(self._run_flow_lm_text_prompt(prepared.tokens, model_state))

        def run_generation():
            try:
                self._autoregressive_generation(
                    model_state, max_gen_len, frames_after_eos, latents_queue
                )
            except Exception as e:  # pragma: no cover
                logger.error("Error in autoregressive generation: %s", e)
                if latents_queue is not None:
                    latents_queue.put(None)
                if result_queue is not None:
                    result_queue.put(("error", e))

        generation_thread = threading.Thread(target=run_generation, daemon=True)
        generation_thread.start()

    def _autoregressive_generation(
        self, model_state: dict, max_gen_len: int, frames_after_eos: int, latents_queue: queue.Queue
    ):
        backbone_input = np.full((1, 1, self.flow_lm.ldim), fill_value=np.nan, dtype=np.float32)
        steps_times: list[int] = []
        eos_step: int | None = None
        rng = np.random.default_rng()
        std = self.temp**0.5
        for generation_step in range(max_gen_len):
            with display_execution_time("Generating latent", print_output=False) as timer:
                noise = self._sample_noise(rng, std)
                next_latent, is_eos, new_state = self._run_flow_lm_gen_step(
                    backbone_input, model_state, noise
                )
                model_state.clear()
                model_state.update(new_state)
                if is_eos and eos_step is None:
                    eos_step = generation_step
                if eos_step is not None and generation_step >= eos_step + frames_after_eos:
                    break
                # next_latent: (1, ldim) -> reshape to (1, 1, ldim)
                next_latent_3d = next_latent.reshape(1, 1, self.flow_lm.ldim)
                latents_queue.put(next_latent_3d)
                backbone_input = next_latent_3d
            steps_times.append(timer.elapsed_time_ms or 0)
        else:
            if os.environ.get("KPOCKET_TTS_ERROR_WITHOUT_EOS", "0") == "1":
                raise RuntimeError("Generation reached maximum length without EOS!")
            logger.warning(
                "Maximum generation length reached without EOS, this very often indicates an error."
            )

        latents_queue.put(None)
        if steps_times:
            logger.info("Average generation step time: %d ms", int(statistics.mean(steps_times)))

    def _sample_noise(self, rng: np.random.Generator, std: float) -> np.ndarray:
        if self.noise_clamp is None:
            noise = rng.standard_normal((1, self.flow_lm.ldim)).astype(np.float32) * std
        else:
            # Truncated normal sampled by rejection.
            clamp = float(self.noise_clamp)
            noise = rng.standard_normal((1, self.flow_lm.ldim)).astype(np.float32) * std
            mask = np.abs(noise) > clamp * std
            while mask.any():
                fresh = rng.standard_normal(noise.shape).astype(np.float32) * std
                noise = np.where(mask, fresh, noise)
                mask = np.abs(noise) > clamp * std
        return noise


# ----------------------------------------------------------------------------
# Helpers needed at module load time.
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Text helpers (unchanged from the legacy implementation).
# ----------------------------------------------------------------------------


def prepare_text_prompt(
    text: str, pad_with_spaces_for_short_inputs: bool, remove_semicolons: bool
) -> tuple[str, int]:
    text = text.strip()
    if text == "":
        raise ValueError("Text prompt cannot be empty")
    text = text.replace("\n", " ").replace("\r", " ").replace("  ", " ")
    if remove_semicolons:
        text = text.replace(";", ",")
    number_of_words = len(text.split())
    if number_of_words <= 4:
        frames_after_eos_guess = 3
    else:
        frames_after_eos_guess = 1
    if not text[0].isupper():
        text = text[0].upper() + text[1:]
    if text[-1].isalnum():
        text = text + "."
    if pad_with_spaces_for_short_inputs and len(text.split()) < 5:
        text = " " * 8 + text
    return text, frames_after_eos_guess


def _find_boundary_indices(list_of_tokens, boundary_tokens):
    indices = [0]
    previous_was_boundary = False
    for idx, token in enumerate(list_of_tokens):
        if token in boundary_tokens:
            previous_was_boundary = True
        else:
            if previous_was_boundary:
                indices.append(idx)
            previous_was_boundary = False
    indices.append(len(list_of_tokens))
    return indices


def _segments_from_boundaries(list_of_tokens, boundary_indices, tokenizer):
    segments = []
    for i in range(len(boundary_indices) - 1):
        start = boundary_indices[i]
        end = boundary_indices[i + 1]
        text = tokenizer.sp.decode(list_of_tokens[start:end])
        segments.append((end - start, text))
    return segments


def split_into_best_sentences(
    tokenizer,
    text_to_generate: str,
    max_tokens: int,
    pad_with_spaces_for_short_inputs: bool,
    remove_semicolons: bool,
) -> list[str]:
    text_to_generate, _ = prepare_text_prompt(
        text_to_generate, pad_with_spaces_for_short_inputs, remove_semicolons
    )
    text_to_generate = text_to_generate.strip()
    tokens = tokenizer(text_to_generate)
    list_of_tokens = tokens.tokens[0].tolist()

    _, *end_of_sentence_tokens = tokenizer(".!...?").tokens[0].tolist()
    sentence_boundaries = _find_boundary_indices(list_of_tokens, end_of_sentence_tokens)
    nb_tokens_and_sentences = _segments_from_boundaries(
        list_of_tokens, sentence_boundaries, tokenizer
    )

    _, *fallback_tokens = tokenizer(",;:").tokens[0].tolist()
    refined_segments = []
    for nb_tokens, text in nb_tokens_and_sentences:
        if nb_tokens <= max_tokens:
            refined_segments.append((nb_tokens, text))
        else:
            sub_tokens = tokenizer(text.strip()).tokens[0].tolist()
            sub_boundaries = _find_boundary_indices(sub_tokens, fallback_tokens)
            sub_segments = _segments_from_boundaries(sub_tokens, sub_boundaries, tokenizer)
            if len(sub_segments) > 1:
                refined_segments.extend(sub_segments)
            else:
                refined_segments.append((nb_tokens, text))

    chunks: list[str] = []
    current_chunk = ""
    current_count = 0
    for nb_tokens, sentence in refined_segments:
        if current_chunk == "":
            current_chunk = sentence
            current_count = nb_tokens
            continue
        if current_count + nb_tokens > max_tokens:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_count = nb_tokens
        else:
            current_chunk += " " + sentence
            current_count += nb_tokens
    if current_chunk != "":
        chunks.append(current_chunk.strip())

    for chunk in chunks:
        chunk_tokens = tokenizer(chunk.strip()).tokens[0].tolist()
        if len(chunk_tokens) > max_tokens:
            logger.warning(
                "Chunk has %d tokens (max %d), generation may skip words: '%.50s...'",
                len(chunk_tokens),
                max_tokens,
                chunk,
            )

    return chunks


def export_model_state(model_state: dict[str, dict[str, np.ndarray]], dest: str | Path) -> None:
    flat: dict[str, np.ndarray] = {}
    for module_name, module_state in model_state.items():
        for key, tensor_value in module_state.items():
            flat[f"{module_name}/{key}"] = np.asarray(tensor_value)
    safetensors.numpy.save_file(flat, str(dest))


def _import_model_state(
    source: str | Path, schema: list[tuple[str, str]] | None = None
) -> dict[str, dict[str, np.ndarray]]:
    """Import a voice-prompt safetensors file.

    Supports both the legacy `{cache, offset}` layout and the new
    `{k, v, offset}` layout used by this MAX implementation. When the
    legacy layout is detected, it is converted on the fly using the offset
    field to truncate the NaN-filled trailing positions.
    """
    raw: dict[str, dict[str, np.ndarray]] = {}
    with safetensors.safe_open(str(source), framework="numpy") as f:
        for key in f.keys():
            module_name, tensor_key = key.split("/")
            raw.setdefault(module_name, {})
            raw[module_name][tensor_key] = f.get_tensor(key)

    converted: dict[str, dict[str, np.ndarray]] = {}
    for module_name, sub in raw.items():
        if "cache" in sub and "offset" in sub:
            cache = sub["cache"]
            offset = int(np.asarray(sub["offset"]).reshape(-1)[0])
            # Legacy shape: (2, B, T_max, H, D). Truncate to (1, offset, H, D).
            k = cache[0, :, :offset, :, :].astype(np.float32)
            v = cache[1, :, :offset, :, :].astype(np.float32)
            converted[module_name] = {
                "k": k,
                "v": v,
                "offset": np.asarray([offset], dtype=np.int64),
            }
        elif "current_end" in sub:
            tensor = sub["current_end"]
            converted[module_name] = {"offset": np.asarray([tensor.shape[0]], dtype=np.int64)}
        else:
            converted[module_name] = sub

    return converted
