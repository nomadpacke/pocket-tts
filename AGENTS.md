# AGENTS.md

This file provides guidance to AI agents when working with code in this repository.

## Project Overview

pocket-tts is a CPU-based text-to-speech (TTS) model. The project uses a flow-based language model architecture with a neural audio codec (Mimi) for efficient speech synthesis.

**Key Architecture Components:**
- **FlowLMModel**: Transformer-based flow language model that generates latent representations from text using Lagrangian Self Distillation (LSD)
- **MimiModel**: Neural audio codec (from the `moshi` package) that compresses/decompresses audio to/from latent representations
- **Conditioners**: Text processing via SentencePiece tokenizer and lookup table embeddings
- **Streaming Architecture**: The entire pipeline supports streaming generation via stateful modules
- **Web API**: FastAPI-based server for HTTP API access with web interface

## Common Commands

### Setup and Development
```bash
# Install pre-commit hooks
uvx pre-commit install

# Run tests (3 parallel workers)
uv run pytest -n 3 -v

# Run a single test
uv run pytest tests/test_python_api.py -v

# Run CLI locally (editable install)
uv run pocket-tts generate
uv run pocket-tts serve
```

### Linting and Formatting
Pre-commit handles this automatically, but you can run manually:
```bash
# Ruff will run automatically on commit via pre-commit
# Includes: ruff-check, ruff-format (with --fix), and import sorting
```

### Building (No Build Step)
This is a pure Python package with Rust extensions in `training/rust_exts/audio_ds/` for training-time audio processing. The main package does not require building.

## Code Structure

### Main Package (`pocket_tts/`)

**Entry Points:**
- `main.py`: CLI implementation with Typer (commands: `generate`, `serve`, and web interface)
- `__init__.py`: Public API exports only `TTSModel`
- `__main__.py`: Python module entry point
- `default_parameters.py`: Default configuration values for generation parameters
- `static/`: Web interface files (HTML for server UI)

**Core Models (`models/`):**
- `tts_model.py`: Main `TTSModel` class - orchestrates the entire TTS pipeline
  - `load_model()`: Downloads weights from HuggingFace and initializes models
  - `get_state_for_audio_prompt()`: Encodes audio prompt (voice) into model state
  - `generate_audio_stream()`: Streaming generation that yields audio chunks
  - Uses LRU cache for voice prompts to avoid reprocessing
- `flow_lm.py`: `FlowLMModel` - transformer that generates latent audio codes from text

**Modules (`modules/`):**
- `transformer.py`: `StreamingTransformer` and `StreamingMultiheadAttention` with RoPE embeddings
- `stateful_module.py`: Base class for streaming support (maintains KV cache and state)
- `rope.py`: Rotary Position Embeddings
- `mlp.py`: `SimpleMLPAdaLN` (AdaLN-conditioned MLP for flow prediction)
- `conv.py`: Convolution utilities
- `seanet.py`: SEANet encoder/decoder (copied from moshi)

**Conditioners (`conditioners/`):**
- `text.py`: `LUTConditioner` - Sentence

## Personal Notes

> **Fork purpose:** Using this for experimenting with local TTS on CPU-only machines.
> The LRU cache in `tts_model.py` for voice prompts is worth tuning if you're
> cycling through many different voice prompts — default cache size may be small.
