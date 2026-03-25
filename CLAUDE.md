# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FunASR is a fundamental end-to-end speech recognition toolkit by Alibaba DAMO Academy. It supports ASR, VAD, punctuation restoration, speaker verification/diarization, emotion recognition, keyword spotting, and multi-talker ASR. Models are hosted on ModelScope and HuggingFace.

Current version: 1.3.1 (in `funasr/version.txt`).

## Build & Install

```bash
# Install in editable mode (preferred for development)
pip3 install -e ./

# Install with extras
pip3 install -e ".[train]"    # training dependencies
pip3 install -e ".[test]"     # test dependencies
pip3 install -e ".[llm]"      # LLM/multimodal dependencies
pip3 install -e ".[all]"      # everything
```

Requirements: Python >= 3.8, PyTorch >= 1.13, torchaudio.

## CLI Entry Points

After installation, these commands are available:

- `funasr` — inference (`funasr.bin.inference:main_hydra`)
- `funasr-train` — training with DDP/FSDP (`funasr.bin.train:main_hydra`)
- `funasr-train-ds` — training with DeepSpeed (`funasr.bin.train_ds:main_hydra`)
- `funasr-export` — model export to ONNX (`funasr.bin.export:main_hydra`)
- `scp2jsonl` / `jsonl2scp` / `sensevoice2jsonl` — data format converters

All CLI commands use Hydra for configuration (pass args as `key=value`).

## Running Tests

```bash
# Run all tests
python tests/run_test.py

# Run a specific test file
python -m pytest tests/test_auto_model.py

# Run with a filename pattern
python tests/run_test.py --pattern "test_vad_*.py"
```

Tests use unittest. The runner is in `tests/run_test.py`. Most tests require model downloads and GPU.

## Architecture

### Registry System (`funasr/register.py`)

All models, encoders, decoders, frontends, tokenizers, datasets, etc. are registered via `tables.register()` decorator. The `RegisterTables` dataclass holds dictionaries keyed by class name. Each model file registers itself on import.

### Auto-import (`funasr/__init__.py`)

On package import, `import_submodules` recursively walks all subpackages, triggering all `@tables.register()` decorators. This makes every registered component discoverable by name through `AutoModel`.

### AutoModel (`funasr/auto/auto_model.py`)

The primary user-facing API. Handles:
- Model download from ModelScope/HuggingFace hubs
- Building model + optional VAD, punctuation, and speaker models as a pipeline
- `generate()` method for inference
- `export()` method for ONNX export

Typical usage:
```python
from funasr import AutoModel
model = AutoModel(model="paraformer-zh")
res = model.generate(input="audio.wav")
```

### Key Package Structure

- **`funasr/models/`** — ~50 model implementations (paraformer, conformer, whisper, sense_voice, emotion2vec, campplus, etc.). Each model directory typically contains `model.py` with the registered model class.
- **`funasr/frontends/`** — Audio feature extraction (fbank, wav2vec, etc.)
- **`funasr/tokenizer/`** — Tokenizer implementations (sentencepiece, BPE, character-based)
- **`funasr/datasets/`** — Dataset classes for different tasks (audio, KWS, LLM, SenseVoice)
- **`funasr/train_utils/`** — Training loop (`trainer.py`, `trainer_ds.py`), model averaging, initialization
- **`funasr/losses/`** — Loss functions
- **`funasr/metrics/`** — Evaluation metrics
- **`funasr/bin/`** — CLI entry points (inference, train, train_ds, export)
- **`fun_text_processing/`** — Text normalization and inverse text normalization (separate package)

### Runtime (`runtime/`)

C++ and multi-language SDK implementations for production deployment:
- **`runtime/onnxruntime/`** — C++ server with ONNX Runtime (main production runtime)
- **`runtime/python/`** — Python bindings for libtorch and ONNX runtimes
- **`runtime/websocket/`** / **`runtime/grpc/`** / **`runtime/http/`** — Server protocols
- Client SDKs: `runtime/java/`, `runtime/csharp/`, `runtime/golang/`, `runtime/ios/`, `runtime/android/`
- **`runtime/triton_gpu/`** — Triton Inference Server integration

### Model Pipeline Flow

1. `AutoModel.__init__` downloads model config from hub, looks up model class via `tables.model_classes`
2. Optionally builds VAD → ASR → Punctuation → Speaker pipelines
3. `generate()` runs the pipeline: VAD segments audio → ASR transcribes segments → punctuation/speaker post-processing
4. Results include text, timestamps, and optionally speaker labels
