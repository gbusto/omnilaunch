# Omnilaunch Runner Authoring Guide (for LLM Coding Agents)

Purpose: Create new Omnilaunch runners that are reproducible, minimal, and correct on first try. Support three scenarios:
- Cold-start inference
- LoRA fine-tuning
- Full training

Always follow: build → setup → run. Keep GPU work on GPU entrypoints only; everything else on CPU.

## Core Principles
- Deterministic and reproducible: pin critical deps; keep model/data URIs explicit.
- Minimal surface: only the entrypoints required; no extra features in v1.
- Clear parameters: every entrypoint has a JSON schema and helpful descriptions.
- Fast cold-starts: cache models/datasets on the Modal volume when possible.
- Transparent costs: pick the smallest viable GPU; set timeouts realistically.

## When to Use CPU vs GPU
- CPU (no GPU):
  - `download_files` (snapshot model weights, tokenizer, support files)
  - `setup` (environment checks, light verifications)
  - `download_dataset` (optional; pre-cache HF datasets to volume)
- GPU:
  - `infer` (cold-start inference; simple generation)
  - `train_lora` (parameter-efficient fine-tuning)
  - `train_full` (full fine-tuning or pretraining)

Guidance:
- Prefer A10G for most 7B–20B inference or 8B LoRA training; use H100 for very large models or full training with bigger batches.
- For quantized or memory-efficient models, set dtype and quant configs explicitly; ensure required kernels (e.g., Triton) are installed.

## Secrets and Authentication

If your model requires authentication (gated models, private repos), use Modal secrets:

```python
@app.function(
    image=image,
    gpu="A10G",
    volumes={"/omnilaunch": omnilaunch_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=600
)
def infer(params: dict) -> dict:
    # HF_TOKEN is now available as environment variable
    pass
```

**Document required secrets in README** with setup instructions. For HuggingFace gated models, users need to:
1. Request access to the model (e.g., meta-llama models require accepting Meta's license)
2. Generate HF token with read permissions at https://huggingface.co/settings/tokens
3. Update token permissions to include the gated model repositories
4. Create Modal secret: `modal secret create huggingface-secret HF_TOKEN=your_token_here`

**Note approval times**: Meta models typically approve within minutes; document this in your README.

## Multi-Component Models

For models with multiple components (base model + adapters + custom modules like tokenizers):

```python
# Document all components clearly in constants
HF_MODEL_REPO_BASE = "meta-llama/Llama-3.2-1B"
HF_MODEL_REPO_ADAPTERS = "InternRobotics/MeshCoder"
BASE_MODEL_PATH = "/omnilaunch/models/meta-llama/Llama-3.2-1B"
ADAPTERS_PATH = "/omnilaunch/models/InternRobotics/MeshCoder"
```

**In download_files**:
- Download all components (base model, adapters, tokenizers, config files)
- Use separate `snapshot_download()` calls for each component
- Commit volume after all downloads complete

**In infer**:
- Load components in correct order
- Apply adapters/merges as required by the model architecture
- Document loading sequence in comments

## Non-Text Inputs/Outputs

For models that accept or return non-standard formats (images, 3D meshes, audio, etc.):

**File Inputs**:
- Use base64 encoding for file parameters
- The CLI **auto-encodes** parameters with these names: `image`, `file`, `mesh_path`, `model_path`, `attachment`, `audio_path`, `video_path`, `image_path`
- In your function, decode from base64:
```python
import base64
import tempfile
import os

# CLI sends base64-encoded file content
mesh_b64 = params.get("mesh_path")
with tempfile.NamedTemporaryFile(delete=False, suffix=".glb") as f:
    f.write(base64.b64decode(mesh_b64))
    mesh_path = f.name
```

**Binary/Complex Outputs**:
- For binary data (images, meshes, audio): return appropriate content_type
  - `image/png`, `image/jpeg`
  - `model/gltf-binary`, `application/octet-stream`
  - `audio/mpeg`, `video/mp4`
- For structured data: use `application/json` with nested payload
- Example returning generated code:
```python
return {
    "content_type": "application/json",
    "data": {
        "code": generated_blender_code,
        "success": True,
        "num_points_used": 16384
    }
}
```

## External Code Dependencies

If your model requires external code (not pip-installable on PyPI):

**Clone repos in download_files** (CPU entrypoint to save costs):
```python
@app.function(image=image, volumes={"/omnilaunch": omnilaunch_vol}, timeout=3600)
def download_files() -> dict:
    import subprocess
    import os

    # Clone external repo if not already present
    external_repo_path = "/omnilaunch/code/MeshCoder"
    if not os.path.exists(external_repo_path):
        subprocess.run([
            "git", "clone",
            "https://github.com/InternRobotics/MeshCoder.git",
            external_repo_path
        ], check=True)
        omnilaunch_vol.commit()

    # ... rest of download_files logic
    return {"ok": True, "external_code": external_repo_path}
```

**Add to Python path in inference**:
```python
import sys
sys.path.insert(0, "/omnilaunch/code/MeshCoder")
from meshcoder.utils import some_function  # Now available
```

**Install additional dependencies** from external repo if needed:
```python
.run_commands(
    "pip install -r /omnilaunch/code/MeshCoder/requirements.txt"
)
```

## Runner Structure (required files)
```
registry/<slug>/
├── runner.yaml          # name, version, app_name, entrypoints (with GPU), schemas
├── modal_app.py         # Modal app and functions (download_files, setup, infer, train_*)
├── schema/
│   ├── infer.json       # JSON Schema for inference params
│   ├── train_lora.json  # JSON Schema for LoRA training params
│   ├── train_full.json  # JSON Schema for full training params (if provided)
│   └── dataset.json     # JSON Schema for dataset URI (if applicable)
└── tests/smoke.py       # Minimal assertion (e.g., volume exists)
```

## runner.yaml Template
```yaml
name: omnilaunch/<slug>
version: "0.1.0"
app_name: "omnilaunch-<slug>"
volume: "omnilaunch"

entrypoints:
  download_files:
    function: modal_app.py::download_files
    gpu: null
  setup:
    function: modal_app.py::setup
    gpu: null
  infer:
    function: modal_app.py::infer
    gpu: "A10G"
    schema: schema/infer.json
  train_lora:
    function: modal_app.py::train_lora
    gpu: "A10G"
    schema: schema/train_lora.json
  train_full:
    function: modal_app.py::train_full
    gpu: "H100"
    schema: schema/train_full.json
```

## modal_app.py Requirements
1) Constants section at top for clarity:
```python
APP_NAME = "omnilaunch-<slug>"
VOLUME_NAME = "omnilaunch"
HF_MODEL_REPO = "<org>/<model>"
MODEL_PATH = "/omnilaunch/models/<org>/<model>"
HF_CACHE_DIR = "/omnilaunch/hf_cache"
```

2) Image definition with pinned or compatible versions:
```python
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Torch + CUDA runtime versions must match Modal base; prefer cu121 wheels
        "torch==2.5.0",
        "transformers==4.57.0",
        # add: datasets, trl, peft, accelerate, bitsandbytes (if needed), wandb (optional)
    )
    .env({
        "HF_HOME": HF_CACHE_DIR,
        "HF_HUB_CACHE": HF_CACHE_DIR,
        "TRANSFORMERS_CACHE": HF_CACHE_DIR,
        # For large models prone to VRAM fragmentation
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
)
```

3) Volume binding and app:
```python
omnilaunch_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)
```

4) EntryPoints (minimum semantics and return format)

- download_files (CPU): snapshot model to `MODEL_PATH` if missing.
```python
@app.function(image=image, volumes={"/omnilaunch": omnilaunch_vol}, timeout=3600)
def download_files() -> dict:
    from huggingface_hub import snapshot_download
    import os
    if not os.path.exists(f"{MODEL_PATH}/config.json"):
        snapshot_download(repo_id=HF_MODEL_REPO, local_dir=MODEL_PATH, local_dir_use_symlinks=False)
        omnilaunch_vol.commit()
    return {"ok": True, "model_path": MODEL_PATH}
```

- setup (CPU): verify key libs import and optionally call `download_files.local()`.
```python
@app.function(image=image, volumes={"/omnilaunch": omnilaunch_vol}, timeout=1800)
def setup(run_downloads: bool = True) -> dict:
    checks = {}
    try:
        import torch, transformers  # noqa
        checks["libs"] = "ok"
    except Exception as e:
        return {"ok": False, "error": f"missing libs: {e}"}
    if run_downloads:
        _ = download_files.local()
    return {"ok": True, "checks": checks}
```

- infer (GPU): load model/tokenizer; accept params per schema; return content_type + data.
```python
@app.function(image=image, gpu="A10G", volumes={"/omnilaunch": omnilaunch_vol}, timeout=600, scaledown_window=2)
def infer(params: dict) -> dict:
    # Required params depend on model type; typical: messages or prompt (+ optional image)
    max_tokens = int(params.get("max_tokens", 256))
    temperature = float(params.get("temperature", 0.0))
    # Load model/tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    mdl = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype="auto", device_map="auto")
    # Format input (chat or plain text) and generate
    # Return JSON with content_type and data
    return {"content_type": "application/json", "data": {"ok": True, "response": "..."}}
```

- train_lora (GPU): dataset → conversation/task format → SFT with LoRA; save adapters under `/omnilaunch/runs/<slug>/<run_name>`.
  - Accept `dataset_uri`, `epochs` or `steps`, `learning_rate`, `lora_r`, `lora_alpha`, `batch_size`, `gradient_accumulation_steps`, `run_name`.
  - Generate a human-readable run name if not provided (e.g., `coolname`).
  - Return `{ ok, run_name, output_path, train_runtime, train_loss }`.

- train_full (GPU): like LoRA but full finetune; keep batch sizes modest; use bf16/amp if appropriate.
  - Expose key knobs only (epochs/steps, lr, batch sizes). Avoid exotic flags.

5) Output Contract
- Single artifact: return a dict `{ "content_type": <MIME>, "data": <payload> }`.
- Multi-artifact: return `{ "parts": [ {"content_type": ..., "data": ...}, ... ] }`.
- For JSON, return `content_type = application/json` and an object.

## JSON Schemas (examples)
- `infer.json` (LLM):
```json
{
  "type": "object",
  "required": ["messages"],
  "properties": {
    "messages": {"type": "array", "description": "OpenAI-style chat messages"},
    "max_tokens": {"type": "integer", "default": 256},
    "temperature": {"type": "number", "default": 0.0}
  }
}
```

- `train_lora.json`:
```json
{
  "type": "object",
  "required": ["dataset_uri"],
  "properties": {
    "dataset_uri": {"type": "string"},
    "epochs": {"type": "integer", "default": 1},
    "steps": {"type": "integer", "description": "Optional override for steps"},
    "learning_rate": {"type": "number", "default": 2e-4},
    "lora_r": {"type": "integer", "default": 16},
    "lora_alpha": {"type": "integer", "default": 16},
    "batch_size": {"type": "integer", "default": 2},
    "gradient_accumulation_steps": {"type": "integer", "default": 4},
    "run_name": {"type": "string", "description": "Optional human-readable name"}
  }
}
```

## README Requirements

Every runner must include a comprehensive `README.md` in the same directory as `runner.yaml`. Include:

**Prerequisites Section**:
- HuggingFace account requirements (if gated models)
- Token generation and setup instructions
- Model access approval process (with typical approval times)
- External dependencies (Blender, FFmpeg, etc.)
- System requirements

**Setup Section**:
- Step-by-step build and deploy instructions
- Expected download sizes and times
- Secret configuration commands
- Verification steps

**Usage Section**:
- Simple quick-start example
- Parameter documentation with examples
- Common use cases
- Command-line examples with actual file paths

**Output Section**:
- What the model returns (format, structure)
- How to visualize or use the output
- Post-processing steps if needed

**Important Notes Section**:
- Training data characteristics
- Model limitations (what works well, what doesn't)
- GPU requirements and costs
- Best practices for inputs

**References Section**:
- Link to paper
- Link to model on HuggingFace
- Link to dataset
- Link to original repository

**Example structure**:
```markdown
# Model Name Runner

Brief description of what this model does.

## Quick Start
```bash
# 5 commands to go from zero to working
```

## Prerequisites
### 1. HuggingFace Access Token
[Detailed instructions]

### 2. External Dependencies (if any)
[Installation instructions]

## Setup
[Build and deploy steps]

## Usage
[Parameter examples]

## Visualizing Output (if applicable)
[How to view/use generated files]

## Important Notes
[Limitations, best practices, training data info]

## References
- Paper: [link]
- Model: [link]
- Dataset: [link]
```

## Best Practices & Gotchas
- Always set per-entrypoint `gpu` in `runner.yaml`.
- Leave `download_files` and `setup` on CPU to save cost.
- For decoder-only LLMs, set left padding and pass `attention_mask` during generation if needed.
- For quantized models (e.g., MXFP4), ensure correct kernels are installed and pass proper `quantization_config`.
- For large models, set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Use `--help` support: ensure each entrypoint has a schema so the CLI can show parameters.
- Save LoRA outputs under `/omnilaunch/runs/<slug>/<run_name>` and return `run_name`.
- Keep timeouts realistic: infer(600s), train_lora(7200s+), setup(1800s).

## Example Authoring Flow (Agent)
1) Create folder `registry/<slug>/` with required files.
2) Fill `runner.yaml` with name/version/app_name/entrypoints and per-entrypoint GPUs.
3) Implement `modal_app.py` with constants, image, volume, and functions.
4) Add JSON schemas with clear descriptions and defaults.
5) Add a minimal `tests/smoke.py` (e.g., verify volume exists).
6) Build locally: `omni build omnilaunch/<slug>`
7) Setup: `omni setup omnilaunch/<slug>:0.1.0`
8) Help-test: `omni run omnilaunch/<slug>:0.1.0 --help` and `... <entrypoint> --help`
9) Run infer with a tiny prompt; then (optionally) train_lora with a tiny dataset sample.

## Validation Checklist
- Entrypoints match function names and have correct `gpu` specification.
- All entrypoints with params have a JSON schema present.
- Inference returns proper `{content_type, data}`.
- `download_files` caches to `MODEL_PATH`; volume is committed.
- `setup` imports libs and optionally triggers downloads.
- Docstring and comments are minimal and helpful.

That’s it. Keep it simple, reproducible, and GPU-conscious.


