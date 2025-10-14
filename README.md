# Omnilaunch

Reproducible model training and inference runners for Modal. Package, deploy, and run model pipelines with a simple CLI.

## Installation

From the repo root:
```bash
pip install -e omnilaunch/
```

Or from inside `omnilaunch/`:
```bash
pip install -e .
```

## Quick Start

### 1. Check your environment
```bash
omni doctor
```
Validates Modal SDK, HuggingFace Hub, and authentication.

### 2. Build a runner
```bash
omni build omnilaunch/registry/sdxl/
```
Packages the SDXL runner into a versioned bundle with manifest and schemas.

### 3. Setup on Modal
```bash
omni setup omnilaunch/sdxl:0.1.0
```
Deploys the Modal app, builds the container image, and downloads model weights (~7GB SDXL base).

### 4. Run inference
```bash
# Using inline JSON params
omni run omnilaunch/sdxl:0.1.0 infer \
  --params '{"prompt":"astronaut on mars","steps":20}' \
  --save --outfile result.png

# Or using CLI flags
omni run omnilaunch/sdxl:0.1.0 infer \
  -p prompt="a serene mountain landscape at sunset" \
  -p steps=25 \
  -p width=1024 \
  -p height=768 \
  --save --outfile mountains.png
```

The generated image is saved to `./omni_out/result.png` (or your specified path).

## CLI Commands

| Command | Description |
|---------|-------------|
| `omni doctor` | Check local environment and credentials |
| `omni build <path>` | Package a runner into a versioned bundle |
| `omni setup <runner:version>` | Deploy app, build image, download models |
| `omni run <runner:version> <entrypoint>` | Execute training or inference |

## CLI Flags for `omni run`

- `--params <file or JSON>` - Parameters as YAML/JSON file or inline string
- `-p, --param key=value` - Individual param overrides (repeatable)
- `--save` - Auto-save outputs to disk
- `--outdir <path>` - Directory for saved outputs (default: `./omni_out`)
- `--outfile <name>` - Explicit output filename
- `--dataset <uri>` - Dataset URI for training (e.g., `hf:org/dataset` or `vol:/path`)

## Example Workflow

```bash
# 1. Validate environment
omni doctor

# 2. Build the SDXL runner
omni build omnilaunch/registry/sdxl/

# 3. Setup on Modal (deploys app + downloads models)
omni setup omnilaunch/sdxl:0.1.0

# 4. Run inference and save output
omni run omnilaunch/sdxl:0.1.0 infer \
  -p prompt="a magical forest with glowing mushrooms" \
  -p steps=30 \
  --save --outfile forest.png

# Output saved to: ./omni_out/forest.png
```

## What's a Runner?

A **runner** is a self-contained package for model training or inference:
- `runner.yaml` - Metadata (name, version, GPU, entrypoints)
- `modal_app.py` - Modal app with functions (setup, infer, train_*)
- `schema/*.json` - JSON schemas for input validation
- `tests/smoke.py` - Basic smoke tests

Runners are versioned, bundled as tarballs, and stored in the local registry (`omnilaunch/registry/`).


