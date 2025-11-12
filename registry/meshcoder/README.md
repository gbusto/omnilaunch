# MeshCoder Runner

Convert 3D meshes/point clouds into editable Blender Python scripts using MeshCoder.

## Quick Start

```bash
# 1. Request HuggingFace access (one-time)
# - https://huggingface.co/meta-llama/Llama-3.2-1B (click "Request access")
# - https://huggingface.co/InternRobotics/MeshCoder (click "Request access")

# 2. Configure HuggingFace token
modal secret create huggingface-secret HF_TOKEN=your_token_here

# 3. Setup runner (one-time, ~5-10 min)
omni build omnilaunch/meshcoder
omni setup omnilaunch/meshcoder

# 4. Generate Blender code from mesh
omni run omnilaunch/meshcoder infer -p mesh_path=chair.glb --save

# 5. Visualize (requires Blender installed)
blender --background --python omnilaunch/registry/meshcoder/scripts/visualize_meshcoder_local.py -- omni_out/blender_code.py output.glb
```

## Overview

MeshCoder is a framework that converts 3D point clouds into procedural Blender Python code, enabling programmatic reconstruction and editing of complex human-made objects (furniture, appliances, etc.).

**Model**: Llama-3.2-1B + LoRA fine-tuned on 100K+ object-code pairs
**Input**: Point clouds (16,384 points with normals)
**Output**: Blender Python scripts with part-segmented mesh generation code

## Prerequisites

### 1. HuggingFace Access Token

You need a HuggingFace account with access to gated models:

1. **Create HuggingFace account**: https://huggingface.co/join
2. **Generate access token**: https://huggingface.co/settings/tokens (read access is sufficient)
3. **Request access to required models**:
   - **Llama-3.2-1B**: https://huggingface.co/meta-llama/Llama-3.2-1B
     - Click "Request access" and accept Meta's license
     - Usually approved within a few minutes
   - **MeshCoder**: https://huggingface.co/InternRobotics/MeshCoder
     - Click "Request access"
     - Auto-approved instantly
4. Update your access token in HuggingFace. You need to make sure under `Repositories permissions` that you add `meta-llama/Llama-3.2-1B` (it won't be possible until Meta approves your access to the Llama model) and `InternRobotics/MeshCoder`.
5. **Set your token as a Modal secret**:
```bash
modal secret create huggingface-secret HF_TOKEN=your_token_here
```

### 2. Blender (for visualization)

To visualize generated code, you need Blender installed. Install it, then make sure `blender` is accessible from your command line.

**Install Blender**:
- macOS: `brew install --cask blender`
- Linux: `sudo apt-get install blender`
- Windows: https://www.blender.org/download/

**Make Blender accessible** - Choose one:

```bash
# Option 1: Add alias to your shell profile (~/.zshrc, ~/.bashrc, etc.)
alias blender="/Applications/Blender.app/Contents/MacOS/Blender"  # macOS
alias blender="/usr/bin/blender"                                  # Linux

# Option 2: Set environment variable
export BLENDER="/Applications/Blender.app/Contents/MacOS/Blender"
# Then use: $BLENDER --background --python script.py

# Option 3: Just use the full path (no setup needed)
/Applications/Blender.app/Contents/MacOS/Blender --background --python script.py
```

Test it: `blender --version` (should show Blender version)

## Setup

1. **Build and deploy the runner:**
```bash
# Build the runner (packages it as versioned bundle)
omni build omnilaunch/meshcoder

# Deploy to Modal and download models (one-time setup)
omni setup omnilaunch/meshcoder
```

This will:
- Deploy the Modal app
- Download Llama-3.2-1B base model (~2.5GB)
- Download MeshCoder weights (LoRA + shape tokenizer)
- Clone MeshCoder repository for inference utilities
- Verify the environment

**Note**: Setup requires your HuggingFace token to be configured as `huggingface-secret` in Modal.

## Usage

### Inference: Mesh → Blender Code

Simply pass a mesh file (GLB, OBJ, STL, etc.) - the runner handles point cloud conversion automatically:

```bash
omni run omnilaunch/meshcoder infer \
  -p mesh_path=chair.glb \
  -p max_new_tokens=4096 \
  --save --outfile chair_code.py
```

The mesh will be automatically converted to a point cloud (16,384 points), and MeshCoder will generate Blender Python code.

### Parameters

- `mesh_path` (str): Path to mesh file (GLB, OBJ, STL, etc.) - **required**
- `max_new_tokens` (int): Max tokens to generate (default: 4096)
- `temperature` (float): Sampling temperature (default: 1.0)
- `do_sample` (bool): Use sampling vs greedy decoding (default: false)
- `seed` (int): Random seed for reproducible point sampling (default: 42)

### Output

The result is saved to `omni_out/chair_code.py` (or your specified filename):

```json
{
  "code": "import bpy\nfrom math import radians, pi\nfrom bpy_lib import *\n\ndelete_all()\n\n# object name: chair\n...",
  "success": true,
  "num_points_used": 16384
}
```

The `code` field contains executable Blender Python that reconstructs the mesh.

## Important Notes

1. **Training Data**: Model trained on procedural CAD-style objects (furniture, appliances). Works best on:
   - Furniture (chairs, tables, sofas)
   - Appliances (lamps, microwaves)
   - Architectural elements (doors, windows)

   May not work well on organic/sculpted shapes, characters, or highly artistic models.

2. **Approximation**: MeshCoder generates *parametric* code that creates a similar object, not an exact reconstruction. It learns semantic structure (e.g., "chair with 4 legs, seat, back") rather than memorizing exact geometry.

3. **GPU Requirements**: Inference requires ~12GB VRAM (A10G works fine).

## Visualizing Generated Code

After running inference, you'll have a JSON file with Blender Python code. To visualize it:

### Option 1: Export to GLB/OBJ (Headless)

```bash
# Convert Blender code to GLB mesh
/Applications/Blender.app/Contents/MacOS/Blender --background \
  --python omnilaunch/registry/meshcoder/scripts/visualize_meshcoder_local.py -- \
  omni_out/blender_code.py output.glb

# On Linux:
blender --background \
  --python omnilaunch/registry/meshcoder/scripts/visualize_meshcoder_local.py -- \
  omni_out/blender_code.py output.glb
```

This creates:
- `output.glb` - 3D model (view in Babylon.js, Three.js, etc.)
- `output.obj` - Also exports OBJ format

**Note**: The script automatically corrects orientation (converts to Y-up) since MeshCoder's output is oriented incorrectly by default. If you want the original orientation for some reason, add `--no-yup`.

### Option 2: Interactive Blender GUI

```bash
# Open in Blender to inspect/edit the generated mesh
/Applications/Blender.app/Contents/MacOS/Blender \
  --python omnilaunch/registry/meshcoder/scripts/visualize_meshcoder_local.py -- \
  omni_out/blender_code.py
```

The scene will load with all generated objects. You can inspect, modify, or re-export from Blender's GUI.

### Visualization Script Details

The visualization script (`scripts/visualize_meshcoder_local.py`) uses a minimal implementation of MeshCoder's Blender API that doesn't require torch or heavy dependencies - just Blender's built-in Python.

## Complete Example Workflow

```bash
# 1. Setup (one-time)
# First, request HuggingFace access and create Modal secret:
modal secret create huggingface-secret HF_TOKEN=your_hf_token_here

# Build and deploy
omni build omnilaunch/meshcoder
omni setup omnilaunch/meshcoder

# 2. Run inference on a mesh file
omni run omnilaunch/meshcoder infer \
  -p mesh_path=chair.glb \
  -p max_new_tokens=4096 \
  --save --outfile chair_code.py

# 3. Visualize the generated code
blender --background \
  --python omnilaunch/registry/meshcoder/scripts/visualize_meshcoder_local.py -- \
  omni_out/chair_code.py chair_output.glb

# 4. View the result
# Upload chair_output.glb to https://gltf-viewer.donmccurdy.com/
# Or open in Blender, Babylon.js, Three.js, etc.
```

## References

- Paper: [MeshCoder: LLM-Powered Structured Mesh Code Generation](https://huggingface.co/papers/2508.14879)
- Model: [InternRobotics/MeshCoder](https://huggingface.co/InternRobotics/MeshCoder)
- Dataset: [MeshCoderDataset](https://huggingface.co/datasets/InternRobotics/MeshCoderDataset)
