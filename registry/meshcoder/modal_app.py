import modal
from typing import Dict, Any
import os

# ============================================================================
# Configuration
# ============================================================================

APP_NAME = "omnilaunch-meshcoder"
VOLUME_NAME = "omnilaunch"

# Model paths
# Try Unsloth version first (ungated), fallback to Meta's official (gated)
HF_MODEL_REPO_BASE = "meta-llama/Llama-3.2-1B"  # Gated - requires approval
HF_MODEL_REPO_MESHCODER = "InternRobotics/MeshCoder"
BASE_MODEL_PATH = "/omnilaunch/models/meta-llama/Llama-3.2-1B"
MESHCODER_PATH = "/omnilaunch/models/InternRobotics/MeshCoder"

# HuggingFace cache
HF_CACHE_DIR = "/omnilaunch/hf_cache"

app = modal.App(APP_NAME)
omnilaunch_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# ============================================================================
# Image Definition
# ============================================================================

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-runtime-ubuntu22.04", add_python="3.10")
    .apt_install(
        "git", "libgl1", "libglib2.0-0", "libsm6", "libxext6", "libxrender1",
        "libgomp1", "ffmpeg", "wget", "ca-certificates", "build-essential"
    )
    # Install PyTorch first (PyTorch3D needs it during build)
    # Pinning exact versions from MeshCoder README (CUDA 11.8 → 12.1 for Modal)
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        "torchaudio==2.4.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    # Install build dependencies for PyTorch3D
    .pip_install(
        "wheel",
        "ninja",
        "iopath",
        "fvcore",
    )
    # Install PyTorch3D from source (not available on PyPI)
    .run_commands(
        "CC=gcc CXX=g++ pip install --no-build-isolation 'git+https://github.com/facebookresearch/pytorch3d.git@stable'"
    )
    # Install torch-scatter using prebuilt wheels from PyG
    .run_commands(
        "pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu121.html"
    )
    # Install xformers (pinned version from MeshCoder, adapted for cu121)
    .pip_install(
        "xformers==0.0.28",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    # Install bpy (Blender Python) from official wheel
    .run_commands(
        "wget https://download.blender.org/pypi/bpy/bpy-4.0.0-cp310-cp310-manylinux_2_28_x86_64.whl",
        "pip install bpy-4.0.0-cp310-cp310-manylinux_2_28_x86_64.whl",
        "rm bpy-4.0.0-cp310-cp310-manylinux_2_28_x86_64.whl"
    )
    # Install remaining MeshCoder dependencies
    .pip_install(
        # Core dependencies from MeshCoder
        "transformers==4.50.0",  # Need newer version for FlashAttentionKwargs
        "peft",  # MeshCoder doesn't pin, use latest compatible
        "accelerate",  # MeshCoder doesn't pin, use latest compatible
        "huggingface_hub",  # MeshCoder doesn't pin, use latest compatible
        "datasets",
        # 3D processing (from MeshCoder README)
        "opencv-python",  # cv2
        "trimesh",
        "scikit-image",
        "boto3",
        "wandb",
        # Note: bpy installed separately from official wheel (see above)
        # Other MeshCoder deps
        "einops",
        "omegaconf",
        "jaxtyping",
        "typeguard",
        "pyyaml",
    )
    .env({
        "HF_HOME": HF_CACHE_DIR,
        "HF_HUB_CACHE": HF_CACHE_DIR,
        "TRANSFORMERS_CACHE": HF_CACHE_DIR,
    })
)


# ============================================================================
# Helper Functions
# ============================================================================

def load_mesh_file(mesh_data: str):
    """Load mesh from base64 string or file path.

    Args:
        mesh_data: Either base64-encoded mesh file or path to mesh on volume

    Returns:
        Path to temporary mesh file
    """
    import tempfile
    import base64

    s = (mesh_data or "").strip()

    # Try as file path first
    if os.path.exists(mesh_data):
        print(f"Loading mesh from file: {mesh_data}")
        return mesh_data

    # Otherwise try to decode as base64
    try:
        # Fix missing padding if necessary
        missing = (-len(s)) % 4
        if missing:
            s = s + ("=" * missing)
        mesh_bytes = base64.b64decode(s, validate=False)

        # Write to temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".glb")
        temp_file.write(mesh_bytes)
        temp_file.close()

        print(f"Decoded base64 mesh to temporary file: {temp_file.name}")
        return temp_file.name
    except Exception as e:
        raise ValueError(
            f"Could not load mesh. Expected base64 string or accessible file path.\n"
            f"Error: {e}\n"
            f"Input (first 100 chars): {str(mesh_data)[:100]}..."
        )


# ============================================================================
# Modal Functions
# ============================================================================

@app.function(
    image=image,
    volumes={"/omnilaunch": omnilaunch_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=7200
)
def download_files() -> Dict[str, str]:
    """Download Llama-3.2-1B base model, MeshCoder weights, and clone repo."""
    from huggingface_hub import snapshot_download
    import subprocess

    # Debug: Check HF token
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print(f"✓ HF_TOKEN is set (last 4 chars: ...{hf_token[-4:]})")
    else:
        print("⚠ HF_TOKEN is not set!")

    # Download base Llama model
    if not os.path.exists(f"{BASE_MODEL_PATH}/config.json"):
        print(f"Downloading {HF_MODEL_REPO_BASE} to {BASE_MODEL_PATH}...")
        snapshot_download(
            HF_MODEL_REPO_BASE,
            local_dir=BASE_MODEL_PATH,
            local_dir_use_symlinks=False,
            token=hf_token,
        )
        print("✓ Base model downloaded")
    else:
        print(f"✓ Base model already cached at {BASE_MODEL_PATH}")

    # Download MeshCoder weights (shape tokenizer + LoRA adapters)
    if not os.path.exists(f"{MESHCODER_PATH}/config.yaml"):
        print(f"Downloading {HF_MODEL_REPO_MESHCODER} to {MESHCODER_PATH}...")
        snapshot_download(
            HF_MODEL_REPO_MESHCODER,
            local_dir=MESHCODER_PATH,
            local_dir_use_symlinks=False,
            token=hf_token,
        )
        print("✓ MeshCoder weights downloaded")
    else:
        print(f"✓ MeshCoder weights already cached at {MESHCODER_PATH}")

    # Clone MeshCoder repository for inference utilities
    repo_path = "/omnilaunch/code/meshcoder/MeshCoder"
    if not os.path.exists(repo_path):
        print("Cloning MeshCoder repository...")
        os.makedirs("/omnilaunch/code/meshcoder", exist_ok=True)
        subprocess.run([
            "git", "clone",
            "https://github.com/InternRobotics/MeshCoder.git",
            repo_path
        ], check=True)
        print("✓ Repository cloned")
    else:
        print(f"✓ Repository already exists at {repo_path}")

    omnilaunch_vol.commit()
    return {
        "ok": True,
        "base_model_path": BASE_MODEL_PATH,
        "meshcoder_path": MESHCODER_PATH,
        "repo_path": repo_path
    }


@app.function(
    image=image,
    volumes={"/omnilaunch": omnilaunch_vol},
    timeout=1800,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def setup(run_downloads: bool = True) -> Dict[str, Any]:
    """Prepare MeshCoder runner: verify env and download models."""
    import importlib

    torch_present = importlib.util.find_spec("torch") is not None
    torch_version = None
    cuda_available = False

    if torch_present:
        import torch as _torch
        torch_version = str(_torch.__version__)
        cuda_available = bool(_torch.cuda.is_available())
        del _torch

    transformers_present = importlib.util.find_spec("transformers") is not None
    transformers_version = None
    if transformers_present:
        import transformers
        transformers_version = str(transformers.__version__)

    peft_present = importlib.util.find_spec("peft") is not None
    pytorch3d_present = importlib.util.find_spec("pytorch3d") is not None

    checks = {
        "torch_present": bool(torch_present),
        "torch_version": torch_version,
        "cuda_available": bool(cuda_available),
        "transformers_present": bool(transformers_present),
        "transformers_version": transformers_version,
        "peft_present": bool(peft_present),
        "pytorch3d_present": bool(pytorch3d_present),
        "volume_mounted": os.path.exists("/omnilaunch"),
        "base_model_present_before": os.path.exists(f"{BASE_MODEL_PATH}/config.json"),
        "meshcoder_present_before": os.path.exists(f"{MESHCODER_PATH}/config.yaml"),
    }

    try:
        if run_downloads:
            dl_result = download_files.local()
            checks["download_result"] = dl_result
            checks["base_model_present_after"] = os.path.exists(f"{BASE_MODEL_PATH}/config.json")
            checks["meshcoder_present_after"] = os.path.exists(f"{MESHCODER_PATH}/config.yaml")
            checks["repo_present"] = os.path.exists("/omnilaunch/code/meshcoder/MeshCoder")
        else:
            checks["base_model_present_after"] = checks["base_model_present_before"]
            checks["meshcoder_present_after"] = checks["meshcoder_present_before"]
            checks["repo_present"] = os.path.exists("/omnilaunch/code/meshcoder/MeshCoder")

        ok = (checks["volume_mounted"] and
              checks["base_model_present_after"] and
              checks["meshcoder_present_after"] and
              checks["repo_present"] and
              checks["peft_present"] and
              checks["pytorch3d_present"])
        return {"ok": ok, "checks": checks}
    except Exception as e:
        return {"ok": False, "error": str(e), "checks": checks}


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/omnilaunch": omnilaunch_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=600,
    scaledown_window=2
)
def infer(params: Dict[str, Any]) -> Dict[str, Any]:
    """MeshCoder inference: mesh/point cloud → Blender Python code.

    params: {
      mesh_path: str - path to mesh file (GLB, OBJ, STL, etc.) - will auto-convert,
      points: list/array of shape (N, 3) - point cloud coordinates (alternative),
      normals: list/array of shape (N, 3) - point cloud normals (optional),
      npz_path: str - path to NPZ file with 'points' and 'normals' (alternative),
      num_points: int (default: 16384) - number of points to sample from mesh,
      max_new_tokens: int (default: 4096) - max tokens to generate (increase for complex objects),
      temperature: float (default: 1.0),
      do_sample: bool (default: False - use greedy decoding),
      zero_out_normals: bool (default: True - ignore normals, use zeros),
      seed: int (default: 42) - random seed for reproducible downsampling
    }

    Returns: {
      code: str - generated Blender Python code,
      success: bool - whether code generation succeeded
    }
    """
    import torch
    import yaml
    import trimesh
    from transformers import AutoTokenizer
    from peft import PeftModel
    import numpy as np
    import random

    # Import MeshCoder-specific modules (we'll need to add the repo to path)
    import sys
    sys.path.insert(0, "/omnilaunch/code/meshcoder/MeshCoder/src")

    from llama_recipes.custom_models.modules.shape_tokenizer import ShapeTokenizer
    from llama_recipes.custom_models.modeling_llama_3_2 import LlamaForCausalLM
    from llama_recipes.utils.train_utils_shape2code import shape_to_text

    # Parse parameters
    mesh_path = params.get("mesh_path")
    npz_path = params.get("npz_path")
    points_data = params.get("points")
    normals_data = params.get("normals")
    num_points = int(params.get("num_points", 16384))
    max_new_tokens = int(params.get("max_new_tokens", 4096))
    temperature = float(params.get("temperature", 1.0))
    do_sample = bool(params.get("do_sample", False))
    zero_out_normals = bool(params.get("zero_out_normals", True))
    seed = int(params.get("seed", 42))

    # Set random seeds for reproducibility (matching MeshCoder's approach)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Load or convert to point cloud
    if mesh_path:
        # Handle both base64-encoded files and paths on Modal volume
        actual_mesh_path = load_mesh_file(mesh_path)
        mesh = trimesh.load(actual_mesh_path, force='mesh')

        # Handle scenes (multiple meshes)
        if isinstance(mesh, trimesh.Scene):
            print("Mesh is a scene, combining geometries...")
            mesh = trimesh.util.concatenate([
                geom for geom in mesh.geometry.values()
                if isinstance(geom, trimesh.Trimesh)
            ])

        print(f"Mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Sample points from mesh surface
        print(f"Sampling {num_points} points from mesh surface...")
        points_np, face_indices = trimesh.sample.sample_surface(mesh, num_points)
        normals_np = mesh.face_normals[face_indices]

        # Normalize to -1 to 1 range (as expected by MeshCoder)
        center = points_np.mean(axis=0)
        points_centered = points_np - center
        max_extent = np.abs(points_centered).max()
        points_np = points_centered / max_extent

        print(f"Point cloud normalized to range [{points_np.min():.3f}, {points_np.max():.3f}]")

        points = torch.from_numpy(points_np).float()
        normals = torch.from_numpy(normals_np).float()

    elif npz_path:
        print(f"Loading point cloud from: {npz_path}")
        data = np.load(npz_path)
        points = torch.from_numpy(data['points']).float()
        normals = torch.from_numpy(data.get('normals', np.zeros_like(data['points']))).float()

    elif points_data is not None:
        points = torch.tensor(points_data, dtype=torch.float32)
        if normals_data is not None:
            normals = torch.tensor(normals_data, dtype=torch.float32)
        else:
            normals = torch.zeros_like(points)
    else:
        return {"error": "Either 'mesh_path', 'npz_path', or 'points' must be provided"}

    # Ensure batch dimension
    if points.dim() == 2:
        points = points.unsqueeze(0)  # (1, N, 3)
    if normals.dim() == 2:
        normals = normals.unsqueeze(0)  # (1, N, 3)

    # Downsample to 16384 points if needed (using seeded random sampling for reproducibility)
    num_points_actual = points.shape[1]
    expected_num_points = 16384
    if num_points_actual > expected_num_points:
        print(f"Downsampling from {num_points_actual} to {expected_num_points} points (seed={seed})")
        # Use seeded random sampling (matches MeshCoder's approach)
        idx = random.sample(range(num_points_actual), expected_num_points)
        idx = sorted(idx)  # Sort to maintain some spatial coherence
        points = points[:, idx, :]
        normals = normals[:, idx, :]
    elif num_points_actual < expected_num_points:
        print(f"Warning: Only {num_points_actual} points provided, expected {expected_num_points}")

    # Concatenate points and normals
    if zero_out_normals:
        normals = torch.zeros_like(normals)
    point_cloud = torch.cat([points, normals], dim=2)  # (1, N, 6)

    # Load models
    print(f"Loading base model from {BASE_MODEL_PATH}...")
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        return_dict=True,
        load_in_8bit=False,
        device_map='cuda:0',
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )

    print(f"Loading LoRA adapters from {MESHCODER_PATH}...")
    model = PeftModel.from_pretrained(model, MESHCODER_PATH)
    model.eval()

    # Load shape tokenizer
    print("Loading shape tokenizer...")
    with open(os.path.join(MESHCODER_PATH, 'config.yaml'), 'r') as yaml_file:
        shape_tokenizer_config = yaml.safe_load(yaml_file)

    shape_tokenizer_model = ShapeTokenizer(**shape_tokenizer_config)
    shape_tokenizer_ckpt = torch.load(
        os.path.join(MESHCODER_PATH, 'shape_tokenizer.pt'),
        map_location='cpu',
        weights_only=False
    )
    shape_tokenizer_model.load_state_dict(shape_tokenizer_ckpt['model_state_dict'])
    shape_tokenizer_model.to('cuda')
    shape_tokenizer_model.eval()

    # Load tokenizer
    print("Loading text tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Move to GPU
    point_cloud = point_cloud.to('cuda')

    # Generate code
    print(f"Generating Blender code (max_tokens={max_new_tokens}, temp={temperature})...")
    with torch.no_grad():
        generation_result = shape_to_text(
            model,
            shape_tokenizer_model,
            tokenizer,
            point_cloud,
            points_mask=None,
            gt_points=None,
            eval_reconstruction_performance=False,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            min_length=0,
            use_cache=True,
            top_k=50,
            repetition_penalty=1.0,
            length_penalty=1,
            other_kwargs={}
        )

    generated_code = generation_result['output_text_list'][0]
    success = bool(generation_result.get('success', [True])[0])

    print(f"Code generation {'succeeded' if success else 'failed'}")
    print(f"Generated {len(generated_code)} characters of code")

    return {
        "content_type": "application/json",
        "data": {
            "code": generated_code,
            "success": success,
            "num_points_used": point_cloud.shape[1],
        }
    }


if __name__ == "__main__":
    app.deploy()
