import modal
from typing import Dict, Any
import os

# ============================================================================
# Configuration
# ============================================================================

# App and volume
APP_NAME = "omnilaunch-sdxl-spo"
VOLUME_NAME = "omnilaunch"

# Base model
HF_BASE_MODEL_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
BASE_MODEL_PATH = "/omnilaunch/models/stabilityai/stable-diffusion-xl-base-1.0"
BASE_MODEL_INDEX_FILE = f"{BASE_MODEL_PATH}/model_index.json"

# Custom VAE (fp16 fix)
HF_VAE_REPO = "madebyollin/sdxl-vae-fp16-fix"
VAE_PATH = "/omnilaunch/models/madebyollin/sdxl-vae-fp16-fix"
VAE_CONFIG_FILE = f"{VAE_PATH}/config.json"

# SPO LoRA
HF_SPO_LORA_REPO = "SPO-Diffusion-Models/SPO-SDXL_4k-p_10ep_LoRA"
SPO_LORA_FILENAME = "spo_sdxl_10ep_4k-data_lora_diffusers.safetensors"
SPO_LORA_DIR = "/omnilaunch/loras/SPO-SDXL_4k-p_10ep_LoRA"
SPO_LORA_PATH = f"{SPO_LORA_DIR}/{SPO_LORA_FILENAME}"

# HuggingFace cache
HF_CACHE_DIR = "/omnilaunch/hf_cache"

# ============================================================================
# App and Volume Setup
# ============================================================================

app = modal.App(APP_NAME)
omnilaunch_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# ============================================================================
# Image Definition
# ============================================================================

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-runtime-ubuntu22.04", add_python="3.10")
    .apt_install(
        "git", "libgl1", "libglib2.0-0", "libsm6", "libxext6", "libxrender1",
        "libgomp1", "ffmpeg", "wget", "ca-certificates"
    )
    .pip_install(
        [
            "torch==2.4.0",
            "torchvision==0.19.0",
            "xformers==0.0.27.post2",
            "diffusers==0.31.0",
            "transformers==4.46.3",
            "accelerate==1.2.1",
            "peft==0.13.2",
            "safetensors",
            "huggingface_hub",
            "pillow",
        ],
        extra_index_url="https://download.pytorch.org/whl/cu121",
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

def pil_image_to_base64(img) -> str:
    """Convert PIL Image to base64 string."""
    import io
    import base64
    
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ============================================================================
# Modal Functions
# ============================================================================


@app.function(
    image=image,
    volumes={"/omnilaunch": omnilaunch_vol},
    timeout=3600
)
def download_files() -> Dict[str, str]:
    """Download SDXL base model + custom VAE + SPO LoRA."""
    from huggingface_hub import snapshot_download, hf_hub_download
    
    results = {}
    
    # 1. SDXL base model
    if not os.path.exists(BASE_MODEL_INDEX_FILE):
        print(f"Downloading {HF_BASE_MODEL_REPO}...")
        snapshot_download(
            HF_BASE_MODEL_REPO,
            local_dir=BASE_MODEL_PATH,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.onnx", "*.onnx_data"]
        )
        print("✓ SDXL base downloaded")
    else:
        print("✓ SDXL base already cached")
    results["base_model"] = BASE_MODEL_PATH
    
    # 2. Custom VAE (fp16 fix)
    if not os.path.exists(VAE_CONFIG_FILE):
        print(f"Downloading {HF_VAE_REPO}...")
        snapshot_download(
            HF_VAE_REPO,
            local_dir=VAE_PATH,
            local_dir_use_symlinks=False
        )
        print("✓ Custom VAE downloaded")
    else:
        print("✓ Custom VAE already cached")
    results["vae"] = VAE_PATH
    
    # 3. SPO LoRA
    if not os.path.exists(SPO_LORA_PATH):
        print(f"Downloading {HF_SPO_LORA_REPO}...")
        os.makedirs(SPO_LORA_DIR, exist_ok=True)
        hf_hub_download(
            repo_id=HF_SPO_LORA_REPO,
            filename=SPO_LORA_FILENAME,
            local_dir=SPO_LORA_DIR
        )
        print("✓ SPO LoRA downloaded")
    else:
        print("✓ SPO LoRA already cached")
    results["spo_lora"] = SPO_LORA_PATH
    
    omnilaunch_vol.commit()
    return {"ok": True, **results}


@app.function(
    image=image,
    volumes={"/omnilaunch": omnilaunch_vol},
    timeout=1800
)
def setup(run_downloads: bool = True) -> Dict[str, Any]:
    """Prepare SPO runner: verify env and download models."""
    import importlib
    from diffusers import __version__ as diffusers_version
    
    torch_present = importlib.util.find_spec("torch") is not None
    torch_version = None
    cuda_available = False
    
    if torch_present:
        import torch as _torch
        torch_version = str(_torch.__version__)
        cuda_available = bool(_torch.cuda.is_available())
        del _torch
    
    checks = {
        "torch_present": bool(torch_present),
        "torch_version": torch_version,
        "cuda_available": bool(cuda_available),
        "diffusers_version": str(diffusers_version),
        "volume_mounted": os.path.exists("/omnilaunch"),
        "base_model_present": os.path.exists(BASE_MODEL_INDEX_FILE),
        "vae_present": os.path.exists(VAE_CONFIG_FILE),
        "spo_lora_present": os.path.exists(SPO_LORA_PATH),
    }
    
    try:
        if run_downloads:
            dl_result = download_files.local()
            checks["download_result"] = dl_result
            checks["base_model_present"] = os.path.exists(BASE_MODEL_INDEX_FILE)
            checks["vae_present"] = os.path.exists(VAE_CONFIG_FILE)
            checks["spo_lora_present"] = os.path.exists(SPO_LORA_PATH)
        
        ok = (checks["volume_mounted"] and 
              checks["base_model_present"] and 
              checks["vae_present"] and 
              checks["spo_lora_present"])
        
        return {"ok": ok, "checks": checks}
    except Exception as e:
        return {"ok": False, "error": str(e), "checks": checks}


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/omnilaunch": omnilaunch_vol},
    timeout=900,
    scaledown_window=2
)
def infer(params: Dict[str, Any]) -> Dict[str, Any]:
    """SDXL SPO inference with aesthetic optimization.
    
    Based on: https://huggingface.co/SPO-Diffusion-Models/SPO-SDXL_4k-p_10ep_LoRA
    """
    import torch
    from diffusers import DiffusionPipeline, AutoencoderKL
    
    # Parse params
    prompt = str(params.get("prompt", "")).strip()
    if not prompt:
        return {"error": "prompt required"}
    
    negative = str(params.get("negative_prompt", ""))
    width = int(params.get("width", 1024))
    height = int(params.get("height", 1024))
    steps = int(params.get("steps", 25))
    guidance_scale = float(params.get("guidance_scale", 5.0))
    seed = int(params.get("seed", 42))
    
    # Load base pipeline
    print(f"Loading SDXL base pipeline...")
    pipe = DiffusionPipeline.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        local_files_only=True
    )
    
    # Load custom VAE (fp16 fix)
    print("Loading custom VAE...")
    vae = AutoencoderKL.from_pretrained(
        VAE_PATH,
        torch_dtype=torch.float16,
        local_files_only=True
    )
    pipe.vae = vae
    
    # Load SPO LoRA
    print("Loading SPO LoRA...")
    pipe.load_lora_weights(SPO_LORA_PATH)
    
    pipe.to("cuda")
    
    # Optional: enable memory optimizations
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        print("[-] XFormers not available")
    
    # Generate
    print(f"Generating: {prompt[:50]}...")
    gen = torch.Generator(device="cuda").manual_seed(seed)
    
    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative if negative else None,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=gen,
        )
    
    img_b64 = pil_image_to_base64(result.images[0])
    return {
        "content_type": "image/png",
        "data": img_b64,
    }


if __name__ == "__main__":
    app.deploy()

