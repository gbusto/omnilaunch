import modal
from typing import Dict, Any
import os

# ============================================================================
# Configuration
# ============================================================================

# App and volume
APP_NAME = "omnilaunch-sdxl-pixel-art"
VOLUME_NAME = "omnilaunch"

# Base model
HF_BASE_MODEL_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
BASE_MODEL_PATH = "/omnilaunch/models/stabilityai/stable-diffusion-xl-base-1.0"
BASE_MODEL_INDEX_FILE = f"{BASE_MODEL_PATH}/model_index.json"

# LCM LoRA
HF_LCM_LORA_REPO = "latent-consistency/lcm-lora-sdxl"
LCM_LORA_FILENAME = "pytorch_lora_weights.safetensors"
LCM_LORA_DIR = "/omnilaunch/loras/lcm-lora-sdxl"
LCM_LORA_PATH = f"{LCM_LORA_DIR}/{LCM_LORA_FILENAME}"

# Pixel Art LoRA
HF_PIXEL_LORA_REPO = "nerijs/pixel-art-xl"
PIXEL_LORA_FILENAME = "pixel-art-xl.safetensors"
PIXEL_LORA_DIR = "/omnilaunch/loras/pixel-art-xl"
PIXEL_LORA_PATH = f"{PIXEL_LORA_DIR}/{PIXEL_LORA_FILENAME}"

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
    """Download SDXL base model + LCM LoRA + Pixel Art LoRA."""
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
    
    # 2. LCM LoRA
    if not os.path.exists(LCM_LORA_PATH):
        print(f"Downloading {HF_LCM_LORA_REPO}...")
        os.makedirs(LCM_LORA_DIR, exist_ok=True)
        hf_hub_download(
            repo_id=HF_LCM_LORA_REPO,
            filename=LCM_LORA_FILENAME,
            local_dir=LCM_LORA_DIR
        )
        print("✓ LCM LoRA downloaded")
    else:
        print("✓ LCM LoRA already cached")
    results["lcm_lora"] = LCM_LORA_PATH
    
    # 3. Pixel Art LoRA
    if not os.path.exists(PIXEL_LORA_PATH):
        print(f"Downloading {HF_PIXEL_LORA_REPO}...")
        os.makedirs(PIXEL_LORA_DIR, exist_ok=True)
        hf_hub_download(
            repo_id=HF_PIXEL_LORA_REPO,
            filename=PIXEL_LORA_FILENAME,
            local_dir=PIXEL_LORA_DIR
        )
        print("✓ Pixel Art LoRA downloaded")
    else:
        print("✓ Pixel Art LoRA already cached")
    results["pixel_lora"] = PIXEL_LORA_PATH
    
    omnilaunch_vol.commit()
    return {"ok": True, **results}


@app.function(
    image=image,
    volumes={"/omnilaunch": omnilaunch_vol},
    timeout=1800
)
def setup(run_downloads: bool = True) -> Dict[str, Any]:
    """Prepare pixel-art runner: verify env and download models."""
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
        "lcm_lora_present": os.path.exists(LCM_LORA_PATH),
        "pixel_lora_present": os.path.exists(PIXEL_LORA_PATH),
    }
    
    try:
        if run_downloads:
            dl_result = download_files.local()
            checks["download_result"] = dl_result
            checks["base_model_present"] = os.path.exists(BASE_MODEL_INDEX_FILE)
            checks["lcm_lora_present"] = os.path.exists(LCM_LORA_PATH)
            checks["pixel_lora_present"] = os.path.exists(PIXEL_LORA_PATH)
        
        ok = (checks["volume_mounted"] and 
              checks["base_model_present"] and 
              checks["lcm_lora_present"] and 
              checks["pixel_lora_present"])
        
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
    """SDXL Pixel Art inference with LCM + Pixel Art LoRA.
    
    Based on: https://huggingface.co/nerijs/pixel-art-xl
    """
    import torch
    from diffusers import DiffusionPipeline, LCMScheduler
    
    # Parse params
    prompt = str(params.get("prompt", "")).strip()
    if not prompt:
        return {"error": "prompt required"}
    
    # Add "pixel" prefix if not present (recommended per HF docs)
    if not prompt.lower().startswith("pixel"):
        prompt = f"pixel, {prompt}"
    
    negative = str(params.get("negative_prompt", "3d render, realistic"))
    width = int(params.get("width", 1024))
    height = int(params.get("height", 1024))
    steps = int(params.get("steps", 25))  # Optimized for LCM
    guidance_scale = float(params.get("guidance_scale", 1.5))  # Optimized for LCM
    seed = int(params.get("seed", 42))
    lora_strength = float(params.get("lora_strength", 1.2))
    
    # Load base pipeline
    print(f"Loading SDXL base pipeline...")
    pipe = DiffusionPipeline.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        local_files_only=True
    )
    
    # Set LCM scheduler
    print("Setting LCM scheduler...")
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    
    # Load LoRAs with adapters
    print("Loading LCM LoRA...")
    pipe.load_lora_weights(LCM_LORA_PATH, adapter_name="lcm")
    
    print("Loading Pixel Art LoRA...")
    pipe.load_lora_weights(PIXEL_LORA_PATH, adapter_name="pixel")
    
    # Set adapter weights
    pipe.set_adapters(["lcm", "pixel"], adapter_weights=[1.0, lora_strength])
    
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
            negative_prompt=negative,
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

