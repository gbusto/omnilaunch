import modal
from typing import Dict, Any

# App name follows omnilaunch-<runner> convention
app = modal.App("omnilaunch-sdxl")

# Single shared volume as per OMNILAUNCH_PLAN.md
omnilaunch_vol = modal.Volume.from_name("omnilaunch", create_if_missing=True)

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
)


@app.function(
    image=image,
    volumes={"/omnilaunch": omnilaunch_vol},
    timeout=3600
)
def download_files() -> Dict[str, str]:
    """Download SDXL base model to /omnilaunch/models/stabilityai/stable-diffusion-xl-base-1.0"""
    from huggingface_hub import snapshot_download
    import os

    # Follow OMNILAUNCH_PLAN.md: /omnilaunch/models/<hf_user>/<hf_repo>
    MODEL_PATH = "/omnilaunch/models/stabilityai/stable-diffusion-xl-base-1.0"
    MODEL_REPO = "stabilityai/stable-diffusion-xl-base-1.0"

    if not os.path.exists(f"{MODEL_PATH}/model_index.json"):
        print(f"Downloading {MODEL_REPO} to {MODEL_PATH}...")
        snapshot_download(
            MODEL_REPO,
            local_dir=MODEL_PATH,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.onnx", "*.onnx_data"]
        )
        print("Download complete.")
    else:
        print(f"Model already exists at {MODEL_PATH}")

    omnilaunch_vol.commit()
    return {"ok": True, "model_path": MODEL_PATH}


@app.function(
    image=image,
    volumes={"/omnilaunch": omnilaunch_vol},
    timeout=1800
)
def setup(run_downloads: bool = True) -> Dict[str, Any]:
    """Prepare SDXL runner on Modal: verify env and optionally cache models.

    - Verifies torch/cuda/diffusers
    - Ensures /omnilaunch volume is mounted
    - Calls download_files() to cache SDXL base model (if run_downloads=True)
    """
    import os
    import importlib
    from diffusers import __version__ as diffusers_version

    MODEL_PATH = "/omnilaunch/models/stabilityai/stable-diffusion-xl-base-1.0"

    torch_present = importlib.util.find_spec("torch") is not None
    torch_version = None
    cuda_available = False
    cuda_device_count = 0
    if torch_present:
        # Import inside, then extract only primitive values to avoid returning torch types
        import torch as _torch
        torch_version = str(_torch.__version__)
        cuda_available = bool(_torch.cuda.is_available())
        cuda_device_count = int(_torch.cuda.device_count()) if _torch.cuda.is_available() else 0
        del _torch

    checks = {
        "torch_present": bool(torch_present),
        "torch_version": torch_version,
        "cuda_available": bool(cuda_available),
        "cuda_device_count": int(cuda_device_count),
        "diffusers_version": str(diffusers_version),
        "volume_mounted": os.path.exists("/omnilaunch"),
        "model_present_before": os.path.exists(f"{MODEL_PATH}/model_index.json"),
    }

    try:
        if run_downloads:
            # Call download_files.local() to execute in the same container
            dl_result = download_files.local()
            checks["download_result"] = dl_result
            checks["model_present_after"] = os.path.exists(f"{MODEL_PATH}/model_index.json")
        else:
            checks["model_present_after"] = checks["model_present_before"]
        
        ok = checks["volume_mounted"] and checks["model_present_after"]
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
    """SDXL inference using diffusers library.
    
    params: {
      prompt: str,
      negative_prompt?: str,
      width?: int, height?: int,
      steps?: int, guidance_scale?: float, seed?: int
    }
    """
    import torch
    from diffusers import DiffusionPipeline
    from PIL import Image
    import io, base64

    def _pil_to_b64(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    MODEL_PATH = "/omnilaunch/models/stabilityai/stable-diffusion-xl-base-1.0"

    # Parse params
    prompt = str(params.get("prompt", "")).strip()
    if not prompt:
        return {"error": "prompt required"}
    
    negative = str(params.get("negative_prompt", ""))
    width = int(params.get("width", 1024))
    height = int(params.get("height", 1024))
    steps = int(params.get("steps", 25))
    guidance_scale = float(params.get("guidance_scale", 7.5))
    seed = int(params.get("seed", 42))

    # Load pipeline as per HF docs
    print(f"Loading SDXL pipeline from {MODEL_PATH}...")
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        local_files_only=True
    )
    pipe.to("cuda")
    
    # Optional: enable memory optimizations
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        print("[-] NOTE: XFormers memory efficient attention not available")

    # Generate
    print(f"Generating image: {prompt[:50]}...")
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
    
    # Return single-payload: content_type + base64 data
    img_b64 = _pil_to_b64(result.images[0])
    return {
        "content_type": "image/png",
        "data": img_b64,
    }


if __name__ == "__main__":
    app.deploy()


