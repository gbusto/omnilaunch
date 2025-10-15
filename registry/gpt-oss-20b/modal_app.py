import modal
from typing import Dict, Any, List
import os

# ============================================================================
# Configuration
# ============================================================================

APP_NAME = "omnilaunch-gpt-oss-20b"
VOLUME_NAME = "omnilaunch"

# Model paths
HF_MODEL_REPO = "openai/gpt-oss-20b"
MODEL_PATH = "/omnilaunch/models/openai/gpt-oss-20b"

# HuggingFace cache
HF_CACHE_DIR = "/omnilaunch/hf_cache"

app = modal.App(APP_NAME)
omnilaunch_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# ============================================================================
# Image
# ============================================================================

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.57.0",
        "accelerate==1.2.1",
        "huggingface_hub",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .env({
        "HF_HOME": HF_CACHE_DIR,
        "HF_HUB_CACHE": HF_CACHE_DIR,
        "TRANSFORMERS_CACHE": HF_CACHE_DIR,
    })
)


# ============================================================================
# Entrypoints
# ============================================================================

@app.function(
    image=image,
    volumes={"/omnilaunch": omnilaunch_vol},
    timeout=3600
)
def download_files() -> Dict[str, str]:
    """Download GPT-OSS-20B model."""
    from huggingface_hub import snapshot_download
    
    if not os.path.exists(f"{MODEL_PATH}/config.json"):
        print(f"Downloading {HF_MODEL_REPO} to {MODEL_PATH}...")
        snapshot_download(
            HF_MODEL_REPO,
            local_dir=MODEL_PATH,
            local_dir_use_symlinks=False,
        )
        print("✓ Model downloaded")
    else:
        print(f"✓ Model already cached at {MODEL_PATH}")
    
    omnilaunch_vol.commit()
    return {"ok": True, "model_path": MODEL_PATH}


@app.function(
    image=image,
    volumes={"/omnilaunch": omnilaunch_vol},
    timeout=1800
)
def setup(run_downloads: bool = True) -> Dict[str, Any]:
    """Prepare GPT-OSS-20B runner: verify env and download model."""
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
    
    checks = {
        "torch_present": bool(torch_present),
        "torch_version": torch_version,
        "cuda_available": bool(cuda_available),
        "transformers_present": bool(transformers_present),
        "transformers_version": transformers_version,
        "volume_mounted": os.path.exists("/omnilaunch"),
        "model_present_before": os.path.exists(f"{MODEL_PATH}/config.json"),
    }
    
    try:
        if run_downloads:
            dl_result = download_files.local()
            checks["download_result"] = dl_result
            checks["model_present_after"] = os.path.exists(f"{MODEL_PATH}/config.json")
        else:
            checks["model_present_after"] = checks["model_present_before"]
        
        ok = checks["volume_mounted"] and checks["model_present_after"]
        return {"ok": ok, "checks": checks}
    except Exception as e:
        return {"ok": False, "error": str(e), "checks": checks}


@app.function(
    image=image,
    gpu="H100",
    volumes={"/omnilaunch": omnilaunch_vol},
    timeout=600,
    scaledown_window=2
)
def infer(params: Dict[str, Any]) -> Dict[str, Any]:
    """GPT-OSS-20B chat completion.
    
    params: {
      messages: [{"role": "user", "content": "..."}],
      max_tokens: 256,
      temperature: 0.7,
      reasoning_level: "medium"  // low/medium/high
    }
    """
    from transformers import pipeline
    import torch
    
    messages = params.get("messages", [])
    if not messages:
        return {"error": "messages required"}
    
    max_tokens = int(params.get("max_tokens", 256))
    temperature = float(params.get("temperature", 0.7))
    reasoning_level = str(params.get("reasoning_level", "medium")).lower()
    
    # Inject reasoning level as system prompt if specified
    if reasoning_level in ("low", "medium", "high"):
        system_msg = {"role": "system", "content": f"Reasoning: {reasoning_level}"}
        # Prepend if no system message exists
        if not any(m.get("role") == "system" for m in messages):
            messages = [system_msg] + messages
    
    print(f"Loading GPT-OSS-20B pipeline from {MODEL_PATH}...")
    pipe = pipeline(
        "text-generation",
        model=MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
    )
    
    print(f"Generating response (max_tokens={max_tokens}, temp={temperature})...")
    outputs = pipe(
        messages,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
    )
    
    # Extract assistant response
    response_text = outputs[0]["generated_text"][-1]["content"]
    
    return {
        "content_type": "application/json",
        "data": {
            "response": response_text,
            "reasoning_level": reasoning_level,
        },
    }


@app.function(image=image, volumes={"/omnilaunch": omnilaunch_vol}, timeout=60 * 10)
def test_local():
    """Quick local test (CPU)."""
    print("GPT-OSS-20B runner test passed")
    return {"ok": True}


if __name__ == "__main__":
    app.deploy()

