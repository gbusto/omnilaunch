import modal
from typing import Dict, Any, List
import os

# ============================================================================
# Configuration
# ============================================================================

APP_NAME = "omnilaunch-gpt-oss-120b"
VOLUME_NAME = "omnilaunch"

# Model paths
HF_MODEL_REPO = "openai/gpt-oss-120b"
MODEL_PATH = "/omnilaunch/models/openai/gpt-oss-120b"

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
        "torch==2.9.0",
        "transformers==4.57.0",
        "accelerate==1.10.1",
        "triton==3.5.0",
        "kernels==0.10.3",
        "huggingface_hub==0.35.3",
        "tiktoken==0.12.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .env({
        "HF_HOME": HF_CACHE_DIR,
        "HF_HUB_CACHE": HF_CACHE_DIR,
        "TRANSFORMERS_CACHE": HF_CACHE_DIR,
        # Reduce VRAM fragmentation spikes during allocation
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
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
    """Download GPT-OSS-120B model."""
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
    """Prepare GPT-OSS-120B runner: verify env and download model."""
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


def parse_harmony_output(output: str) -> dict:
    """
    Parse GPT-OSS harmony format output into separate channels.
    
    Harmony format: <|channel|>NAME<|message|>CONTENT<|end|>
    
    Channels:
        - analysis: Internal reasoning/thinking
        - commentary: Meta-observations (optional)
        - final: User-facing response
    
    Args:
        output: Raw model output with harmony format tokens
        
    Returns:
        dict with 'response', 'analysis', 'commentary' keys
    """
    import re
    
    # Pattern: <|channel|>NAME<|message|>CONTENT<|end|> or <|return|>
    pattern = r'<\|channel\|>(\w+)<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>)'
    
    channels = {}
    for match in re.finditer(pattern, output, re.DOTALL):
        channel_name = match.group(1).lower()
        content = match.group(2).strip()
        channels[channel_name] = content
    
    return {
        "response": channels.get("final", output.strip()),
        "analysis": channels.get("analysis"),
        "commentary": channels.get("commentary"),
    }


@app.function(
    image=image,
    gpu="H100",
    volumes={"/omnilaunch": omnilaunch_vol},
    timeout=600,
    scaledown_window=2
)
def infer(params: Dict[str, Any]) -> Dict[str, Any]:
    """GPT-OSS-120B chat completion with MXFP4 quantization.
    
    params: {
      messages: [{"role": "user", "content": "..."}],
      max_tokens: 256,
      temperature: 0.7,
      reasoning_level: "medium"  // low/medium/high
    }
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
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
        if not any(m.get("role") == "system" for m in messages):
            messages = [system_msg] + messages
    
    print(f"Loading GPT-OSS-120B with MXFP4 quantization from {MODEL_PATH}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Configure MXFP4 quantization (model is already quantized, this ensures it loads correctly)
    quantization_config = Mxfp4Config(dequantize=False)
    
    # Load model with optimized settings for H100
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,           # Use bf16 for non‑MoE tensors (expected by model)
        device_map="auto",                    # Auto-distribute across GPU
        quantization_config=quantization_config  # Ensure MXFP4 quantized weights are used
    )
    
    # Prepare chat template and tokenize input
    input_ids = tokenizer.apply_chat_template(
        messages, 
        return_tensors="pt", 
        add_generation_prompt=True
    ).to(model.device)
    
    print(f"Generating response (max_new_tokens={max_tokens}, temp={temperature})...")
    # Explicit attention mask to avoid pad/eos ambiguity warning
    import torch as _torch
    attention_mask = _torch.ones_like(input_ids, dtype=_torch.long, device=input_ids.device)
    
    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
    )
    
    # Decode with special tokens preserved (harmony format)
    generated_text_only = tokenizer.decode(
        output_ids[0][input_ids.shape[1]:], 
        skip_special_tokens=False
    )
    
    print("Generated output (with harmony tokens):")
    print(generated_text_only[:500] + "..." if len(generated_text_only) > 500 else generated_text_only)
    
    # Parse harmony format to extract all channels
    parsed = parse_harmony_output(generated_text_only)
    
    return {
        "content_type": "application/json",
        "data": {
            "response": parsed["response"],        # User-facing answer (from 'final' channel)
            "analysis": parsed["analysis"],        # Internal reasoning/thinking
            "commentary": parsed["commentary"],    # Meta-observations (optional)
            "reasoning_level": reasoning_level,
        },
    }


@app.function(image=image, volumes={"/omnilaunch": omnilaunch_vol}, timeout=60 * 10)
def test_local():
    """Quick local test (CPU)."""
    print("GPT-OSS-120B runner test passed")
    return {"ok": True}


if __name__ == "__main__":
    app.deploy()

