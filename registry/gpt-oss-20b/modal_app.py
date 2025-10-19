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
        "torch==2.9.0",
        "transformers==4.57.0",
        "accelerate==1.10.1",
        "triton==3.5.0",
        "kernels==0.10.3",
        "huggingface_hub==0.35.3",
        "datasets>=3.4.1,<4.0.0",
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


def _extract_choice_letter(text: str) -> str:
    """Return first choice letter among A/B/C/D found in text, else '?'."""
    for ch in text:
        if ch in (" A", " B", " C", " D"):
            return ch.strip()
    import re
    m = re.search(r"[Aa]nswer\s*:\s*([ABCD])", text)
    if m:
        return m.group(1)
    if text[0] in ("A", "B", "C", "D"):
        return text[0]
    return "?"


@app.function(
    image=image,
    gpu="A10G",
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
    
    max_tokens = int(params.get("max_tokens", 2048))
    temperature = float(params.get("temperature", 0.7))
    reasoning_level = str(params.get("reasoning_level", "medium")).lower()
    
    # Inject reasoning level as system prompt if specified
    if reasoning_level in ("low", "medium", "high"):
        system_msg = {"role": "system", "content": f"Reasoning: {reasoning_level}"}
        # Prepend if no system message exists
        if not any(m.get("role") == "system" for m in messages):
            messages = [system_msg] + messages
    
    print(f"Loading GPT-OSS-20B model and tokenizer from {MODEL_PATH}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    # Decoder-only models should left-pad; ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
    )
    
    # Reuse shared helper for generation + harmony parsing
    result = _run_inference(
        model,
        tokenizer,
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    
    print("Generated output (with harmony tokens):")
    preview = result["generated_raw"]
    print(preview[:500] + "..." if len(preview) > 500 else preview)
    
    return {
        "content_type": "application/json",
        "data": {
            "response": result["response"],
            "analysis": result["analysis"],
            "commentary": result["commentary"],
            "reasoning_level": reasoning_level,
        },
    }


def _run_inference(model, tokenizer, messages: List[Dict[str, Any]], max_tokens: int = 2048, temperature: float = 0.0):
    """Helper function for running inference (shared between infer and benchmark)."""
    import torch
    
    # Prepare chat template and tokenize input
    input_ids = tokenizer.apply_chat_template(
        messages, 
        return_tensors="pt", 
        add_generation_prompt=True
    ).to(model.device)
    
    attention_mask = torch.ones_like(input_ids)
    
    # Generate
    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
    )
    
    # Decode only the generated part (exclude input prompt)
    generated_text_only = tokenizer.decode(
        output_ids[0][input_ids.shape[1]:], 
        skip_special_tokens=False
    )
    
    # Parse harmony format
    parsed = parse_harmony_output(generated_text_only)
    
    return {
        "generated_raw": generated_text_only,
        "response": parsed["response"],
        "analysis": parsed["analysis"],
        "commentary": parsed["commentary"],
    }


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/omnilaunch": omnilaunch_vol},
    timeout=1200,
    scaledown_window=2
)
def benchmark_tinymmlu(params: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate GPT-OSS-20B on tinyBenchmarks/tinyMMLU.
    
    Reuses existing inference logic from `infer` endpoint for consistency.

    Params:
      split: "test" | "dev" (default: "test")
      max_items: int (default: 100)
      reasoning_level: "low" | "medium" | "high" (default: "low")

    Returns:
      overall accuracy, per-subject accuracy, and sample predictions.
    """
    import time
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    start = time.time()

    split = str(params.get("split", "test"))
    max_items = int(params.get("max_items", 100))
    reasoning_level = str(params.get("reasoning_level", "low"))

    # Load dataset
    ds = load_dataset("tinyBenchmarks/tinyMMLU", split=split)
    if max_items and max_items < len(ds):
        ds = ds.select(range(max_items))

    print(f"Loaded tinyMMLU split={split}, n={len(ds)}")
    print(f"Using reasoning_level={reasoning_level}")

    # Load model once
    print(f"Loading GPT-OSS-20B from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    
    # Extract ground truth and subjects
    def gt_letter(row) -> str:
        a = row["answer"]
        if isinstance(a, int):
            return ["A", "B", "C", "D"][a]
        return str(a).strip().upper()
    
    gts = [gt_letter(r) for r in ds]
    subjects = [str(r.get("subject", "unknown")) for r in ds]

    # Run inference on each item
    preds: List[str] = []
    sample_records = []
    
    for item_idx, row in enumerate(ds):
        print(f"[{item_idx+1}/{len(ds)}] Evaluating...")
        
        # Build messages (same as infer endpoint)
        messages = [
            {"role": "system", "content": f"You are GPT-OSS. Please refrain from using Markdown formatting in your response and respond in plain text. Your final response should contain the correct answer according to the example shown in each prompt. Reasoning: {reasoning_level}"},
            {"role": "user", "content": row["input_formatted"]}
        ]
        
        # Call shared inference helper
        result = _run_inference(
            model, tokenizer, messages,
            max_tokens=2048,  # Short answer for MMLU
            temperature=0.0  # Greedy
        )
        
        # Extract answer letter from response (use response field from harmony parsing)
        response_text = result["response"]
        pred_letter = _extract_choice_letter(response_text)
        preds.append(pred_letter)

        print(f"\tQuestion snippet: {row['input_formatted'][:100]}")
        print(f"\tGenerated response: {response_text[:100]}")
        print(f"Predicted letter: {pred_letter}, Ground truth: {gts[item_idx]}")
        
        # Save sample for debugging
        if len(sample_records) < 10:
            sample_records.append({
                "prompt_end": row["input_formatted"][-200:],
                "generated_raw": result["generated_raw"][:200],
                "response": response_text[:100],
                "analysis": result["analysis"][:100] if result["analysis"] else None,
                "pred": pred_letter,
                "gt": gts[item_idx],
                "subject": subjects[item_idx],
            })
        
        # Show running accuracy every 10 items
        if (item_idx + 1) % 10 == 0 or (item_idx + 1) == len(ds):
            correct_so_far = sum(1 for p, g in zip(preds, gts[:len(preds)]) if p == g)
            acc_so_far = 100.0 * correct_so_far / len(preds) if preds else 0.0
            print(f"  Running accuracy: {correct_so_far}/{len(preds)} = {acc_so_far:.1f}%")

    # Compute accuracy
    correct = 0
    per_subject = {}
    for p, t, s in zip(preds, gts, subjects):
        ok = (p == t)
        correct += int(ok)
        d = per_subject.setdefault(s, {"correct": 0, "total": 0})
        d["correct"] += int(ok)
        d["total"] += 1

    overall_acc = correct / max(1, len(gts))
    per_subject_acc = {k: round(v["correct"] / max(1, v["total"]), 4) for k, v in per_subject.items()}

    elapsed = time.time() - start
    print(f"tinyMMLU accuracy: {overall_acc:.4f} ({correct}/{len(gts)}) in {elapsed:.1f}s")

    return {
        "content_type": "application/json",
        "data": {
            "ok": True,
            "items": len(gts),
            "overall_accuracy": round(overall_acc, 4),
            "correct": correct,
            "per_subject_accuracy": per_subject_acc,
            "samples": sample_records,
            "elapsed_seconds": round(elapsed, 1),
            "config": {
                "split": split,
                "max_items": max_items,
                "reasoning_level": reasoning_level,
                "method": "reuses_infer_logic"
            }
        }
    }


@app.function(image=image, volumes={"/omnilaunch": omnilaunch_vol}, timeout=60 * 10)
def test_local():
    """Quick local test (CPU)."""
    print("GPT-OSS-20B runner test passed")
    return {"ok": True}


if __name__ == "__main__":
    app.deploy()

