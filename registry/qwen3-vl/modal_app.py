import modal
from typing import Dict, Any
import os
import base64
import io

# ============================================================================
# Configuration
# ============================================================================

APP_NAME = "omnilaunch-qwen3-vl"
VOLUME_NAME = "omnilaunch"

# Model paths
HF_MODEL_REPO = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
MODEL_PATH = "/omnilaunch/models/unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"

# HuggingFace cache
HF_CACHE_DIR = "/omnilaunch/hf_cache"

app = modal.App(APP_NAME)
omnilaunch_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Optional WandB secret (gracefully handles if not set)
try:
    wandb_secret = modal.Secret.from_name("wandb-secret")
except Exception:
    wandb_secret = modal.Secret.from_dict({})

# ============================================================================
# Image Definition
# ============================================================================

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.0",
        "torchvision==0.20.0",
        "transformers==4.57.0",
        "trl==0.22.2",
        "bitsandbytes==0.48.1",
        "peft==0.17.1",
        "accelerate==1.10.1",
        "datasets>=3.4.1,<4.0.0",
        "huggingface_hub>=0.34.0",
        "pillow==12.0.0",
        "cut_cross_entropy==25.1.1",
        "xformers==0.0.28.post2",
        "sentencepiece==0.2.1",
        "protobuf==6.33.0",
        "wandb==0.22.2",  # Optional: For training metrics logging
        "coolname",  # Human-readable run names
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .run_commands(
        # Install unsloth (no deps to avoid conflicts)
        "pip install --no-deps unsloth",
        "pip install --no-deps unsloth_zoo",
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

def generate_run_name() -> str:
    """Generate a human-readable run name like 'brave-lion-1234'."""
    from coolname import generate_slug
    import time
    
    # Generate 2-word slug (adjective-noun)
    slug = generate_slug(2)
    
    # Add timestamp suffix for uniqueness
    timestamp = int(time.time()) % 10000  # Last 4 digits
    
    return f"{slug}-{timestamp}"


def decode_image(image_data: str):
    """Decode base64 image or load from path.
    
    Note: File paths must be accessible in the Modal container.
    For local files, encode to base64 first.
    """
    from PIL import Image
    s = (image_data or "").strip()
    
    # Handle data URI prefix
    if s.startswith("data:image"):
        s = s.split(",", 1)[1]
    
    # Try base64 decode FIRST (raw base64 often contains '/' characters)
    try:
        # Fix missing padding if necessary
        missing = (-len(s)) % 4
        if missing:
            s = s + ("=" * missing)
        image_bytes = base64.b64decode(s, validate=False)
        return Image.open(io.BytesIO(image_bytes))
    except Exception:
        # Fallback: try as file path inside the container
        if os.path.exists(image_data):
            print(f"Loading image from file: {image_data}")
            return Image.open(image_data)
        raise ValueError(
            "Could not decode image. Expected base64 string or accessible file path.\n"
            f"Input (first 100 chars): {str(image_data)[:100]}..."
        )


# ============================================================================
# Modal Functions
# ============================================================================

@app.function(
    image=image,
    volumes={"/omnilaunch": omnilaunch_vol},
    timeout=3600
)
def download_files() -> Dict[str, str]:
    """Download Qwen3-VL-8B model."""
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
    timeout=3600
)
def download_dataset(params: Dict[str, Any] = None) -> Dict[str, str]:
    """Download and cache HuggingFace dataset to volume (CPU, no GPU needed).
    
    params: {
      dataset_uri: "hf:unsloth/LaTeX_OCR" (default)
    }
    """
    from datasets import load_dataset
    
    params = params or {}
    dataset_uri = params.get("dataset_uri", "hf:unsloth/LaTeX_OCR")
    
    # Parse dataset URI
    if dataset_uri.startswith("hf:"):
        dataset_name = dataset_uri[3:]
    else:
        dataset_name = dataset_uri
    
    print(f"Downloading and caching dataset: {dataset_name}")
    print(f"Cache location: {HF_CACHE_DIR}")
    
    # Load dataset - this will cache it to HF_CACHE_DIR on the volume
    dataset = load_dataset(dataset_name, split="train")
    print(f"✓ Dataset cached: {len(dataset)} examples")
    
    omnilaunch_vol.commit()
    return {
        "ok": True, 
        "dataset": dataset_name,
        "num_examples": len(dataset),
        "cache_dir": HF_CACHE_DIR
    }


@app.function(
    image=image,
    volumes={"/omnilaunch": omnilaunch_vol},
    timeout=1800
)
def setup(run_downloads: bool = True) -> Dict[str, Any]:
    """Prepare Qwen3-VL runner: verify env and download model."""
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
    
    unsloth_present = importlib.util.find_spec("unsloth") is not None
    
    checks = {
        "torch_present": bool(torch_present),
        "torch_version": torch_version,
        "cuda_available": bool(cuda_available),
        "transformers_present": bool(transformers_present),
        "transformers_version": transformers_version,
        "unsloth_present": bool(unsloth_present),
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
        
        ok = checks["volume_mounted"] and checks["model_present_after"] and checks["unsloth_present"]
        return {"ok": ok, "checks": checks}
    except Exception as e:
        return {"ok": False, "error": str(e), "checks": checks}


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/omnilaunch": omnilaunch_vol},
    timeout=600,
    scaledown_window=2
)
def infer(params: Dict[str, Any]) -> Dict[str, Any]:
    """Qwen3-VL-8B vision-language inference.
    
    params: {
      image: str (base64 or path),
      prompt: str,
      lora_run: str (optional, e.g. "brave-lion-1234"),
      max_tokens: 128,
      temperature: 1.5,
      min_p: 0.1
    }
    """
    from unsloth import FastVisionModel
    
    image_data = params.get("image")
    prompt = params.get("prompt", "Describe this image.")
    lora_run = params.get("lora_run")
    max_tokens = int(params.get("max_tokens", 128))
    temperature = float(params.get("temperature", 1.5))
    min_p = float(params.get("min_p", 0.1))
    
    if not image_data:
        return {"error": "image required"}
    
    print(f"Loading Qwen3-VL-8B model from {MODEL_PATH}...")
    model, tokenizer = FastVisionModel.from_pretrained(
        MODEL_PATH,
        load_in_4bit=True,
    )
    
    # Load LoRA adapters if specified
    if lora_run:
        lora_path = f"/omnilaunch/runs/qwen3-vl/{lora_run}"
        if os.path.exists(lora_path):
            print(f"Loading LoRA adapters from: {lora_path}")
            model.load_adapter(lora_path)
        else:
            print(f"⚠ LoRA run '{lora_run}' not found at {lora_path}, using base model")
    
    FastVisionModel.for_inference(model)
    
    # Decode image
    image = decode_image(image_data)
    
    # Format messages in vision format
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt}
        ]}
    ]
    
    # Apply chat template and tokenize
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")
    
    print(f"Generating response (max_tokens={max_tokens}, temp={temperature}, min_p={min_p})...")
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        use_cache=True,
        temperature=temperature if temperature > 0 else None,
        do_sample=temperature > 0,
        min_p=min_p if temperature > 0 else None,
    )
    
    # Decode only the generated part
    response = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    
    print(f"Response: {response}")
    
    return {
        "content_type": "application/json",
        "data": {"response": response}
    }


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/omnilaunch": omnilaunch_vol},
    secrets=[wandb_secret],
    timeout=7200,
)
def train_lora(params: Dict[str, Any]) -> Dict[str, Any]:
    """Fine-tune Qwen3-VL-8B with LoRA using Unsloth.
    
    params: {
      dataset_uri: "hf:unsloth/LaTeX_OCR" (or path to dataset),
      run_name: "my-experiment" (optional, auto-generated if not provided),
      steps: 30,
      epochs: 1 (alternative to steps),
      learning_rate: 2e-4,
      lora_r: 16,
      lora_alpha: 16,
      batch_size: 2,
      gradient_accumulation_steps: 4,
      max_train_samples: None (limit dataset for quick tests),
      instruction: "Write the LaTeX representation for this image." (optional)
    }
    """
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset
    import wandb
    
    # Parse dataset URI
    dataset_uri = params.get("dataset_uri", "hf:unsloth/LaTeX_OCR")
    if dataset_uri.startswith("hf:"):
        dataset_name = dataset_uri[3:]
    else:
        dataset_name = dataset_uri
    
    # Generate or use provided run name
    run_name = params.get("run_name")
    if not run_name:
        run_name = generate_run_name()
    
    # Output path: /omnilaunch/runs/qwen3-vl/<run-name>/
    output_path = f"/omnilaunch/runs/qwen3-vl/{run_name}"
    
    steps = int(params.get("steps", 30))
    epochs = params.get("epochs")
    try:
        epochs = float(epochs) if epochs is not None else None
    except Exception:
        epochs = None
    learning_rate = float(params.get("learning_rate", 2e-4))
    lora_r = int(params.get("lora_r", 16))
    lora_alpha = int(params.get("lora_alpha", 16))
    batch_size = int(params.get("batch_size", 2))
    gradient_accumulation_steps = int(params.get("gradient_accumulation_steps", 4))
    instruction = params.get("instruction", "Write the LaTeX representation for this image.")
    max_train_samples = params.get("max_train_samples")
    try:
        max_train_samples = int(max_train_samples) if max_train_samples is not None else None
    except Exception:
        max_train_samples = None
    
    print(f"Training run: {run_name}")
    print(f"Output path: {output_path}")
    
    print(f"Loading model from {MODEL_PATH}...")
    model, tokenizer = FastVisionModel.from_pretrained(
        MODEL_PATH,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    
    print("Adding LoRA adapters...")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    print(f"Loading dataset: {dataset_name}...")
    dataset = load_dataset(dataset_name, split="train")
    # Optionally limit dataset size for quick tests
    if max_train_samples is not None and max_train_samples > 0:
        dataset = dataset.select(range(min(max_train_samples, len(dataset))))
    
    print("Converting dataset to conversation format...")
    def convert_to_conversation(sample):
        return {
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": sample["image"]}
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": sample["text"]}
                ]}
            ]
        }
    
    converted_dataset = [convert_to_conversation(sample) for sample in dataset]

    # Compute steps for epoch-based training if requested
    if epochs is not None and epochs > 0:
        import math
        effective_batch_size = batch_size * gradient_accumulation_steps
        total_examples = len(converted_dataset)
        computed_steps = max(1, int(math.ceil((total_examples * epochs) / max(1, effective_batch_size))))
        print(f"Computed steps from epochs={epochs}: {computed_steps} (dataset={total_examples}, effective_bs={effective_batch_size})")
        steps = computed_steps
    
    print(f"Training for {steps} steps...")
    FastVisionModel.for_training(model)
    
    # Optional WandB integration (if WANDB_API_KEY is set in Modal secrets)
    use_wandb = False
    if os.environ.get("WANDB_API_KEY"):
        try:
            wandb.init(
                project="omnilaunch-qwen3-vl",
                config={
                    "model": "Qwen3-VL-8B-Instruct",
                    "dataset": dataset_name,
                    "lora_r": lora_r,
                    "lora_alpha": lora_alpha,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "max_steps": steps,
                    "epochs": epochs,
                    "max_train_samples": max_train_samples,
                }
            )
            use_wandb = True
            print("✓ WandB logging enabled")
        except Exception as e:
            print(f"⚠ WandB initialization failed: {e}. Continuing without logging.")
    else:
        print("⚠ WANDB_API_KEY not found. Skipping WandB logging (set wandb-secret in Modal to enable).")
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=converted_dataset,
        args=SFTConfig(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=5,
            max_steps=steps,
            learning_rate=learning_rate,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_path,
            report_to="wandb" if use_wandb else "none",
            # Required for vision finetuning
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=2048,
        ),
    )
    
    # Show GPU stats
    import torch
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    
    trainer_stats = trainer.train()
    
    # Show final stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    
    # Save LoRA adapters
    print(f"Saving LoRA adapters to {output_path}...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    omnilaunch_vol.commit()
    
    # Finish WandB run if active
    if use_wandb:
        try:
            wandb.finish()
        except Exception:
            pass
    
    return {
        "content_type": "application/json",
        "data": {
            "ok": True,
            "run_name": run_name,
            "output_path": output_path,
            "train_runtime_seconds": trainer_stats.metrics['train_runtime'],
            "train_runtime_minutes": round(trainer_stats.metrics['train_runtime'] / 60, 2),
            "train_loss": trainer_stats.metrics.get('train_loss'),
            "peak_memory_gb": used_memory,
            "lora_memory_gb": used_memory_for_lora,
        }
    }


@app.function(image=image, volumes={"/omnilaunch": omnilaunch_vol}, timeout=60)
def test_local():
    """Quick local test (CPU)."""
    print("Qwen3-VL runner test passed")
    return {"ok": True}


if __name__ == "__main__":
    app.deploy()

