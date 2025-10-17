# Qwen3-VL-8B

Vision-language model for image understanding and generation tasks. Fine-tuned with Unsloth for 2x faster training.

## Quick Start

```bash
# 1. Build runner
omni build omnilaunch/registry/qwen3-vl

# 2. Setup (download model, verify env)
omni setup omnilaunch/qwen3-vl:0.1.0

# 3. (Optional) Pre-download dataset on CPU to save GPU costs
omni run omnilaunch/qwen3-vl:0.1.0 download_dataset

# 4. Run inference on an image
omni run omnilaunch/qwen3-vl:0.1.0 infer \
  -p image="path/to/image.jpg" \
  -p prompt="What's in this image?" \
  --save --outfile response.json

# 5. Fine-tune on LaTeX OCR (auto-generates run name like "brave-lion-1234")
omni run omnilaunch/qwen3-vl:0.1.0 train_lora \
  -p dataset_uri="hf:unsloth/LaTeX_OCR" \
  -p epochs=1 \
  -p max_train_samples=200 \
  --save --outfile training_results.json
# Returns: {"run_name": "brave-lion-1234", "output_path": "/omnilaunch/runs/qwen3-vl/brave-lion-1234", ...}

# 6. Use your trained LoRA for inference
omni run omnilaunch/qwen3-vl:0.1.0 infer \
  -p image="math_problem.jpg" \
  -p prompt="Write the LaTeX for this." \
  -p lora_run="brave-lion-1234" \
  --save
```

## Entrypoints

### `download_files`
Downloads the `unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit` model from HuggingFace to the volume.

**GPU:** CPU only  
**Timeout:** 3600s  
**Cost:** ~$0.003

### `download_dataset`
Pre-downloads and caches a HuggingFace dataset to the volume on CPU (no GPU needed). **Recommended before training** to avoid expensive GPU time for dataset downloads.

**GPU:** CPU only  
**Timeout:** 3600s  
**Cost:** ~$0.003-0.01 depending on dataset size

**Parameters:**
- `dataset_uri` (default: "hf:unsloth/LaTeX_OCR"): HuggingFace dataset to cache

**Example:**
```bash
# Cache default LaTeX dataset
omni run omnilaunch/qwen3-vl:0.1.0 download_dataset

# Cache a different dataset
omni run omnilaunch/qwen3-vl:0.1.0 download_dataset \
  -p dataset_uri="hf:your/dataset"
```

### `setup`
Verifies environment (torch, CUDA, transformers, unsloth) and calls `download_files` to cache the model.

**GPU:** CPU only  
**Timeout:** 1800s  
**Cost:** ~$0.003

### `infer`
Vision-language inference - ask questions about images, describe images, extract text, etc.

**GPU:** A10G (24GB)  
**Timeout:** 600s

**Parameters:**
- `image` (required): Base64 encoded image or file path
- `prompt` (required): Text prompt/question about the image
- `lora_run` (optional): Run name of trained LoRA to use (e.g., "brave-lion-1234")
- `max_tokens` (default: 128): Maximum tokens to generate
- `temperature` (default: 1.5): Sampling temperature
- `min_p` (default: 0.1): Minimum probability threshold

**Returns:**
```json
{
  "response": "The image shows a handwritten mathematical formula..."
}
```

**Using trained LoRA:**
```bash
# After training, use the run_name from training output
omni run omnilaunch/qwen3-vl:0.1.0 infer \
  -p image="test.jpg" \
  -p prompt="Describe this." \
  -p lora_run="brave-lion-1234" \
  --save
```

### `train_lora`
Fine-tune the vision-language model with LoRA using Unsloth optimizations.

**GPU:** A10G (24GB)  
**Timeout:** 7200s (2 hours)

**Parameters:**
- `dataset_uri` (default: "hf:unsloth/LaTeX_OCR"): Dataset for training
- `run_name` (optional): Human-readable name for this training run (auto-generated if not provided, e.g., "brave-lion-1234")
- `steps` (default: 30): Number of training steps
- `epochs` (optional): Compute steps from epochs (overrides `steps` if provided)
- `max_train_samples` (optional): Limit dataset size for quick tests
- `learning_rate` (default: 2e-4): Learning rate
- `lora_r` (default: 16): LoRA rank
- `lora_alpha` (default: 16): LoRA alpha
- `batch_size` (default: 2): Per-device batch size
- `gradient_accumulation_steps` (default: 4): Gradient accumulation
- `instruction` (default: "Write the LaTeX representation for this image."): Task instruction

**Outputs saved to:** `/omnilaunch/runs/qwen3-vl/<run_name>/`

**Returns:**
```json
{
  "ok": true,
  "run_name": "brave-lion-1234",
  "output_path": "/omnilaunch/runs/qwen3-vl/brave-lion-1234",
  "train_runtime_seconds": 370.6,
  "train_runtime_minutes": 6.2,
  "train_loss": 0.119,
  "peak_memory_gb": 8.2,
  "lora_memory_gb": 0.5
}
```

**Tip:** Copy the `run_name` from the output and use it with `-p lora_run=<run_name>` in `infer` to test your trained model!

## Examples

### Image Description
```bash
omni run omnilaunch/qwen3-vl:0.1.0 infer \
  -p image="vacation_photo.jpg" \
  -p prompt="Describe this image in detail." \
  --save
```

### OCR (Text Extraction)
```bash
omni run omnilaunch/qwen3-vl:0.1.0 infer \
  -p image="document.png" \
  -p prompt="Extract all text from this image." \
  --save
```

### LaTeX Formula Recognition
```bash
omni run omnilaunch/qwen3-vl:0.1.0 infer \
  -p image="math_formula.jpg" \
  -p prompt="Write the LaTeX representation for this image." \
  --save
```

### Fine-tune for Custom Vision Task
```bash
# Recommended workflow: download dataset on CPU first to save GPU costs
omni run omnilaunch/qwen3-vl:0.1.0 download_dataset \
  -p dataset_uri="hf:yourname/custom-vision-dataset"

# Then train on GPU (dataset already cached)
omni run omnilaunch/qwen3-vl:0.1.0 train_lora \
  -p dataset_uri="hf:yourname/custom-vision-dataset" \
  -p instruction="Describe this medical image." \
  -p steps=100 \
  -p output_path="/omnilaunch/runs/medical-vision-lora"
```

### Quick Test Training (1 epoch on small subset)
```bash
# Download dataset first (CPU, cheap)
omni run omnilaunch/qwen3-vl:0.1.0 download_dataset

# Quick test: 1 epoch on 200 examples
omni run omnilaunch/qwen3-vl:0.1.0 train_lora \
  -p epochs=1 \
  -p max_train_samples=200 \
  --save
```

## Comparison to Unsloth Notebook

This runner replicates [Unsloth's Qwen3-VL notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_VL_(8B)-Vision.ipynb) with production improvements:

| Feature | Unsloth Colab | Omnilaunch Runner |
|---------|---------------|-------------------|
| **Setup** | Manual: install deps, auth HF, download model (~5-10 min) | One command: `omni setup` (~5 min first time, instant after) |
| **Environment** | Google Colab (Tesla T4, 15GB VRAM, session limits) | Modal (A10G, 24GB VRAM, no time limits) |
| **Dataset prep** | Manual load + format to conversation structure | Auto: `download_dataset` (CPU), auto-format in training |
| **Training** | Run cells sequentially, restart kernel if OOM | Single command: `omni run ... train_lora` |
| **Checkpoints** | Saves to `/content/model` (lost on disconnect) | Persists to Modal volume (permanent) |
| **Reproducibility** | Manual version management, cell execution order matters | Pinned deps, versioned runner, single entrypoint |
| **Parameters** | Hardcoded in cells (must edit code) | CLI flags: `-p epochs=1 -p lora_r=32` |
| **Image input** | Manual base64 encoding in code | Auto-encodes local files: `-p image="photo.jpg"` |
| **Interruption** | Lose progress if disconnected | Runs on Modal (disconnect-safe) |
| **Cost** | Free (limited GPU time, queues) | ~$0.13 for quick test, ~$0.65 for 100 steps (predictable) |
| **GPU** | Tesla T4 (free tier, shared) | A10G (dedicated, consistent performance) |

## Performance & Cost

**Actual benchmarks** (tested on Modal A10G, $0.000306/sec):

| Entrypoint | GPU | Duration | Cost* | Notes |
|------------|-----|----------|-------|-------|
| `download_files` | CPU | ~1-2 min | ~$0.001 | One-time model download (~10GB) |
| `download_dataset` | CPU | ~1-2 min | ~$0.001 | Pre-cache dataset (saves GPU time) |
| `setup` | CPU | ~2 min | ~$0.001 | Verify env + download model |
| `infer` (single image) | A10G | **~90s** | **~$0.03** | ~60s generation, ~30s model load |
| `train_lora` (1 epoch, 200 samples) | A10G | **~7 min** | **~$0.13** | 25 steps, loss 0.41→0.01 |
| `train_lora` (1 epoch, 1000 samples) | A10G | **~15 min** | **~$0.27** | 125 steps, loss 0.36→0.01 |
| `train_lora` (1 epoch, full 7k dataset) | A10G | ~60-70 min | ~$1.10-1.28 | ~875 steps (estimated) |

*Cold start adds ~60-90s first time. Subsequent runs reuse cached image (~10-20s startup).

**Key metrics from 1000-sample run:**
- Training time: 882 seconds (~14.7 min actual)
- Final loss: 0.033 (excellent convergence: 0.359 → 0.011)
- Peak VRAM: 8.2 GB (well within A10G's 24GB)
- LoRA memory overhead: Only 0.48 GB!
- Throughput: 1.13 samples/sec, 0.14 steps/sec
- **Quality:** Significant improvement in LaTeX OCR accuracy (tested before/after)

## Managing Trained Models

**Path convention:**  
All trained LoRAs are saved to `/omnilaunch/runs/qwen3-vl/<run_name>/` on the Modal volume.

**Listing trained models:**
```bash
# SSH into Modal volume to see all runs
modal volume ls omnilaunch runs/qwen3-vl/
# Example output:
# brave-lion-1234/
# clever-fox-5678/
# swift-hawk-9012/
```

**Using a trained model:**
Just pass the run name (directory name) to `infer`:
```bash
omni run omnilaunch/qwen3-vl:0.1.0 infer \
  -p image="test.jpg" \
  -p prompt="Analyze this." \
  -p lora_run="brave-lion-1234"
```

## Understanding Training Output

**What gets saved:**
- **LoRA adapters**: Saved to `/omnilaunch/runs/qwen3-vl/<run_name>/`
  - `adapter_config.json` - LoRA configuration
  - `adapter_model.safetensors` - Trained weights (~50MB for r=16)
  - `tokenizer_config.json` - Tokenizer settings
- **Training metrics**: Returned as JSON (loss, runtime, memory usage)
- **Location**: Persists on Modal volume (accessible across runs)

**What does NOT get saved:**
- ❌ Intermediate checkpoints (only final weights)
- ❌ Training logs/curves (use WandB if needed)
- ❌ Base model (already cached separately)

**Using trained LoRA (programmatically):**
```python
from unsloth import FastVisionModel

# Load base model
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
    load_in_4bit=True
)

# Load your trained LoRA
model.load_adapter("/omnilaunch/runs/qwen3-vl/brave-lion-1234")
```

**Interpreting metrics:**
```json
{
  "train_loss": 0.119,           // Lower = better (0.41→0.01 in example)
  "train_runtime_minutes": 6.2,  // Total training time
  "peak_memory_gb": 8.2,          // Max VRAM used
  "lora_memory_gb": 0.5           // Memory used by LoRA training
}
```

Good training: Loss decreases smoothly, no sudden spikes. Example: `0.41 → 0.29 → 0.18 → 0.08 → 0.01`

## WandB Integration (Optional)

Training metrics can be automatically logged to [Weights & Biases](https://wandb.ai) for visualization and tracking across runs.

**Setup (one-time):**
1. Get your WandB API key from https://wandb.ai/authorize
2. Create a Modal secret:
   ```bash
   modal secret create wandb-secret WANDB_API_KEY=<your-key>
   ```
3. Rebuild and redeploy: `omni build omnilaunch/registry/qwen3-vl`

The runner automatically detects and uses the `wandb-secret` if it exists. No code changes needed!

**What gets logged:**
- Loss, learning rate, gradient norms (per step)
- Training runtime, memory usage, throughput
- All hyperparameters (LoRA rank, batch size, learning rate, epochs, etc.)
- System info (GPU type, VRAM usage)

**Viewing runs:**
- Runs appear at: `https://wandb.ai/<your-username>/omnilaunch-qwen3-vl`
- Compare training curves, filter by hyperparameters, track experiments over time
- Export data for custom analysis
- **Note:** WandB generates its own run name (e.g., "glorious-glitter-1") separate from the Omnilaunch run name (e.g., "carrot-worm-6829")

**Without WandB:** Training works perfectly fine without it. You'll see:
```
⚠ WANDB_API_KEY not found. Skipping WandB logging (set wandb-secret in Modal to enable).
```
All metrics are still returned in JSON response and printed to console.

## Tips

- **First run:** Use `omni setup` to cache the model (~10GB download)
- **Before training:** Run `download_dataset` on CPU to avoid expensive GPU time for dataset downloads
- **Fast inference:** After setup, inference takes ~90s (60s for generation)
- **Training:** Start with `epochs=1 max_train_samples=200` to test (~$0.13), then scale up
- **Watch the loss:** Should decrease smoothly. If loss spikes, reduce learning rate
- **Custom datasets:** Keep same format (image + text fields)
- **LoRA rank:** Higher `lora_r` = better accuracy but slower training (r=16 is good default)
- **Memory:** 4-bit quantization keeps training under 10GB (lots of A10G headroom)

## Technical Details

- **Model:** Unsloth's 4-bit quantized Qwen3-VL-8B-Instruct
- **Optimization:** Unsloth's FastVisionModel (2x faster than vanilla)
- **Quantization:** 4-bit (bitsandbytes) for memory efficiency
- **LoRA:** Fine-tunes both vision and language layers
- **Dataset format:** HuggingFace datasets with image + text fields
- **Training:** SFTTrainer with UnslothVisionDataCollator

