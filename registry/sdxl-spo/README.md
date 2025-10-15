# SDXL SPO

High-fidelity image generation using SDXL with SPO (Self-Play Preference Optimization) LoRA for improved quality.

## Model

- **Base Model**: [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- **VAE**: [madebyollin/sdxl-vae-fp16-fix](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) - Fixed VAE for better color/quality
- **LoRA**: [SPO-Diffusion-Models/SPO-SDXL_4k-p_10ep_LoRA](https://huggingface.co/SPO-Diffusion-Models/SPO-SDXL_4k-p_10ep_LoRA)
- **GPU**: A10G (24GB)
- **Special Feature**: SPO fine-tuning improves image quality, prompt adherence, and aesthetic appeal

## Quick Start

```bash
# Build the runner
omni build omnilaunch/registry/sdxl-spo

# Setup (download model weights + LoRA + custom VAE, ~14GB)
omni setup omnilaunch/sdxl-spo:0.1.0

# Generate high-quality image
omni run omnilaunch/sdxl-spo:0.1.0 infer \
  -p prompt="a majestic lion in the savanna, golden hour lighting, national geographic style" \
  -p steps=40 \
  --save --outfile lion.png
```

## Entrypoints

### `infer`

Generate images with SPO-enhanced quality and prompt adherence.

**Parameters:**
- `prompt` (string, required): Text description of the desired image
- `negative_prompt` (string, optional): What to avoid (default: `"ugly, blurry, low quality, distorted"`)
- `width` (int, optional): Image width in pixels (default: `1024`)
- `height` (int, optional): Image height in pixels (default: `1024`)
- `steps` (int, optional): Number of diffusion steps (default: `40`)
- `guidance_scale` (float, optional): How closely to follow the prompt (default: `7.5`)
- `seed` (int, optional): Random seed for reproducibility (default: random)

**Examples:**

```bash
# Photorealistic portrait
omni run omnilaunch/sdxl-spo:0.1.0 infer \
  -p prompt="close-up portrait of an elderly woman, wrinkled skin, kind eyes, soft natural lighting" \
  -p steps=50 -p guidance_scale=8.0 \
  --save --outfile portrait.png

# Architectural visualization
omni run omnilaunch/sdxl-spo:0.1.0 infer \
  -p prompt="modern minimalist house, glass walls, surrounded by forest, architectural photography" \
  -p width=1344 -p height=768 \
  --save --outfile architecture.png

# Product photography
omni run omnilaunch/sdxl-spo:0.1.0 infer \
  -p prompt="luxury watch on marble surface, studio lighting, product photography, high detail" \
  -p steps=45 -p seed=123 \
  --save --outfile watch.png
```

### `download_files`

Downloads base model, custom VAE, and SPO LoRA weights (runs automatically during `setup`).

### `setup`

Verifies environment and downloads all required models.

## Performance & Cost

| Operation | GPU | Duration | Cost* |
|-----------|-----|----------|-------|
| Setup (first time) | N/A (CPU) | ~3-4 min | ~$0.002 |
| Inference (25 steps) | A10G | ~45 sec | ~$0.014 |

*Inference is meant to be used for quick testing, so it releases the GPU after each response. This means it incurs a slight cost for cold start on each `run` call. I plan to add a `serve` function for optimized, fast inference, as well as supporting quantized versions that can use less expensive GPUs.

## SPO (Self-Play Preference Optimization)

SPO is a training technique that fine-tunes the model using preference learning:

- **Better quality**: Images have improved aesthetic appeal and coherence
- **Prompt adherence**: More accurate interpretation of complex prompts
- **Fewer artifacts**: Reduced distortions, anatomical errors, and visual glitches
- **Color accuracy**: Fixed VAE ensures proper color rendering without oversaturation

## Tips

- **Quality**: Use 40-50 steps for best results (SPO benefits from higher step counts)
- **Guidance**: 7.5-9.0 works well for most prompts
- **Prompting**: SPO responds well to detailed, descriptive prompts with style references
- **Use cases**: Excellent for photorealistic images, portraits, product photography, and architectural renders
- **Comparison**: Generally produces higher quality outputs than base SDXL at the same settings

## Version

- **Runner**: `0.1.0`
- **Last Updated**: 2025-10-15

