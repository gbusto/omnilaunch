# SDXL Base 1.0

High-quality text-to-image generation using Stable Diffusion XL Base 1.0.

## Model

- **Base Model**: [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- **GPU**: A10G (24GB)
- **Architecture**: Latent diffusion model with dual text encoders (CLIP ViT-L/14 + OpenCLIP ViT-bigG/14)
- **Resolution**: Native 1024×1024, supports 512×2048 and other aspect ratios

## Quick Start

```bash
# Build the runner
omni build omnilaunch/registry/sdxl

# Setup (download model weights, ~13GB)
omni setup omnilaunch/sdxl:0.1.0

# Generate an image
omni run omnilaunch/sdxl:0.1.0 infer \
  -p prompt="a serene mountain landscape at sunset, oil painting" \
  -p negative_prompt="blurry, low quality" \
  -p steps=30 \
  --save --outfile mountain.png
```

## Entrypoints

### `infer`

Generate images from text prompts.

**Parameters:**
- `prompt` (string, required): Text description of the desired image
- `negative_prompt` (string, optional): What to avoid in the image (default: `"ugly, blurry, low quality"`)
- `width` (int, optional): Image width in pixels (default: `1024`)
- `height` (int, optional): Image height in pixels (default: `1024`)
- `steps` (int, optional): Number of diffusion steps, higher = better quality (default: `30`)
- `guidance_scale` (float, optional): How closely to follow the prompt, 7-12 recommended (default: `7.5`)
- `seed` (int, optional): Random seed for reproducibility (default: random)

**Examples:**

```bash
# Portrait
omni run omnilaunch/sdxl:0.1.0 infer \
  -p prompt="professional headshot of a woman, studio lighting" \
  -p width=768 -p height=1024 \
  --save --outfile portrait.png

# Landscape with custom seed
omni run omnilaunch/sdxl:0.1.0 infer \
  -p prompt="cyberpunk city at night, neon lights, rain" \
  -p seed=42 -p steps=50 \
  --save --outfile cyberpunk.png
```

### `download_files`

Downloads model weights to the Modal volume (runs automatically during `setup`).

### `setup`

Verifies environment and downloads model if needed.

## Performance & Cost

| Operation | GPU | Duration | Cost* |
|-----------|-----|----------|-------|
| Setup (first time) | N/A (CPU) | ~2-3 min | ~$0.002 |
| Inference (25 steps) | A10G | ~30 sec | ~$0.009 |

*Inference is meant to be used for quick testing, so it releases the GPU after each response. This means it incurs a slight cost for cold start on each `run` call. I plan to add a `serve` function for optimized, fast inference, as well as supporting quantized versions that can use less expensive GPUs.

## Tips

- **Quality**: Use 30-50 steps for best results
- **Speed**: 20-25 steps for faster generation with acceptable quality
- **Prompt engineering**: Be specific and descriptive, use artist/style references
- **Negative prompts**: Add unwanted elements like "blurry, ugly, deformed" to improve output
- **Aspect ratios**: SDXL works best at 1024×1024 or similar high-res dimensions

## Version

- **Runner**: `0.1.0`
- **Last Updated**: 2025-10-15

