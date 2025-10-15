# SDXL Pixel Art

Fast pixel art generation using SDXL with LCM and Pixel Art LoRA adapters.

## Model

- **Base Model**: [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- **LoRA Adapters**:
  - [nerijs/pixel-art-xl](https://huggingface.co/nerijs/pixel-art-xl) - Pixel art style
  - [latent-consistency/lcm-lora-sdxl](https://huggingface.co/latent-consistency/lcm-lora-sdxl) - Fast generation (4-8 steps)
- **GPU**: A10G (24GB)
- **Special Feature**: Ultra-fast generation with LCM scheduler (4-8 steps vs 30-50)

## Quick Start

```bash
# Build the runner
omni build omnilaunch/registry/sdxl-pixel-art

# Setup (download model weights + LoRAs, ~14GB)
omni setup omnilaunch/sdxl-pixel-art:0.1.0

# Generate pixel art
omni run omnilaunch/sdxl-pixel-art:0.1.0 infer \
  -p prompt="a medieval castle, pixel art style" \
  -p steps=6 \
  --save --outfile castle.png
```

## Entrypoints

### `infer`

Generate pixel art images with fast LCM sampling.

**Parameters:**
- `prompt` (string, required): Text description of the desired image
- `negative_prompt` (string, optional): What to avoid (default: `"blurry, smooth, 3d render"`)
- `width` (int, optional): Image width in pixels (default: `1024`)
- `height` (int, optional): Image height in pixels (default: `1024`)
- `steps` (int, optional): Number of diffusion steps, 4-8 recommended (default: `6`)
- `guidance_scale` (float, optional): How closely to follow the prompt (default: `1.5`)
- `seed` (int, optional): Random seed for reproducibility (default: random)
- `lora_strength` (float, optional): Pixel art LoRA intensity, 0-1 (default: `1.0`)

**Examples:**

```bash
# Character sprite
omni run omnilaunch/sdxl-pixel-art:0.1.0 infer \
  -p prompt="8-bit video game character, pixel art, knight with sword" \
  -p steps=8 -p lora_strength=1.0 \
  --save --outfile knight.png

# Landscape
omni run omnilaunch/sdxl-pixel-art:0.1.0 infer \
  -p prompt="isometric pixel art city, cyberpunk, night, neon lights" \
  -p width=1024 -p height=768 \
  --save --outfile city.png

# Item icon
omni run omnilaunch/sdxl-pixel-art:0.1.0 infer \
  -p prompt="pixel art potion bottle, glowing purple, item icon" \
  -p width=512 -p height=512 -p steps=4 \
  --save --outfile potion.png
```

### `download_files`

Downloads base model and LoRA weights (runs automatically during `setup`).

### `setup`

Verifies environment and downloads all required models.

## Performance & Cost

| Operation | GPU | Duration | Cost* |
|-----------|-----|----------|-------|
| Setup (first time) | N/A (CPU) | ~2-3 min | ~$0.002 |
| Inference (25 steps) | A10G | ~35 sec | ~$0.010 |

*Inference is meant to be used for quick testing, so it releases the GPU after each response. This means it incurs a slight cost for cold start on each `run` call. I plan to add a `serve` function for optimized, fast inference, as well as supporting quantized versions that can use less expensive GPUs.

## Tips

- **Speed**: Use 4-6 steps for ultra-fast generation, 8 steps for slightly better quality
- **Guidance**: Keep `guidance_scale` between 1.0-2.0 for LCM (higher values may degrade quality)
- **LoRA strength**: Reduce to 0.6-0.8 for subtle pixel art effect, use 1.0 for full effect
- **Prompting**: Include "pixel art", "8-bit", "16-bit", or "isometric" in prompts for best results
- **Negative prompts**: Add "smooth, 3d render, realistic" to avoid non-pixel-art aesthetics

## Version

- **Runner**: `0.1.0`
- **Last Updated**: 2025-10-15

