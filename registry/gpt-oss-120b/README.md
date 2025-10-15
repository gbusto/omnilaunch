# GPT-OSS-120B

OpenAI's open-source 120B parameter LLM with MXFP4 quantization, reasoning channels, and harmony format output.

## Quick Start

```bash
# 1. Build runner
omni build omnilaunch/registry/gpt-oss-120b

# 2. Setup (download model, verify env)
omni setup omnilaunch/gpt-oss-120b:0.1.0

# 3. Run inference
omni run omnilaunch/gpt-oss-120b:0.1.0 infer \
  -p messages='[{"role": "user", "content": "Explain quantum computing"}]'
```

## Entrypoints

### `download_files`
Downloads the `openai/gpt-oss-120b` model from HuggingFace to `/omnilaunch/models/openai/gpt-oss-120b`.

**GPU:** CPU only  
**Timeout:** 3600s

### `setup`
Verifies environment (torch, CUDA, transformers) and calls `download_files` to cache the model.

**GPU:** CPU only  
**Timeout:** 1800s

### `infer`
Chat completion with reasoning levels and harmony format parsing.

**GPU:** H100 (80GB) with MXFP4 quantization  
**Timeout:** 600s

**Parameters:**
- `messages` (required): Array of message objects with `role` and `content`
- `max_tokens` (default: 256): Max new tokens to generate
- `temperature` (default: 0.7): Sampling temperature
- `reasoning_level` (default: "medium"): "low", "medium", or "high"

**Returns:**
```json
{
  "response": "User-facing answer (from 'final' channel)",
  "analysis": "Internal reasoning/thinking (from 'analysis' channel)",
  "commentary": "Meta-observations (from 'commentary' channel, optional)",
  "reasoning_level": "medium"
}
```

## Harmony Format

GPT-OSS models use an internal "harmony format" for structured output:

```
<|channel|>analysis<|message|>Internal reasoning here...<|end|>
<|channel|>commentary<|message|>Meta-observations...<|end|>
<|channel|>final<|message|>User-facing response<|return|>
```

The runner automatically parses this format and returns clean JSON with separate fields for each channel.

## Examples

### Basic chat
```bash
omni run omnilaunch/gpt-oss-120b:0.1.0 infer \
  -p messages='[{"role": "user", "content": "What is recursion?"}]' \
  --save --outfile answer.json
```

### High reasoning
```bash
omni run omnilaunch/gpt-oss-120b:0.1.0 infer \
  -p messages='[{"role": "user", "content": "Prove the Pythagorean theorem"}]' \
  -p reasoning_level=high \
  -p max_tokens=1024
```

### Multi-turn conversation
```bash
omni run omnilaunch/gpt-oss-120b:0.1.0 infer \
  -p messages='[
    {"role": "user", "content": "I have 10 apples"},
    {"role": "assistant", "content": "Got it, you have 10 apples."},
    {"role": "user", "content": "I eat 3. How many left?"}
  ]'
```

## Performance & Cost

| Entrypoint | GPU | Duration | Cost* |
|------------|-----|----------|-------|
| Setup (first time) | N/A (CPU) | ~5-8 min | ~$0.005 |
| `infer` (256 tokens; medium reasoning) | H100 | ~1 min | ~$0.066 |
| `infer` (~2.5k tokens; medium reasoning) | H100 | ~5 min | ~$0.330 |

*Inference is meant to be used for quick testing, so it releases the GPU after each response. This means it incurs a slight cost for cold start on each `run` call. I plan to add a `serve` function for optimized, fast inference.

Note that for this larger model, cold start time was ~1min and the inference was ~4min. So this ended up being ~10tps which isn't very good. It will be improved in the future!

## Tips

- Use `reasoning_level=high` for complex math, logic, or coding tasks
- The `analysis` field contains the model's step-by-step thinking
- Set `max_tokens` higher (512-2048) for detailed reasoning tasks
- The harmony format is automatically parsed - you get clean JSON output

