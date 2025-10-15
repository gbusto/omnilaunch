# GPT-OSS-20B

OpenAI's GPT-OSS-20B with harmony format reasoning - see the model's internal thought process.

## Model

- **Base Model**: [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)
- **GPU**: A10G (24GB) with MXFP4 quantization
- **Architecture**: Transformer-based language model (20B parameters)
- **Special Feature**: Harmony format with separate reasoning, commentary, and final response channels

## Quick Start

```bash
# Build the runner
omni build omnilaunch/registry/gpt-oss-20b

# Setup (download model weights, ~40GB)
omni setup omnilaunch/gpt-oss-20b:0.1.0

# Chat completion with reasoning
omni run omnilaunch/gpt-oss-20b:0.1.0 infer \
  --params '{"messages": [{"role": "user", "content": "Explain quantum entanglement in simple terms"}], "reasoning_level": "high"}' \
  --save --outfile response.json
```

## Entrypoints

### `infer`

Chat completion with internal reasoning exposed via harmony format (MXFP4 quantized on A10G).

**Parameters:**
- `messages` (array, required): Chat messages in OpenAI format `[{"role": "user", "content": "..."}]`
- `max_tokens` (int, optional): Maximum tokens to generate (default: `256`)
- `temperature` (float, optional): Sampling temperature, 0-2 (default: `0.7`)
- `reasoning_level` (string, optional): Amount of reasoning - `"low"`, `"medium"`, or `"high"` (default: `"medium"`)

**Output Format:**

The response includes the model's internal reasoning process:

```json
{
  "response": "The final answer to the user",
  "analysis": "Internal reasoning and planning",
  "commentary": "Meta-observations (optional)",
  "reasoning_level": "medium"
}
```

**Examples:**

```bash
# Simple query with medium reasoning
omni run omnilaunch/gpt-oss-20b:0.1.0 infer \
  --params '{"messages": [{"role": "user", "content": "What is the capital of France?"}]}' \
  --save --outfile capital.json

# Complex reasoning task
omni run omnilaunch/gpt-oss-20b:0.1.0 infer \
  --params '{"messages": [{"role": "user", "content": "Design a sorting algorithm for nearly-sorted arrays"}], "reasoning_level": "high", "max_tokens": 512}' \
  --save --outfile algorithm.json

# Multi-turn conversation
omni run omnilaunch/gpt-oss-20b:0.1.0 infer \
  --params '{"messages": [{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi! How can I help?"}, {"role": "user", "content": "Tell me a joke"}]}' \
  --save --outfile joke.json
```

### `download_files`

Downloads model weights to the Modal volume (runs automatically during `setup`).

### `setup`

Verifies environment and downloads model if needed.

## Performance & Cost

| Entrypoint | GPU | Duration | Cost* |
|------------|-----|----------|-------|
| Setup (first time) | N/A (CPU) | ~5-8 min | ~$0.005 |
| `infer` (256 tokens; medium reasoning) | A10G | ~1 min | ~$0.018 |
| `infer` (~1k tokens; medium reasoning) | A10G | ~1-2 min | ~$0.028 |

*Inference is meant to be used for quick testing, so it releases the GPU after each response. This means it incurs a slight cost for cold start on each `run` call. I plan to add a `serve` function for optimized, fast inference.

## Harmony Format

GPT-OSS uses a unique "harmony" format with three channels:

- **`analysis`**: Internal reasoning and planning - what the model is thinking
- **`commentary`**: Meta-level observations or notes (optional)
- **`final`**: The actual response shown to the user

This allows you to see the model's "chain of thought" before it produces the final answer.

## Tips

- **MXFP4 Quantization**: The model uses MXFP4 quantization (requires `triton>=3.4.0` and `kernels` package) to run efficiently on A10G GPUs
- **Reasoning levels**: Use `"high"` for complex problems, `"low"` for simple queries to save tokens
- **Temperature**: Use `0.0` for deterministic outputs, `0.7-1.0` for creative responses
- **Analyzing reasoning**: The `analysis` field shows how the model approached the problem
- **Token budget**: Set `max_tokens` appropriately - reasoning can consume significant tokens
- **Cost optimization**: A10G provides excellent price/performance for this quantized model

## Version

- **Runner**: `0.1.0`
- **Last Updated**: 2025-10-15

