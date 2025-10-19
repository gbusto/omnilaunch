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
### `benchmark_tinymmlu`

Evaluate the model on the tinyBenchmarks/tinyMMLU dataset (100 examples). Fast, lightweight benchmark for sanity-checking reasoning performance.

**Parameters:**
- `split` (default: `"test"`): `"test"` or `"dev"`
- `max_items` (default: `100`): Limit number of items
- `batch_size` (default: `8`)
- `max_new_tokens` (default: `1`): Single-token answer decoding (A/B/C/D)
- `temperature` (default: `0.0`): Greedy decoding
- `use_formatted` (default: `true`): Use dataset `input_formatted` prompt

**Usage:**
```bash
# Standard eval (matches paper setup: harmony format + reasoning:high + log-likelihood)
omni run omnilaunch/gpt-oss-20b:0.1.0 benchmark_tinymmlu \
  --save --outfile tinymmlu.json

# Custom reasoning level
omni run omnilaunch/gpt-oss-20b:0.1.0 benchmark_tinymmlu \
  -p reasoning_level=medium \
  --save --outfile tinymmlu_medium.json

# Quick test (first 20 items)
omni run omnilaunch/gpt-oss-20b:0.1.0 benchmark_tinymmlu \
  -p max_items=20 \
  --save --outfile quick_test.json
```

**Method:** Reuses `infer` endpoint logic for consistency:
- Same harmony chat format and parsing as `infer`
- System prompt with `Reasoning: low` (configurable)
- Greedy decoding (temp=0, max 32 tokens)
- Extracts answer letter from parsed response field

**Output:**
```json
{
  "ok": true,
  "items": 100,
  "overall_accuracy": 0.68,
  "correct": 68,
  "per_subject_accuracy": {"math": 0.65, "biology": 0.72, ...},
  "samples": [
    {
      "prompt_end": "...",
      "generated_raw": "<|channel|>analysis<|message|>...<|channel|>final<|message|>B<|return|>",
      "response": "B",
      "analysis": "...",
      "pred": "B",
      "gt": "B",
      "subject": "math"
    }
  ],
  "elapsed_seconds": 180.5,
  "config": {
    "split": "test",
    "max_items": 100,
    "reasoning_level": "low",
    "method": "reuses_infer_logic"
  }
}
```

**Note:** TinyMMLU is a 100-item subset of MMLU with different distribution, so expect lower/noisier scores vs. the paper's full MMLU results (~85% for gpt-oss-20b @ high reasoning on full MMLU).

Dataset: `tinyBenchmarks/tinyMMLU` ([link](https://huggingface.co/datasets/tinyBenchmarks/tinyMMLU))


Verifies environment and downloads model if needed.

## Performance & Cost

| Entrypoint | GPU | Duration | Cost* |
|------------|-----|----------|-------|
| Setup (first time) | N/A (CPU) | ~5-8 min | ~$0.005 |
| `infer` (256 tokens; medium reasoning) | A10G | ~1 min | ~$0.018 |
| `infer` (~1k tokens; medium reasoning) | A10G | ~1-2 min | ~$0.028 |
| `benchmark_tinymmlu` (10 items) | A10G | ~6.5 min | ~$0.12 |
| `benchmark_tinymmlu` (100 items) | A10G | ~65 min | ~$1.19 |

*Inference is meant to be used for quick testing, so it releases the GPU after each response. This means it incurs a slight cost for cold start on each `run` call. I plan to add a `serve` function for optimized, fast inference.

**Benchmark Results (10-sample tinyMMLU, reasoning_level=low):**
- **Accuracy**: 60% (6/10 correct)
- **Total time**: 392.1s (6.5 min)
- **Per-question**: ~39 sec/question (after model load)
- **Cost estimate**: ~$0.11 for 10 items, ~$1.08 for full 100 items

**Estimated 100-sample run:**
- Model loading: ~9s (one-time)
- 100 questions × 39s = 3,900s (65 min)
- Total: ~65 min (~$1.08 on A10G @ $0.99/hr)

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

