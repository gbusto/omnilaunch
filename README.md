# Omnilaunch

> **The execution layer for reproducible AI**  
> Run any model, anywhere, reproducibly — training, inference, fine-tuning, benchmarking.

---

## 🎯 The Vision

Run any model, anywhere, reproducibly. One CLI (`omni run <runner> <entrypoint>`) for training, inference, fine-tuning, and benchmarking.

**Omnilaunch standardizes AI execution** — built on Modal + Hugging Face for transparent, reproducible, cost-effective model workflows. No more broken Colabs, missing dependencies, or incompatible environments.

---

## ⚡ Quick Start (90 seconds)

Follow the [Modal account setup](./docs/MODAL_SETUP.md) guide first, then you can run this quickstart. Modal account setup will literally take you less than 2 minutes. You don't even need to enter your credit card if you don't want to! But I would recommend it to unlock the free $30/mo of free usage (they're basically *paying* you to use their platform!).

```bash
# 1. Install
pip install -e omnilaunch/

# 2. Check environment
omni doctor

# 3. Build a runner
omni build omnilaunch/registry/sdxl/

# 4. Setup (deploys app + downloads model)
omni setup omnilaunch/sdxl:0.1.0

# 5. Run inference
omni run omnilaunch/sdxl:0.1.0 infer \
  -p prompt="astronaut riding a horse on mars" \
  -p steps=25 \
  --save --outfile astronaut.png

# ✨ Image saved to ./omni_out/astronaut.png
```

**That's it.** No Docker, no CUDA setup, no dependency hell. It just works — reproducibly, every time.

---

## 💡 Why Modal?

**Transparent & Predictable:**
- Per-second GPU billing (not per-step like Fal/Fireworks)
- [Pick your GPU](https://modal.com/pricing) (A10G, H100, etc.) — no black-box allocation
- Modal shows exact execution time and cost

**Just Python:**
- No Kubernetes, no Docker builds, no YAML hell
- Simple decorators + functions = deployed model
- Persistent volumes for model caching (no re-downloads)

**Reliable:**
- Competitors (Fal, Fireworks) have frequent compatibility issues
- Modal's abstraction is minimal — closer to "your code on a GPU"
- Clear error messages, real logs (not cryptic failures)

**Supportive Company:**
- Modal provides credit grants for [academics](https://modal.com/academics) and [startups](https://modal.com/startups)
- If you have committed spend with AWS, Azure, GCP, or OCI, you will soon be able to use that commit on Modal
- Get free credits ($30/mo of free credits on the Free plan, $100/mo of free credits on the $250/mo plan) to put towards your compute

---

## 📊 Omnilaunch vs Alternatives

- [Replicate](https://replicate.com) - conveniently hosts models for you; no-code UI + API; finding models to fine-tune is not intuitive or clear, and you pay per generation for inference and per-step for fine-tuning. You basically pay extra for convenience. Fine-tuning is also designed for small datasets and very limited. 
- [Fal](https://fal.ai) - same benefits and drawbacks as Replicate; although I find it more intuitive to find which models can be fine-tuned.
- [Fireworks](https://fireworks.ai) - mixed pricing; some setups/models are per-token, some require you to pay for GPU-time. They require a minimum of 5 minutes of GPU time once spun up, meaning you're wasting money unless you have a high-traffic app where this makes sense. Training is done-for-you, but blackbox; they charge per-step, and you can end up with cryptic errors and failures.
- [Lambda AI](https://lambda.ai) - no option to spin up a VM/container/instance detached from a GPU, so you pay for every second your instance is running, even when idle and not using the GPU. Persistent storage is not always guaranteed; most of the time what you're doing is ephemeral.
- [Notebooks](https://colab.research.google.com/) - these are great and widely used; they solve reproducibility. But seem like a pain to get setup, there are lots of new platforms that have their own "Notebooks" (Modal has this now too), and from what I've heard many people use this to get free GPU/TPU time. But it sounds like in many cases, you can be left waiting for a LONG time to secure free GPU/TPU compute. And it just requires more setup and maintenance.
- Local - building and running locally is a minefield of dependency issues, wide variance of consumer hardware, and operating systems + versions. What works on one person's machine is not likely to work on another's. You also usually need to run quantized, watered-down versions of models. It's "free", but you pay with your time and sanity.

With Omnilaunch, you get the convenience of Replicate and Fal, and infinite training customization (as much as the desired model supports). Thoug for one-off inference, you likely get better pricing from platforms like Replicate or Fal. But this will improve over time.

Compared to Fireworks AI, you'll have full insight into training runs via logs, and the same convenience they provide of ~basically "done-for-you" training. Except you don't overpay by paying per step, you pay for GPU time, which in my experience is usually much cheaper - even without lots of special optimizations.

Compared to Lambda, with Omnilauch you get the ease of running stuff as if you're just writing Python on a Linux instance with an attached GPU, but you only pay per second of GPU time. AND you get persistent storage. You get none of these with Lambda, and in my experience, they don't have the same volume of available GPUs are Modal does; I often had to wait for the desired GPU to become available again.

Compared to Notebooks-like tools and products, this is highly reproducible. You essentially get "free" compute through Modal's pricing plans: for the Free plan, you get $30/mo of free compute, and for their $250/mo plan you get $100/mo of free compute. AND you can pick which GPUs you use!

---

## 🧩 What You Can Do Today

### 💬 **LLM Inference with Reasoning**
Run OpenAI's GPT-OSS models with parsed reasoning channels:

```bash
# 20B model on A10G
omni run omnilaunch/gpt-oss-20b:0.1.0 infer \
  -p messages='[{"role": "user", "content": "Explain quantum entanglement"}]' \
  -p reasoning_level=high \
  --save --outfile response.json

# 120B model on H100
omni run omnilaunch/gpt-oss-120b:0.1.0 infer \
  -p messages='[{"role": "user", "content": "Prove the Pythagorean theorem"}]' \
  -p reasoning_level=high \
  -p max_tokens=1024 \
  --save
```

Returns structured JSON with `response`, `analysis` (internal reasoning), and `commentary` (meta-observations).

### 🎨 **Image Generation**
Multiple SDXL variants ready to use:

```bash
# Base SDXL
omni run omnilaunch/sdxl:0.1.0 infer \
  -p prompt="serene mountain landscape" --save

# Pixel art style (with LCM + custom LoRA)
omni run omnilaunch/sdxl-pixel-art:0.1.0 infer \
  -p prompt="cute robot" --save

# Aesthetic-optimized (SPO)
omni run omnilaunch/sdxl-spo:0.1.0 infer \
  -p prompt="cinematic sunset" --save
```

### 🎵 **Audio Generation** *(coming soon)*
```bash
# Text-to-speech
omni run omnilaunch/orpheus:0.1.0 infer \
  -p text="Hello, world!" \
  --save --outfile speech.wav

# Music generation
omni run omnilaunch/musicgen:0.1.0 infer \
  -p prompt="upbeat jazz piano" \
  -p duration=30 \
  --save
```

### 🎨 **3D Generation** *(coming soon)*
```bash
# Text to 3D mesh
omni run omnilaunch/hunyuan3d-2.1:0.1.0 infer \
  -p prompt="a wooden chair" \
  --save --outfile chair.obj

# Image to 3D
omni run omnilaunch/trellis:0.1.0 infer \
  -p image="photo.png" \
  --save
```

### 🔧 **Fine-tuning** *(coming soon)*
```bash
# SDXL LoRA for personalized images
omni run omnilaunch/hunyuanimage-3.0:0.2.0 train_lora \
  --dataset hf:yourname/photos \
  -p subject="sks person" \
  -p steps=1000

# Use your trained LoRA
omni run omnilaunch/hunyuanimage-3.0:0.2.0 infer \
  -p prompt="sks person as an astronaut" \
  -p lora_path="/omnilaunch/runs/hunyuanimage-3.0-lora-xyz/model.safetensors" \
  --save
```

### 🔬 **Benchmarking** *(coming soon)*
```bash
# Run MMLU benchmark
omni run omnilaunch/gpt-oss-20b-bench:0.1.0 eval_mmlu \
  --save --outfile results.json

# Results include reproducible metrics
```

### 📋 **List All Runners**
```bash
omni list

# Output:
#   omnilaunch/gpt-oss-20b
#     Entrypoints: download_files (CPU), setup (CPU), infer (A10G)
#   omnilaunch/gpt-oss-120b
#     Entrypoints: download_files (CPU), setup (CPU), infer (H100)
#   omnilaunch/sdxl
#     Entrypoints: download_files (CPU), setup (CPU), infer (A10G)
#   ...
```

---

## 🏗️ How It Works

### **What's a Runner?**

A **runner** is a self-contained, reproducible package for model execution. Think of it as a specialized container that knows how to run a specific model or workflow.

**Runners can represent:**
- Official model deployments (e.g., Stability AI's SDXL)
- Optimized inference stacks (e.g., vLLM for LLMs, optimized diffusion pipelines)
- SaaS-ready backends (e.g., PhotoAI-style personalized image generation)
- Benchmarking setups (e.g., `omnilaunch/gpt-oss-20b-bench` with supported benchmarks as entrypoints)
- Educational tools (e.g., compare pre-alignment vs post-alignment LLM checkpoints)

**Everything is just:** `omni run <runner> <entrypoint>`

```
omnilaunch/registry/sdxl/
├── runner.yaml         # Metadata: name, version, GPU per entrypoint
├── modal_app.py        # Modal app with functions (setup, infer, train_*)
├── schema/
│   ├── infer.json      # JSON schema for inference params
│   └── dataset.json    # JSON schema for training data
└── tests/smoke.py      # Basic smoke tests
```

**Key features:**
- **Versioned** — `omnilaunch/sdxl:0.1.0`, `omnilaunch/sdxl:0.2.0`
- **Bundled** — Packaged as `.tar.gz` with manifest and schemas
- **Signed** *(coming)* — Cryptographically verified with Sigstore/GPG
- **Backend-agnostic** *(future)* — Modal first, then AWS/RunPod/local via IaC tools

### **Entrypoints**

Each runner defines standardized entrypoints. CPU and GPU resources are specified per-entrypoint:

```yaml
entrypoints:
  download_files:
    function: modal_app.py::download_files
    gpu: null  # CPU-only, ultra-cheap
  setup:
    function: modal_app.py::setup
    gpu: null  # CPU verification
  infer:
    function: modal_app.py::infer
    gpu: "A10G"  # GPU inference
    schema: schema/infer.json
  train_lora:
    function: modal_app.py::train_lora
    gpu: "A100"
    schema: schema/train_lora.json
```

**CPU inference** provides "Ollama for servers" — cheap, reproducible inference for smaller models or batch jobs.

### **Reproducibility**

Every run produces a signed manifest *(coming soon)*:

```json
{
  "runner": "omnilaunch/sdxl:0.1.0",
  "entrypoint": "infer",
  "dataset": "hf:user/data@a7cd4f",
  "params_hash": "f26b4...",
  "image_hash": "sha256:c7b2a...",
  "backend": "modal",
  "gpu": "A10G",
  "signatures": ["cosign:6f3e..."],
  "metrics": {"steps": 25, "guidance": 7.5}
}
```

---

## 🚀 Roadmap

**✅ Available Now (v0.1):**
- LLM inference (GPT-OSS-20B/120B with reasoning)
- Image generation (SDXL + pixel-art + SPO variants)
- Modal deployment & execution
- CLI: `build`, `setup`, `run`, `list`, `doctor`
- Reproducible manifests with pinned dependencies

**🔄 In Progress (v0.2):**
- Audio generation (TTS, music)
- 3D generation (text/image to mesh)
- Fine-tuning (SDXL LoRA, LLM LoRA)
- Benchmarking runners (MMLU, GSM8K)
- Production serving (`omni serve` with vLLM)

**📋 Planned (v0.3+):**
- More models (Llama, Mistral, Flux, etc.)
- Registry with signing (Sigstore/GPG)
- Benchmark leaderboards (reproducible, verifiable)
- Educational runners (checkpoint comparison)

**🌍 Future (community-driven):**
- Multi-backend support (AWS, RunPod, local via IaC)
- Playground UI (open-source, connects to your Modal)
- SaaS starter templates

---

## 🎯 Core Use Cases

### 🧠 **Research & Replication**
Reproduce any paper with one command:
```bash
# Paper authors package their experiment as a runner
omni run omnilaunch/llama3-alignment-paper:0.1.0 train_full \
  --dataset hf:openmath/gsm8k@rev

# Benchmark results included
omni run omnilaunch/llama3-alignment-paper:0.1.0 eval_mmlu
omni run omnilaunch/llama3-alignment-paper:0.1.0 eval_gsm8k
```

Share *executable* results, not just papers. Encapsulate the entire experimental setup in a runner — training, evaluation, benchmarks. Full provenance from data → model → results.

### 🎓 **Education & Model Exploration**
Understand model evolution through training and alignment:

```bash
# Compare checkpoints at different training stages
omni run omnilaunch/llama3-checkpoints:0.1.0 infer \
  -p checkpoint="100k_steps" \
  -p prompt="Explain photosynthesis" \
  --save --outfile step_100k.json

omni run omnilaunch/llama3-checkpoints:0.1.0 infer \
  -p checkpoint="500k_steps" \
  -p prompt="Explain photosynthesis" \
  --save --outfile step_500k.json

omni run omnilaunch/llama3-checkpoints:0.1.0 infer \
  -p checkpoint="final" \
  -p prompt="Explain photosynthesis" \
  --save --outfile final.json

# See how responses improve with training

# Compare pre- and post-alignment behavior
omni run omnilaunch/llama3-base:0.1.0 infer \
  -p prompt="How do I build a bomb?" \
  --save --outfile pre_safety.json

omni run omnilaunch/llama3-safety-aligned:0.1.0 infer \
  -p prompt="How do I build a bomb?" \
  --save --outfile post_safety.json

# See how safety training affects responses
```

Interactive learning about training phases, RLHF, alignment, and safety interventions.

### 🚀 **SaaS Prototyping**
Omnilaunch standardizes **execution**, not hosting. Use runners as backends for your apps:

```bash
# Your backend uses the runner
omni run omnilaunch/sdxl-dreambooth:0.1.0 train_lora \
  --dataset hf:user/photos \
  --subject "sks person"

# Wrap with your UI, payment, and database
# → PhotoAI-style personalized generation
```

Go from idea to product in hours, not months.

### 🔬 **Benchmarking**
Model-specific, reproducible evaluation:

```bash
# Benchmark runner with targeted entrypoints
omni run omnilaunch/gpt-oss-20b-bench:0.1.0 mmlu
omni run omnilaunch/gpt-oss-20b-bench:0.1.0 gsm8k
omni run omnilaunch/diffusion-bench:0.1.0 fid --dataset coco

# → Signed results feed reproducible leaderboards
```

Each benchmark runner targets a model family. Simple, clear, verifiable.

### 🔏 **Compliance & Governance**
Full audit trail for every execution — dataset versions, code revisions, environment hashes, GPU specs. Verifiable lineage for regulatory compliance and transparency.

---

## 💡 Why This Matters

### **For Researchers**
- Reproduce any experiment with one command
- Share *executable* results with verifiable provenance
- Build on others' work without setup hell
- Transparent model lineage from data to results

### **For Developers**
- No DevOps, no GPU setup, no container debugging
- Swap models and backends like changing a config file
- Compose workflows across modalities
- CPU + GPU options for cost optimization

### **For Startups**
- Launch ML-powered products in days, not months
- Pre-built runners for common use cases
- Scale inference without infrastructure teams
- Backend flexibility (Modal → AWS when ready)

### **For the Ecosystem**
- Open registry like Docker Hub for AI
- Community-driven runner marketplace
- Transparent, auditable model provenance
- Foundation for reproducible AI governance

---

## 🧭 Core Principles

Omnilaunch is built on four guiding principles:

1. **Reproducibility > Convenience** — Verifiable execution over quick hacks
2. **Standardization > Flexibility** — One interface (`omni run`) for everything
3. **Composability > Scope Creep** — Runners do one thing well
4. **Interoperability** — Built on Modal + Hugging Face for open infrastructure

These principles ensure stability, transparency, and long-term ecosystem health.

---

## 📚 CLI Reference

### Commands

| Command | Description |
|---------|-------------|
| `omni doctor` | Check local environment and credentials |
| `omni list` | List available runners from the registry |
| `omni build <path>` | Package a runner into a versioned bundle |
| `omni setup <runner:version>` | Deploy app, build image, download models |
| `omni run <runner:version> <entrypoint>` | Execute any entrypoint (train, infer, benchmark, etc.) |

### Flags for `omni run`

- `--params <file or JSON>` - Parameters as YAML/JSON file or inline string
- `-p, --param key=value` - Individual param overrides (repeatable)
- `--save` - Auto-save outputs to disk
- `--outdir <path>` - Directory for saved outputs (default: `./omni_out`)
- `--outfile <name>` - Explicit output filename
- `--dataset <uri>` - Dataset URI for training (e.g., `hf:org/dataset` or `vol:/path`)

---

## 🌍 The Goal

Omnilaunch aims to be the standard way to package and run AI models — reproducible, transparent, and accessible to everyone.

Built on **Modal + Hugging Face** for open, reproducible infrastructure.

---

## 🙏 Credits

Built on:
- [Modal](https://modal.com) — Serverless GPU infrastructure
- [Hugging Face](https://huggingface.co) — Models & datasets
- The open-source AI community

---

**This is early-stage. Feedback welcome.** 🚀
