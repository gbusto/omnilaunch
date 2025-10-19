# Omnilaunch

> **AI workflows that just work.**  
> Run, train, benchmark, and reproduce *any* AI workflow on GPUs — cleanly, consistently, and reproducibly.

Omnilaunch is a **Python CLI** for building, deploying, and running GPU-powered AI workflows on [Modal](https://modal.com).
It wraps the Modal API and CLI with a clean, consistent layer that standardizes how AI experiments, model training runs, and benchmarks are executed.

At its core, Omnilaunch turns complex infrastructure into a simple pattern:

```bash
omni build  → package a runner (creates versioned tarball + hash)
omni setup  → deploy to Modal and perform setup tasks (build image, download weights, run checks)
omni run    → execute an entrypoint on Modal (train, infer, benchmark, etc.)
```

Everything runs **on Modal** (where GPUs live), but can be **triggered from anywhere** using the CLI.
The goal is simple: **no broken Colabs, no dependency hell, no "it worked on my machine" excuses — just reproducible AI workflows that work.**

---

## ⚡ Quick Start

Follow the [Modal account setup](./docs/MODAL_SETUP.md) guide first, then run this quickstart.
Setup takes ~2 minutes, no credit card required (Modal includes $30/mo of free GPU time).

```bash
# 0. Setup
git clone https://github.com/gbusto/omnilaunch
python3.12 -m venv .venv
source .venv/bin/activate

# 1. Install
pip install -e omnilaunch/

# 2. Setup modal and authenticate via command line (if not already done from the Modal account setup walkthrough linked above)
modal setup

# 3. Check environment
omni doctor

# 4. Build a runner
omni build omnilaunch/registry/sdxl/

# 5. Setup; deploys app + builds image + downloads model
#    This will take ~10 mins, so go grab some coffee ☕️
#    No need to call `setup` again in the future; this is a one-time operation
omni setup omnilaunch/sdxl:0.1.0

# 6. Run inference
omni run omnilaunch/sdxl:0.1.0 infer \
  -p prompt="astronaut riding a horse on mars" \
  -p steps=25 \
  --save --outfile astronaut.png

# ✨ Image saved to ./omni_out/astronaut.png
```

**That's it.** It just works — reproducibly, every time.

**Tip:** Use `--help` to explore any runner:
```bash
omni run omnilaunch/sdxl:0.1.0 --help       # List all entrypoints
omni run omnilaunch/sdxl:0.1.0 infer --help # Show parameters for infer
```

---

## 🧩 What Omnilaunch Does

Omnilaunch provides a lightweight, opinionated framework for defining **reproducible AI workflows** — from quick tests to full-scale training pipelines.

You can:

* 🧠 **Prototype fast:** Run a single model or test a new LoRA without environment setup.
* ⚙️ **Reproduce experiments:** Capture exact code, weights, and dependencies for reliable reruns.
* 🧪 **Benchmark at scale:** Standardize evaluation across many models or tasks.
* 🎓 **Teach or share workflows:** Package multi-stage examples (e.g., pretrain → RL → alignment → chat) as reusable runners.
* 🚀 **Reproduce production-scale jobs:** Launch the same training workflows used in papers or production systems — deterministically.

It's flexible enough for small experiments and powerful enough to re-run large-scale jobs.

---

## 🧠 Why It Exists

Running open-source AI models today often means juggling fragile Colab notebooks, expensive hosted endpoints, or custom scripts that drift over time.
Omnilaunch fixes that by providing a **repeatable, transparent, versioned** way to build and execute GPU workloads.

* **Deterministic builds:** Modal images are immutable once built.
* **Versioned workflows:** Every runner build gets a unique hash and tarball.
* **No boilerplate:** Define a `modal_app.py` and a config — Omnilaunch handles the rest.
* **It just works:** If your workflow runs locally with Python, it'll run the same way in Modal — reproducibly.

---

## 👥 Who Is This For

| Persona                   | Use Case                                   | Example                                                                         |
| ------------------------- | ------------------------------------------ | ------------------------------------------------------------------------------- |
| **Builders & OSS devs**   | Quickly test new models or adapters        | Spin up 5 text-to-image models and compare prompts and outputs.                             |
| **Researchers**           | Package and reproduce experiments          | Bundle a paper's training pipeline as a reproducible runner.                    |
| **Educators**             | Teach AI workflows step-by-step            | Define entrypoints for pretrain → RL → alignment → test.                        |
| **Students & learners**   | Run heavy workflows without setup          | Use Modal GPUs for small experiments safely and reproducibly.                   |
| **Automation / AI tools** | Generate or extend workflows automatically | AI writes new runners using the standard pattern, usually correct on first try. |
| **Teams & startups**      | Ensure consistency in internal workflows   | Every run has a version, hash, and reproducible setup.                          |

---

## 💡 Why Modal?

**Transparent & Predictable:**
- Per-second GPU billing (not per-step like Fireworks.ai and other similar platforms)
- [Pick your GPU](https://modal.com/pricing) (A10G, H100, etc.) — no black-box allocation
- Modal shows exact execution time and cost

**Just Python:**
- No Kubernetes, no Docker builds, no YAML hell
- Simple decorators + functions = deployed model
- Persistent volumes for model caching (no re-downloads)

**Reliable:**
- Other platforms have frequent compatibility issues or cryptic errors and failures
- Modal's abstraction is minimal — closer to "your code on a GPU"
- Clear error messages, real logs (not cryptic failures)

**Supportive Company:**
- Modal provides credit grants for [academics](https://modal.com/academics) and [startups](https://modal.com/startups)
- If you have committed spend with AWS, Azure, GCP, or OCI, you will soon be able to use that commit on Modal
- Get free credits ($30/mo of free credits on the Free plan, $100/mo of free credits on the $250/mo plan) to put towards your compute

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

# Benchmark on tinyMMLU (60% accuracy on 10 samples, ~$0.12)
omni run omnilaunch/gpt-oss-20b:0.1.0 benchmark_tinymmlu \
  -p max_items=10 \
  -p reasoning_level=low \
  --save --outfile benchmark.json
```

Returns structured JSON with `response`, `analysis` (internal reasoning), and `commentary` (meta-observations). Benchmarking returns accuracy metrics, per-subject breakdown, and sample predictions.

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

### 🔧 **Fine-tuning Vision-Language Models**
Train Qwen3-VL-8B on custom datasets with LoRA:

```bash
# Fine-tune on LaTeX OCR dataset (200 samples, ~8 min, ~$0.13)
omni run omnilaunch/qwen3-vl:0.1.0 train_lora \
  -p dataset_uri="unsloth/LaTeX_OCR" \
  -p epochs=1 \
  -p max_train_samples=200 \
  --save --outfile training_results.json

# Inference with your trained LoRA
omni run omnilaunch/qwen3-vl:0.1.0 infer \
  -p image="handwritten_math.png" \
  -p prompt="Convert this to LaTeX" \
  -p lora_run="brave-lion-1234" \
  --save --outfile output.json

# Pre-cache datasets on CPU (free!)
omni run omnilaunch/qwen3-vl:0.1.0 download_dataset \
  -p dataset_uri="unsloth/LaTeX_OCR"
```

LoRA adapters are saved with human-readable names (e.g., "brave-lion-1234") to `/omnilaunch/runs/qwen3-vl/`. Supports WandB integration for tracking metrics.

### 🔬 **Benchmarking**
Evaluate models on standard datasets:

```bash
# GPT-OSS-20B on tinyMMLU (100 samples, ~65 min, ~$1.19)
omni run omnilaunch/gpt-oss-20b:0.1.0 benchmark_tinymmlu \
  -p max_items=100 \
  -p reasoning_level=low \
  --save --outfile results.json

# Quick sanity check (10 samples, ~6.5 min, ~$0.12)
omni run omnilaunch/gpt-oss-20b:0.1.0 benchmark_tinymmlu \
  -p max_items=10 \
  --save
```

Results include overall accuracy, per-subject breakdown, sample predictions, and full reproducibility metadata.

### 📋 **List All Runners**
```bash
omni list

# Shows all runners (built and unbuilt) with status indicators:
#
# Available Runners (5):
#
#   [✓] omnilaunch/gpt-oss-20b
#       Latest: 0.1.0
#       Entrypoints: download_files (CPU), setup (CPU), infer (A10G), benchmark_tinymmlu (A10G)
#
#   [ ] omnilaunch/qwen3-vl
#       Status: Not built yet
#       Build: omni build omnilaunch/registry/qwen3-vl
#
#   [✓] omnilaunch/sdxl
#       Latest: 0.1.0
#       Entrypoints: download_files (CPU), setup (CPU), infer (A10G)
#
# Built: 2, Not built: 3
```

---

## ⚙️ How It Works

Each workflow is packaged as a **runner** — a directory with Python code and metadata describing how to execute a particular AI process.

### Runner Structure

```bash
omnilaunch/registry/sdxl/
├── modal_app.py       # Defines modal.Functions and entrypoints (train, infer, benchmark, etc.)
├── runner.yaml        # Metadata: name, version, GPU per entrypoint
├── schema/
│   ├── infer.json     # JSON schema for inference params
│   └── dataset.json   # JSON schema for training data
├── tests/smoke.py     # Basic smoke tests
└── README.md          # Workflow card with usage examples
```

Each runner can contain *any code* that can be written in Python and benefits from execution on a GPU. Common use cases:

* Model training (LLM, diffusion, RLHF, etc.)
* LoRA or fine-tuning workflows
* Evaluation and benchmarking scripts
* Multi-phase experiments or educational demos

**Runners can represent:**
- Official model deployments (e.g., Stability AI's SDXL)
- Optimized inference stacks (e.g., vLLM for LLMs, optimized diffusion pipelines)
- SaaS-ready backends (e.g., PhotoAI-style personalized image generation)
- Benchmarking setups (e.g., `omnilaunch/gpt-oss-20b` with `benchmark_tinymmlu` entrypoint)
- Educational tools (e.g., compare pre-alignment vs post-alignment LLM checkpoints)

### CLI Flow

| Command      | What it does                                                                                                             |
| ------------ | ------------------------------------------------------------------------------------------------------------------------ |
| `omni build` | Packages a runner directory into a versioned tarball and computes a hash. May later support signing for provenance.      |
| `omni setup` | Deploys the Modal app (builds Docker image, downloads weights, verifies environment). One-time setup per runner version. |
| `omni run`   | Invokes an entrypoint (train, infer, benchmark, etc.) on Modal's GPUs. Supports parameters and output saving.            |

### Entrypoints

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

**Everything is just:** `omni run <runner> <entrypoint>`

### Reproducibility

Every runner build creates:
- **Versioned tarball** — `runner.tar.gz` with all code and configs
- **Manifest hash** — Unique identifier for the exact runner version
- **Pinned dependencies** — Immutable Modal images with locked package versions

Coming soon:
- **Signed manifests** — Cryptographically verified with Sigstore/GPG
- **Run provenance** — Full audit trail from data → model → results

**Key features:**
- **Versioned** — `omnilaunch/sdxl:0.1.0`, `omnilaunch/sdxl:0.2.0`
- **Bundled** — Packaged as `.tar.gz` with manifest and schemas
- **Signed** *(coming)* — Cryptographically verified with Sigstore/GPG
- **Backend-agnostic** *(future)* — Modal first, then AWS/RunPod/local via IaC tools

---

## 🚧 Current Limitations

* 🧩 Runs **only on Modal** for now (multi-backend support is possible with IaC tooling).
* 🌐 No web UI yet (planned registry + metrics portal, like Docker Hub for AI workflows).

---

## 🔮 Future Vision

* **Public registry of verified runners:** Browse, version, and reproduce any workflow.
* **Provenance tracking & signing:** Each runner hash verifiable via Sigstore or similar.
* **Community templates:** Easy scaffolds for HF, PyTorch, or custom training setups.
* **Web interface:** Launch workflows, view logs, track metrics, and share results.
* **Automated runner generation:** LLMs that extend the registry automatically.

---

## 🚀 Roadmap

**✅ Available Now (v0.1):**
- LLM inference (GPT-OSS-20B/120B with reasoning)
- LLM benchmarking (tinyMMLU with reproducible results)
- Vision-language fine-tuning (Qwen3-VL-8B with LoRA)
- Image generation (SDXL + pixel-art + SPO variants)
- Modal deployment & execution
- CLI: `build`, `setup`, `run`, `list`, `doctor`
- Reproducible manifests with pinned dependencies
- WandB integration for training metrics

**📋 Planned (v0.2+):**
- Gabe will add several more runners for model training/fine tuning and sampling

**🌍 Future (community-driven):**
- Multi-backend support (AWS, RunPod, local via IaC)
- Playground UI (open-source, connects to your Modal)
- SaaS starter templates
- Production-grade inference
- Whatever the community needs and is willing to contribute!

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
# Available now: tinyMMLU for GPT-OSS models
omni run omnilaunch/gpt-oss-20b:0.1.0 benchmark_tinymmlu \
  -p max_items=100 \
  --save --outfile results.json

# Coming soon: more benchmarks
omni run omnilaunch/gpt-oss-20b:0.1.0 benchmark_gsm8k
omni run omnilaunch/gpt-oss-20b:0.1.0 benchmark_humaneval
omni run omnilaunch/diffusion-bench:0.1.0 fid --dataset coco

# → Signed results feed reproducible leaderboards
```

Each benchmark includes full provenance: model version, dataset version, parameters, and hardware specs. Verifiable, reproducible results.

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

- `-h, --help` - Show entrypoints for a runner or parameters for a specific entrypoint
- `--params <file or JSON>` - Parameters as YAML/JSON file or inline string
- `-p, --param key=value` - Individual param overrides (repeatable)
- `--save` - Auto-save outputs to disk
- `--outdir <path>` - Directory for saved outputs (default: `./omni_out`)
- `--outfile <name>` - Explicit output filename
- `--dataset <uri>` - Dataset URI for training (e.g., `hf:org/dataset` or `vol:/path`)

### Getting Help

```bash
# List all entrypoints for a runner
omni run omnilaunch/sdxl:0.1.0 --help

# Show parameters for a specific entrypoint
omni run omnilaunch/sdxl:0.1.0 infer --help

# Output shows:
# - Entrypoint function and GPU requirement
# - All parameters with types, defaults, and descriptions (from JSON schema)
# - Usage examples
```

---

## 🌍 The Goal

Omnilaunch aims to be the standard way to package and run AI models — reproducible, transparent, and accessible to everyone.

Built on **Modal + Hugging Face** for open, reproducible infrastructure.

---

## 🤝 Contributing

Contributions are welcome!
If you'd like to add a new runner, check the `/registry` directory and open a PR.
Runners follow a consistent structure — `build → setup → run` — making it easy for both humans and AI to add new workflows.

---

## 💬 Feedback

Open an issue or tag [@gabebusto](https://x.com/gabebusto) on X for feedback, ideas, or collaboration.
If you're at Modal, Hugging Face, Replicate, Fal, or any infra team building for reproducible AI — I'd love to chat!

---

## 🙏 Credits

Built on:
- [Modal](https://modal.com) — Serverless GPU infrastructure
- [Hugging Face](https://huggingface.co) — Models & datasets
- The open-source AI community

---

**Omnilaunch — AI workflows that just work.**
