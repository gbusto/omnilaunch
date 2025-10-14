"""
Minimal smoke test for the SDXL runner.

Ideas for future enhancement:
- Try importing diffusers/torch (CPU), just to catch missing wheels early
- Validate environment variables or secrets needed by the runner (e.g., HF tokens)
- Perform a tiny no-op call to a helper in modal_app (if exposed for CPU)

The CLI will run this inside a lightweight Modal container during `omni preflight`.
"""

import modal
from modal import Image


def run() -> bool:
    # Ensure Modal SDK can be imported and a trivial image can be constructed
    _ = Image.debian_slim().pip_install("huggingface_hub")
    # Optional: uncomment to assert CPU-only imports
    # import torch  # noqa: F401
    # import diffusers  # noqa: F401
    print("smoke: basic environment looks OK")
    return True


