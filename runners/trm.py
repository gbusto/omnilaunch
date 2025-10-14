import os, json, modal
from pathlib import Path
from typing import Optional, Dict, List

# ================================
# App & Volume layout
# ================================
APP_NAME = "omnilaunch-trm"
VOLUME_NAME = "omnilaunch"
MOUNT_ROOT = "/omnilaunch"

DIR_CODE = f"{MOUNT_ROOT}/code/trm"          # TRM repo lives here
DIR_MODELS = f"{MOUNT_ROOT}/models"          # reserved for future caches
DIR_DATA = f"{MOUNT_ROOT}/data"              # datasets (pre-built)
DIR_RUNS = f"{MOUNT_ROOT}/runs/trm"          # run outputs + training.json

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# ================================
# Base image (CUDA 12.6 + Python 3.12)
# ================================
image = (
    # Use devel image so nvcc is available for building CUDA extensions like adam-atan2
    modal.Image.from_registry("nvidia/cuda:12.6.2-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential")
    # Core packaging/build tooling
    .pip_install(
        "pip",
        "wheel",
        "setuptools",
        "setuptools_scm",
        "ninja",
        "packaging",
    )
    # CUDA 12.6 torch stack (let pip resolve compatible versions)
    .pip_install(
        "torch",
        "torchvision",
        "torchaudio",
        index_url="https://download.pytorch.org/whl/cu126",
    )
    # TRM Python deps (excluding 'torch' and 'adam-atan2' which we handle explicitly)
    .pip_install(
        "einops",
        "tqdm",
        "coolname",
        "pydantic",
        "argdantic",
        "wandb",
        "omegaconf",
        "hydra-core",
        "huggingface_hub",
        "pydantic-core",
        "numba",
        "llvmlite",
        "triton",
    )
    # adam-atan2 requires nvcc; install with flags via a build command to avoid pip_install extra_options issues
    .run_commands(["CXX=g++ CC=gcc python -m pip install --no-cache-dir --no-build-isolation adam-atan2"])
)


# ================================
# Small helpers
# ================================
def _write_json_file(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _update_status(run_dir: Path, update: Dict) -> Dict:
    status_path = run_dir / "training.json"
    current: Dict = {}
    if status_path.exists():
        try:
            current = json.loads(status_path.read_text())
        except Exception:
            current = {}
    current.update(update)
    current["updated_at"] = current.get("updated_at") or current.get("started_at")
    _write_json_file(status_path, current)
    return current


# ================================
# 1) Downloader: clone/update TRM repo
# ================================
@app.function(
    image=image,
    volumes={MOUNT_ROOT: vol},
    timeout=60 * 40,
)
def download_files(
    repo_url: str = "https://github.com/SamsungSAILMontreal/TinyRecursiveModels",
    branch: str = "main",
    install_requirements: bool = False,
) -> Dict[str, str]:
    """
    Clone or update the TinyRecursiveModels repo under /omnilaunch/code/trm/TinyRecursiveModels.
    Optionally installs requirements (CPU-only) using the repo's requirements.txt.
    """
    import subprocess, sys

    os.makedirs(DIR_CODE, exist_ok=True)
    code_path = Path(DIR_CODE) / "TinyRecursiveModels"

    if code_path.exists():
        try:
            subprocess.run(["git", "-C", str(code_path), "fetch", "--all", "-p"], check=True)
            subprocess.run(["git", "-C", str(code_path), "checkout", branch], check=True)
            subprocess.run(["git", "-C", str(code_path), "pull", "--ff-only"], check=True)
        except Exception:
            subprocess.run(["rm", "-rf", str(code_path)], check=False)
            subprocess.run(["git", "clone", "-b", branch, "--depth=1", repo_url, str(code_path)], check=True)
    else:
        subprocess.run(["git", "clone", "-b", branch, "--depth=1", repo_url, str(code_path)], check=True)

    if install_requirements:
        req = code_path / "requirements.txt"
        if req.exists():
            print(f"[install] requirements: {req}")
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req), "--no-cache-dir", "--no-build-isolation"], check=True)
        else:
            print("[install] requirements.txt not found; skipping")

    vol.commit()
    return {"code_dir": str(code_path)}


# ================================
# 2) Train Full (Sudoku attention defaults)
# ================================
@app.function(
    image=image,
    gpu="L40S",
    volumes={MOUNT_ROOT: vol},
    timeout=60 * 60 * 24,
    secrets=[
        modal.Secret.from_name("wandb-secret"),  # optional
    ],
)
def train_full(
    codename: str,
    dataset_dir: str,
    params: Optional[Dict] = None,
    repo_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    Launch TRM training using pretrain.py with attention-based Sudoku defaults.

    - codename: run identifier used under /omnilaunch/runs/trm/<codename>
    - dataset_dir: absolute or relative to /omnilaunch/data
    - params: optional overrides for the recipe
    - repo_dir: override for the TRM repo directory (defaults to /omnilaunch/code/trm/TinyRecursiveModels)
    """
    import subprocess, sys, time, shutil

    # Resolve repo path
    repo_path = Path(repo_dir) if repo_dir else (Path(DIR_CODE) / "TinyRecursiveModels")
    pretrain_py = repo_path / "pretrain.py"
    assert pretrain_py.exists(), f"pretrain.py not found at {pretrain_py}; run download_files() first"

    # Resolve dataset path
    ds_path = Path(dataset_dir)
    if not ds_path.is_absolute():
        ds_path = Path(DIR_DATA) / dataset_dir
    assert ds_path.exists(), f"dataset_dir not found: {ds_path}"

    # Prepare run directory
    run_dir = Path(DIR_RUNS) / codename
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # training.json (not_started -> running -> completed|errored)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    status = {
        "status": "not_started",
        "config": {
            "model": "trm",
            "type": "full",
            "gpu": "L40S",
            "params": params or {},
            "dataset": str(ds_path),
        },
        "started_at": started_at,
        "updated_at": started_at,
        "modal_run_url": "",
        "wandb_url": "",
        "run_dir": str(run_dir),
    }
    _write_json_file(run_dir / "training.json", status)
    _update_status(run_dir, {"status": "running"})

    # Environment
    if os.environ.get("WANDB_API_KEY"):
        os.environ.setdefault("WANDB_PROJECT", "omnilaunch-trm")
        os.environ.setdefault("WANDB_NAME", codename)

    # Default Sudoku attention params (can be overridden)
    default_params: Dict[str, str] = {
        "arch": "trm",
        "data_paths": f"[{ds_path}]",
        "evaluators": "[]",
        "epochs": "50000",
        "eval_interval": "5000",
        "lr": "1e-4",
        "puzzle_emb_lr": "1e-4",
        "weight_decay": "1.0",
        "puzzle_emb_weight_decay": "1.0",
        "arch.L_layers": "2",
        "arch.H_cycles": "3",
        "arch.L_cycles": "6",
        "+run_name": codename,
        "ema": "True",
        # Hydra: direct outputs under our run dir
        "hydra.run.dir": str(run_dir / "hydra-out"),
    }

    if params:
        for k, v in params.items():
            default_params[str(k)] = str(v)

    # Convert params dict to CLI args
    cli_args: List[str] = []
    for k, v in default_params.items():
        cli_args.append(f"{k}={v}")

    cmd = [
        sys.executable,
        str(pretrain_py),
        *cli_args,
    ]

    print("Launching:", " ".join(cmd))
    try:
        subprocess.run(cmd, cwd=str(repo_path), check=True)
        finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _update_status(run_dir, {"status": "completed", "finished_at": finished_at})
        vol.commit()
        return {"run_dir": str(run_dir), "dataset_used": str(ds_path), "status": "completed"}
    except subprocess.CalledProcessError as e:
        print(f"[train_full] Training failed with exit code {e.returncode}")
        finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _update_status(run_dir, {"status": "errored", "finished_at": finished_at, "error_code": e.returncode})
        vol.commit()
        raise RuntimeError(f"Training failed with exit code {e.returncode}") from e
    except Exception as e:
        print(f"[train_full] Unexpected error: {e}")
        finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _update_status(run_dir, {"status": "errored", "finished_at": finished_at, "error": str(e)})
        vol.commit()
        raise


# ================================
# 3) Inference (MVP placeholder)
# ================================
@app.function(
    image=image,
    volumes={MOUNT_ROOT: vol},
    timeout=60 * 10,
    scaledown_window=2,
)
def infer(
    codename: Optional[str] = None,
    run_dir: Optional[str] = None,
    checkpoint: Optional[str] = None,
    input_text: str = "",
) -> Dict[str, str]:
    """
    MVP inference placeholder for Sudoku: accepts a string and echoes back.
    A future revision will load a checkpoint and perform single-sample decoding.
    """
    resolved_run = Path(run_dir) if run_dir else (Path(DIR_RUNS) / (codename or ""))
    result = {
        "run_dir": str(resolved_run) if resolved_run else "",
        "checkpoint": checkpoint or "",
        "input": input_text,
        "output": input_text,
        "note": "TRM inference not implemented yet; returns echo for now",
    }
    return result


# ================================
# 4) Quick layout helper
# ================================
@app.function(image=image, volumes={MOUNT_ROOT: vol})
def layout() -> Dict[str, List[str]]:
    def ls(p: str) -> List[str]:
        path = Path(p)
        if not path.exists():
            return []
        return [x.name for x in sorted(path.iterdir())]

    return {
        "code": ls(DIR_CODE),
        "models": ls(DIR_MODELS),
        "data": ls(DIR_DATA),
        "runs_trm": ls(DIR_RUNS),
    }


if __name__ == "__main__":
    app.deploy()
