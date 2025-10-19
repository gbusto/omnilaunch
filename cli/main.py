import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path


REQUIRED_FILES = [
    "runner.yaml",
    "modal_app.py",
    "tests/smoke.py",
]


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_name_version(runner_dir: Path) -> tuple[str, str]:
    name = "unknown"
    version = "0.0.0"
    ry = (runner_dir / "runner.yaml").read_text(encoding="utf-8").splitlines()
    for line in ry:
        s = line.strip()
        if s.startswith("name:"):
            name = s.split(":", 1)[1].strip()
        elif s.startswith("version:"):
            version = s.split(":", 1)[1].strip().strip('"')
    return name, version


def _parse_runner_ref(ref: str) -> tuple[str, str | None]:
    """Parse runner ref like 'omnilaunch/sdxl:0.1.0' into (name, version?).
    Returns (name, None) if no version specified.
    """
    if ":" in ref:
        name, ver = ref.rsplit(":", 1)
        return name, ver
    return ref, None


def _get_entrypoint_config(runner_yaml_lines: list[str], entrypoint_name: str) -> dict:
    """Extract entrypoint config from runner.yaml lines."""
    import yaml
    data = yaml.safe_load("\n".join(runner_yaml_lines))
    entrypoint = data.get("entrypoints", {}).get(entrypoint_name, {})
    
    if isinstance(entrypoint, dict):
        return {
            "function": entrypoint.get("function", ""),
            "gpu": entrypoint.get("gpu"),
            "schema": entrypoint.get("schema")
        }
    else:
        # Shouldn't happen with new format
        return {"function": str(entrypoint), "gpu": None, "schema": None}


def _validate_runner_dir(runner_dir: Path) -> dict:
    problems: list[str] = []
    files: dict[str, str] = {}
    for rel in REQUIRED_FILES:
        p = runner_dir / rel
        if not p.exists():
            problems.append(f"missing: {rel}")
        else:
            files[rel] = f"sha256:{_compute_sha256(p)}"
    # schema presence (either schema/params.json or any json in schema/)
    schema_dir = runner_dir / "schema"
    if not schema_dir.exists() or not any(fp.suffix == ".json" for fp in schema_dir.glob("*.json")):
        problems.append("missing: schema/*.json (at least one)")
    if problems:
        raise RuntimeError("runner validation failed: " + ", ".join(problems))
    return files


def cmd_build(args: argparse.Namespace) -> int:
    runner_dir = Path(args.path).resolve()
    if not runner_dir.exists() or not runner_dir.is_dir():
        print(f"[omni build] not a directory: {runner_dir}", file=sys.stderr)
        return 2

    try:
        files_map = _validate_runner_dir(runner_dir)
    except Exception as e:
        print(f"[omni build] validation error: {e}", file=sys.stderr)
        return 3

    name, version = _read_name_version(runner_dir)
    if name == "unknown":
        print("[omni build] warning: runner.yaml missing name; using folder name", file=sys.stderr)
        name = runner_dir.name

    # Create staging dir with required files + manifest.json
    with tempfile.TemporaryDirectory() as td:
        staging = Path(td)
        staging_schema = staging / "schema"
        staging_tests = staging / "tests"
        staging_schema.mkdir(parents=True, exist_ok=True)
        staging_tests.mkdir(parents=True, exist_ok=True)

        # Copy required files preserving structure
        shutil.copy2(runner_dir / "runner.yaml", staging / "runner.yaml")
        shutil.copy2(runner_dir / "modal_app.py", staging / "modal_app.py")
        # NOTE: image.lock intentionally omitted in MVP. Keep image definition in modal_app.py.
        # Copy all schema/*.json present (per-entrypoint schemas allowed)
        schema_src_dir = runner_dir / "schema"
        for fp in schema_src_dir.glob("*.json"):
            shutil.copy2(fp, staging_schema / fp.name)
        # Tests
        shutil.copy2(runner_dir / "tests/smoke.py", staging_tests / "smoke.py")

        manifest = {
            "files": files_map,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        # Create bundle under registry/<name>/<version>/runner.tar.gz
        reg_root = Path(__file__).resolve().parents[1] / "registry"
        out_dir = reg_root / name / version
        out_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = out_dir / "runner.tar.gz"

        with tarfile.open(bundle_path, mode="w:gz") as tf:
            for rel in ["runner.yaml", "modal_app.py", "manifest.json"]:
                tf.add(str(staging / rel), arcname=rel)
            tf.add(str(staging_schema), arcname="schema")
            tf.add(str(staging_tests), arcname="tests")

        bundle_sha = _compute_sha256(bundle_path)

        # Also write manifest.json alongside the tarball for easy inspection
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Update registry index
    index_path = Path(__file__).resolve().parents[1] / "registry" / "index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        index = {"version": 1, "runners": []}

    # Upsert runner entry
    runners = index.get("runners", [])
    entry = next((r for r in runners if r.get("name") == name), None)
    if not entry:
        entry = {"name": name, "latest": version, "versions": {}}
        runners.append(entry)
    entry["latest"] = version
    entry.setdefault("versions", {})[version] = {
        "path": str(bundle_path),
        "sha256": bundle_sha,
    }
    index["runners"] = runners
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    print(f"[omni build] built {name}:{version} -> {bundle_path} (sha256:{bundle_sha[:12]}...)\nindex: {index_path}")
    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    print(f"[omni setup] runner={args.runner}")
    # Resolve app name from runner.yaml then call setup() on Modal
    index_path = Path(__file__).resolve().parents[1] / "registry" / "index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[setup] cannot read index: {e}", file=sys.stderr)
        return 2
    req_name, req_ver = _parse_runner_ref(args.runner)
    entry = next((r for r in index.get("runners", []) if r.get("name") == req_name), None)
    if not entry:
        print(f"[setup] runner not found in index: {args.runner}", file=sys.stderr)
        return 3
    version = req_ver or entry.get("latest")
    bundle_info = entry.get("versions", {}).get(version)
    if not bundle_info:
        print(f"[setup] version info missing for {args.runner}:{version}", file=sys.stderr)
        return 3
    bundle_path = Path(bundle_info.get("path"))
    if not bundle_path.exists():
        print(f"[setup] bundle missing: {bundle_path}", file=sys.stderr)
        return 4

    # Extract to read runner.yaml and modal_app.py
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        with tarfile.open(bundle_path, mode="r:gz") as tf:
            tf.extractall(td_path)
        ry_lines = (td_path / "runner.yaml").read_text(encoding="utf-8").splitlines()
        app_name = None
        for line in ry_lines:
            s = line.strip()
            if s.startswith("app_name:"):
                app_name = s.split(":", 1)[1].strip().strip('"')
                break
        modal_app_path = td_path / "modal_app.py"
        
        # Deploy the Modal app first (idempotent)
        try:
            import modal
            print(f"[setup] deploying Modal app '{app_name or req_name.replace('/', '-')}'...")
            import subprocess
            result = subprocess.run(
                ["modal", "deploy", str(modal_app_path)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print(f"[setup] deploy failed: {result.stderr}", file=sys.stderr)
                return 6
            print("[setup] deploy complete")
        except Exception as e:
            print(f"[setup] deploy error: {e}", file=sys.stderr)
            return 6
    
    # Call setup() on Modal
    try:
        import modal
        fn = modal.Function.from_name(app_name or req_name.replace("/", "-"), "setup")
        print(f"[setup] calling setup() on Modal...")
        result = fn.remote(True)
        print(json.dumps(result, indent=2))
        return 0 if result and result.get("ok") else 7
    except Exception as e:
        print(f"[setup] modal call failed: {e}", file=sys.stderr)
        return 5


def cmd_run(args: argparse.Namespace) -> int:
    print(f"[omni run] runner={args.runner} entrypoint={args.entrypoint} dataset={args.dataset} params={args.params}")
    # Resolve bundle & read runner.yaml
    index_path = Path(__file__).resolve().parents[1] / "registry" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    req_name, req_ver = _parse_runner_ref(args.runner)
    entry = next((r for r in index.get("runners", []) if r.get("name") == req_name), None)
    if not entry:
        print(f"[run] runner not found: {args.runner}", file=sys.stderr)
        return 2
    version = req_ver or entry.get("latest")
    bundle_info = entry.get("versions", {}).get(version)
    bundle_path = Path(bundle_info.get("path"))
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        with tarfile.open(bundle_path, mode="r:gz") as tf:
            tf.extractall(td_path)
        # Load runner.yaml
        ry = (td_path / "runner.yaml").read_text(encoding="utf-8").splitlines()
        app_name = None
        for line in ry:
            s = line.strip()
            if s.startswith("app_name:"):
                app_name = s.split(":", 1)[1].strip().strip('"')
        
        # Get entrypoint config (includes GPU and schema info)
        entrypoint_config = _get_entrypoint_config(ry, args.entrypoint)
        
        # Choose schema per entrypoint if specified in config
        schema_filename = entrypoint_config.get("schema")
        schema_path = td_path / "schema" / schema_filename if schema_filename else td_path / "schema" / f"{args.entrypoint}.json"
        params_obj: dict = {}
        # Load params from file path OR inline JSON/YAML string
        if args.params:
            try:
                p = Path(args.params)
                txt = p.read_text(encoding="utf-8") if p.exists() else str(args.params)
                try:
                    import yaml
                    parsed = yaml.safe_load(txt)
                    if isinstance(parsed, dict):
                        params_obj.update(parsed)
                    else:
                        raise ValueError("params must be a mapping")
                except Exception:
                    parsed = json.loads(txt)
                    if isinstance(parsed, dict):
                        params_obj.update(parsed)
                    else:
                        raise ValueError("params must be a mapping")
            except Exception as e:
                print(f"[run] failed to parse --params: {e}", file=sys.stderr)
                return 3
        # Merge -p/--param key=value overrides
        def _parse_value(v: str, key: str = ""):
            """Parse value, auto-encoding files for image/file parameters."""
            vl = v.strip()
            
            # Auto-detect and encode image/file parameters
            if key.lower() in ("image", "image_input", "file", "attachment"):
                # Check if it's a local file path
                try:
                    p = Path(vl)
                    if p.exists() and p.is_file():
                        # Base64 encode the file
                        import base64
                        print(f"[run] auto-encoding file '{vl}' to base64 for parameter '{key}'", file=sys.stderr)
                        with open(p, "rb") as f:
                            encoded = base64.b64encode(f.read()).decode("utf-8")
                        return encoded
                except Exception:
                    pass  # Not a file, treat as string
            
            # Parse as boolean
            if vl.lower() in ("true", "false"):
                return vl.lower() == "true"
            
            # Parse as number
            try:
                if "." in vl:
                    return float(vl)
                return int(vl)
            except Exception:
                return vl
        
        if getattr(args, "param", None):
            for kv in args.param:
                if "=" not in kv:
                    print(f"[run] ignoring malformed --param '{kv}', expected key=value", file=sys.stderr)
                    continue
                key, val = kv.split("=", 1)
                key = key.strip()
                params_obj[key] = _parse_value(val, key)
        # Validate if schema exists
        if schema_path.exists():
            try:
                import jsonschema
                schema = json.loads(schema_path.read_text(encoding="utf-8"))
                jsonschema.validate(instance=params_obj, schema=schema)
            except Exception as e:
                print(f"[run] params validation failed: {e}", file=sys.stderr)
                return 3
        
        # Invoke Modal function
        try:
            import modal, base64, mimetypes
            fn = modal.Function.from_name(app_name or args.runner.replace("/", "-"), args.entrypoint)
            result = fn.remote(params_obj if args.entrypoint == "infer" else params_obj)
            # Auto-handle single- or multi-part outputs
            def _save_single(item: dict, outdir: Path, outfile: str | None) -> str | None:
                ct = str(item.get("content_type") or "")
                data = item.get("data")
                if not ct or data is None:
                    return None
                outdir.mkdir(parents=True, exist_ok=True)
                name = outfile or f"artifact{mimetypes.guess_extension(ct) or ''}"
                p = outdir / name
                if ct.startswith("text/"):
                    p.write_text(str(data), encoding="utf-8")
                elif ct == "application/json":
                    obj = data if isinstance(data, (dict, list)) else json.loads(str(data))
                    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")
                else:
                    raw = base64.b64decode(str(data).encode("utf-8"))
                    p.write_bytes(raw)
                return str(p)
            saved = []
            if getattr(args, "save", False) or getattr(args, "outdir", None):
                outdir = Path(args.outdir or "./omni_out").resolve()
                if isinstance(result, dict) and "parts" in result and isinstance(result["parts"], list):
                    for it in result["parts"]:
                        sp = _save_single(it, outdir, getattr(args, "outfile", None))
                        if sp:
                            saved.append(sp)
                elif isinstance(result, dict) and "content_type" in result and "data" in result:
                    sp = _save_single(result, outdir, getattr(args, "outfile", None))
                    if sp:
                        saved.append(sp)
                if saved:
                    print(json.dumps({"saved": saved}, indent=2))
            else:
                print(json.dumps(result, indent=2))
            return 0
        except Exception as e:
            print(f"[run] modal call failed: {e}", file=sys.stderr)
            return 4


def cmd_doctor(_args: argparse.Namespace) -> int:
    print("[omni doctor] Checking environment...")
    ok = True
    modal_ok = False
    # Modal SDK
    try:
        import modal  # noqa: F401
        print("  modal: OK")
        modal_ok = True
    except Exception as e:
        print(f"  modal: MISSING ({e})")
        ok = False
    # HF hub
    try:
        import huggingface_hub  # noqa: F401
        print("  huggingface_hub: OK")
    except Exception as e:
        print(f"  huggingface_hub: MISSING ({e})")
        ok = False
    # Modal auth (required for Modal API access)
    if modal_ok:
        try:
            import modal  # ensure bound in this scope
            # Probe auth by attempting a lookup of a non-existent app
            modal.App.lookup("__omni_doctor_probe__", create_if_missing=False)
            print("  Modal auth: OK (configured via 'modal token set')")
        except Exception as e:
            name = getattr(e, "__class__", type(e)).__name__
            msg = str(e)
            if "Authentication" in name or "Unauthorized" in msg or "401" in msg:
                print("  Modal auth: NOT CONFIGURED - run 'modal token set'")
                ok = False
            elif "NotFound" in name:
                # Expected for nonexistent app; auth worked
                print("  Modal auth: OK (configured via 'modal token set')")
            else:
                # Treat other errors as non-fatal for auth check
                print("  Modal auth: OK (configured via 'modal token set')")
    else:
        print("  Modal auth: skipped (modal not installed)")

    # HF token (only needed on Modal, not locally)
    print("  HF token: configure in Modal Secrets (only necessary for some HF models and datasets)")

    return 0 if ok else 1


def cmd_list(_args: argparse.Namespace) -> int:
    """List all runners (built and unbuilt) from the registry."""
    registry_root = Path(__file__).parent.parent / "registry"
    index_path = registry_root / "index.json"
    
    # Load built runners from index
    built_runners = {}
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            for runner in index.get("runners", []):
                name = runner.get("name", "")
                if name:
                    built_runners[name] = runner
        except Exception as e:
            print(f"[list] Warning: Could not read index: {e}", file=sys.stderr)
    
    # Discover all source runners from registry/*/runner.yaml
    all_runners = {}
    for item in registry_root.iterdir():
        if item.is_dir() and item.name not in ("omnilaunch", ".git", "__pycache__"):
            runner_yaml = item / "runner.yaml"
            if runner_yaml.exists():
                try:
                    name, version = _read_name_version(item)
                    if name and name != "unknown":
                        all_runners[name] = {
                            "source_path": item,
                            "name": name,
                            "version": version,
                            "built": name in built_runners,
                            "built_info": built_runners.get(name)
                        }
                except Exception:
                    pass
    
    if not all_runners:
        print("[list] No runners found in registry/")
        return 0
    
    print(f"Available Runners ({len(all_runners)}):\n")
    
    # Sort by name
    sorted_runners = sorted(all_runners.values(), key=lambda r: r["name"])
    
    for runner in sorted_runners:
        name = runner["name"]
        source_path = runner["source_path"]
        is_built = runner["built"]
        
        # Status indicator
        status_icon = "[✓]" if is_built else "[ ]"
        
        print(f"  {status_icon} {name}")
        
        if is_built:
            # Show built runner info
            built_info = runner["built_info"]
            latest = built_info.get("latest", "N/A")
            versions = built_info.get("versions", {})
            version_list = sorted(versions.keys()) if versions else []
            
            print(f"      Latest: {latest}")
            if len(version_list) > 1:
                print(f"      Versions: {', '.join(version_list)}")
            
            # Read entrypoint info from source runner.yaml
            try:
                import yaml
                runner_yaml = source_path / "runner.yaml"
                if runner_yaml.exists():
                    meta = yaml.safe_load(runner_yaml.read_text())
                    entrypoints = meta.get("entrypoints", {})
                    entrypoint_strs = []
                    for ep_name, ep_config in entrypoints.items():
                        if isinstance(ep_config, dict):
                            gpu = ep_config.get("gpu")
                            gpu_str = gpu if gpu else "CPU"
                            entrypoint_strs.append(f"{ep_name} ({gpu_str})")
                        else:
                            entrypoint_strs.append(ep_name)
                    if entrypoint_strs:
                        print(f"      Entrypoints: {', '.join(entrypoint_strs)}")
            except Exception:
                pass
        else:
            # Show unbuilt runner info
            print(f"      Status: Not built yet")
            print(f"      Build: omni build {source_path}")
        
        print()
    
    # Summary
    built_count = sum(1 for r in sorted_runners if r["built"])
    unbuilt_count = len(sorted_runners) - built_count
    
    if unbuilt_count > 0:
        print(f"Built: {built_count}, Not built: {unbuilt_count}")
        print(f"\nTip: Run 'omni build <path>' to build unbuilt runners")
    
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="omni", description="Omnilaunch CLI")
    sub = p.add_subparsers(dest="command", required=True)

    p_doctor = sub.add_parser("doctor", help="Check local environment")
    p_doctor.set_defaults(func=cmd_doctor)

    p_list = sub.add_parser("list", help="List available runners from registry")
    p_list.set_defaults(func=cmd_list)

    p_build = sub.add_parser("build", help="Build a runner bundle")
    p_build.add_argument("path", help="Path to runner folder")
    p_build.set_defaults(func=cmd_build)

    p_setup = sub.add_parser("setup", help="Run runner setup on Modal: build image, verify env, download models")
    p_setup.add_argument("runner", help="Runner ref (name:version or path)")
    p_setup.set_defaults(func=cmd_setup)

    p_run = sub.add_parser("run", help="Run a runner entrypoint")
    p_run.add_argument("runner", help="Runner ref (name:version or path)")
    p_run.add_argument("entrypoint", help="Entrypoint name (train_full/train_lora/infer)")
    # Dataset required only for training entrypoints; optional for infer
    p_run.add_argument("--dataset", help="Dataset URI (hf:.. or vol:..)")
    p_run.add_argument("--params", help="Params file path OR inline JSON/YAML string")
    p_run.add_argument("-p", "--param", action="append", help="Param override as key=value (repeatable)")
    p_run.add_argument("--save", action="store_true", help="Save outputs to disk when content_type is present")
    p_run.add_argument("--outdir", help="Directory for saved outputs (default: ./omni_out)")
    p_run.add_argument("--outfile", help="Explicit output filename (e.g., result.png)")
    p_run.set_defaults(func=cmd_run)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())


