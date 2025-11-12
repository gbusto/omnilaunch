#!/usr/bin/env python3
"""
Example usage of MeshCoder runner via Modal API.

This script demonstrates how to:
1. Convert a mesh file to point cloud
2. Upload it to Modal volume
3. Run MeshCoder inference
4. Get back Blender Python code
"""

import modal
import numpy as np
import trimesh
from pathlib import Path


def convert_mesh_to_pointcloud(mesh_path: str, num_points: int = 16384):
    """Convert mesh to normalized point cloud."""
    mesh = trimesh.load(mesh_path, force='mesh')

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([
            geom for geom in mesh.geometry.values()
            if isinstance(geom, trimesh.Trimesh)
        ])

    # Sample points
    points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
    normals = mesh.face_normals[face_indices]

    # Normalize to -1 to 1
    center = points.mean(axis=0)
    points_centered = points - center
    max_extent = np.abs(points_centered).max()
    points = points_centered / max_extent

    return points, normals


def run_meshcoder_inference(mesh_file: str, output_code_file: str = None):
    """
    Complete workflow: mesh → point cloud → Blender code.

    Args:
        mesh_file: Path to input mesh (OBJ, GLB, etc.)
        output_code_file: Optional path to save generated Blender code
    """
    print(f"Converting mesh to point cloud: {mesh_file}")
    points, normals = convert_mesh_to_pointcloud(mesh_file)
    print(f"✓ Point cloud created: {points.shape}")

    # Look up the Modal app
    print("Connecting to MeshCoder Modal app...")
    app = modal.App.lookup("omnilaunch-meshcoder", create_if_missing=False)

    # Run inference
    print("Running MeshCoder inference...")
    result = app.infer.remote({
        "points": points.tolist(),
        "normals": normals.tolist(),
        "max_new_tokens": 2048,
        "do_sample": False,
        "zero_out_normals": True
    })

    # Extract results
    data = result.get("data", {})
    blender_code = data.get("code", "")
    success = data.get("success", False)
    num_points = data.get("num_points_used", 0)

    print(f"\n{'='*60}")
    print(f"Inference completed!")
    print(f"  Success: {success}")
    print(f"  Points used: {num_points}")
    print(f"  Code length: {len(blender_code)} characters")
    print(f"{'='*60}\n")

    # Save code if requested
    if output_code_file:
        Path(output_code_file).write_text(blender_code)
        print(f"✓ Blender code saved to: {output_code_file}")
    else:
        print("Generated Blender code:")
        print("-" * 60)
        print(blender_code[:500] + "..." if len(blender_code) > 500 else blender_code)
        print("-" * 60)

    return blender_code, success


# Alternative: Using NPZ file on Modal volume
def run_meshcoder_with_npz(npz_path: str):
    """
    Run inference using NPZ file already uploaded to Modal volume.

    Args:
        npz_path: Path to NPZ on Modal volume (e.g., "/omnilaunch/data/my_pc.npz")
    """
    print(f"Running MeshCoder inference on: {npz_path}")

    app = modal.App.lookup("omnilaunch-meshcoder", create_if_missing=False)

    result = app.infer.remote({
        "npz_path": npz_path,
        "max_new_tokens": 2048,
        "do_sample": False,
        "zero_out_normals": True
    })

    data = result.get("data", {})
    blender_code = data.get("code", "")
    success = data.get("success", False)

    print(f"Success: {success}")
    print(f"Code length: {len(blender_code)} characters")

    return blender_code, success


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python example_usage.py input.obj [output.py]")
        print("\nExample:")
        print("  python example_usage.py chair.obj chair_blender.py")
        sys.exit(1)

    mesh_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        code, success = run_meshcoder_inference(mesh_file, output_file)
        if success:
            print("\n✓ Success! You can now run the generated code in Blender.")
        else:
            print("\n⚠ Code generation completed but may have issues. Check the output.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
