#!/usr/bin/env python3
"""
Convert mesh files (OBJ, GLB, STL, etc.) to NPZ point clouds for MeshCoder.

Usage:
    python convert_mesh_to_npz.py input.obj output.npz
    python convert_mesh_to_npz.py input.glb output.npz --num-points 16384
"""

import argparse
import numpy as np
import trimesh


def mesh_to_pointcloud(mesh_path: str, num_points: int = 16384, normalize: bool = True):
    """Convert mesh to point cloud with normals.

    Args:
        mesh_path: Path to input mesh file (OBJ, GLB, STL, etc.)
        num_points: Number of points to sample (default: 16384)
        normalize: Whether to normalize to -1 to 1 range (default: True)

    Returns:
        Tuple of (points, normals) as numpy arrays of shape (num_points, 3)
    """
    print(f"Loading mesh from: {mesh_path}")
    mesh = trimesh.load(mesh_path, force='mesh')

    if isinstance(mesh, trimesh.Scene):
        # If it's a scene, combine all geometries
        mesh = trimesh.util.concatenate([
            geom for geom in mesh.geometry.values()
            if isinstance(geom, trimesh.Trimesh)
        ])

    print(f"Mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    # Sample points from surface
    print(f"Sampling {num_points} points from mesh surface...")
    points, face_indices = trimesh.sample.sample_surface(mesh, num_points)

    # Get normals for sampled points
    normals = mesh.face_normals[face_indices]

    if normalize:
        # Normalize to -1 to 1 range (as expected by MeshCoder)
        print("Normalizing point cloud to [-1, 1] range...")
        center = points.mean(axis=0)
        points_centered = points - center
        max_extent = np.abs(points_centered).max()
        points = points_centered / max_extent

    print(f"Point cloud created: {points.shape}, range: [{points.min():.3f}, {points.max():.3f}]")

    return points, normals


def main():
    parser = argparse.ArgumentParser(description="Convert mesh to NPZ point cloud for MeshCoder")
    parser.add_argument("input", help="Input mesh file (OBJ, GLB, STL, etc.)")
    parser.add_argument("output", help="Output NPZ file")
    parser.add_argument(
        "--num-points",
        type=int,
        default=16384,
        help="Number of points to sample (default: 16384)"
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Don't normalize points to [-1, 1] range"
    )

    args = parser.parse_args()

    # Convert mesh to point cloud
    points, normals = mesh_to_pointcloud(
        args.input,
        num_points=args.num_points,
        normalize=not args.no_normalize
    )

    # Save as NPZ
    print(f"Saving to: {args.output}")
    np.savez(args.output, points=points, normals=normals)
    print("✓ Done!")

    # Print stats
    print(f"\nPoint cloud stats:")
    print(f"  Points shape: {points.shape}")
    print(f"  Normals shape: {normals.shape}")
    print(f"  Points range: [{points.min():.3f}, {points.max():.3f}]")
    print(f"  Normals range: [{normals.min():.3f}, {normals.max():.3f}]")
    print(f"\nYou can now use this with MeshCoder:")
    print(f'  omnilaunch run meshcoder infer --npz_path="/omnilaunch/data/{args.output}"')


if __name__ == "__main__":
    main()
