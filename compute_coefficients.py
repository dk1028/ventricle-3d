#!/usr/bin/env python3
"""
compute_coefficients.py

Project GPA-aligned STL meshes into PCA shape space to compute latent coefficients.
If meshes have differing vertex counts, nearest-neighbor remapping to the mean shape vertices is applied.

Usage:
  python compute_coefficients.py \
    [--input_dir <path to GPA STL folder>] \
    [--mean <path to mean_shape.npy>] \
    [--modes <path to modes.npy>] \
    [--out_dir <output coefficients root>]

Default paths are set for convenience.
"""

import os
import argparse
import numpy as np
import trimesh
from sklearn.neighbors import NearestNeighbors

# ─── 경로 설정 ──────────────────────────────────────────
DEFAULT_INPUT_DIR = r"C:\Users\AV75950\python\env\shapenet5\procrustes_new1"
DEFAULT_MEAN      = r"C:\Users\AV75950\python\env\shapenet5\ssm_new\mean_shape.npy"
DEFAULT_MODES     = r"C:\Users\AV75950\python\env\shapenet5\ssm_new\modes.npy"
DEFAULT_OUT_DIR   = r"C:\Users\AV75950\Documents\coefficients"

# ─── PCA 투영 함수 ────────────────────────────────────────
def project_to_pca(shape_vec, mean_vec, modes):
    """
    Compute PCA coefficients for a shape vector:
      coeff = modes @ (shape_vec - mean_vec)
    shape_vec, mean_vec: (3K,)
    modes: (M, 3K)
    returns: (M,)
    """
    diff = shape_vec - mean_vec
    return modes.dot(diff)

# ─── 메인 함수 ──────────────────────────────────────────
def main(input_dir, mean_path, modes_path, out_dir):
    # Load PCA basis
    mean_vec = np.load(mean_path)
    modes    = np.load(modes_path)  # shape (M,3K)
    M, D     = modes.shape
    K        = D // 3
    mean_verts = mean_vec.reshape(K, 3)

    # Prepare output directory
    os.makedirs(out_dir, exist_ok=True)

    # Iterate over STL files
    stl_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.stl')])
    for stl_file in stl_files:
        shape_name = os.path.splitext(stl_file)[0]
        mesh_path  = os.path.join(input_dir, stl_file)
        mesh       = trimesh.load_mesh(mesh_path)
        verts      = mesh.vertices  # (Ni,3)

        # If vertex count differs, remap using nearest neighbor to mean shape vertices
        if verts.shape[0] != K:
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(verts)
            _, indices = nbrs.kneighbors(mean_verts)
            mapped_verts = verts[indices.flatten(), :]
        else:
            # Same topology, use directly
            mapped_verts = verts

        # Flatten to vector
        shape_vec = mapped_verts.reshape(-1)

        # Compute coefficients
        coeff = project_to_pca(shape_vec, mean_vec, modes)

        # Save to NPY
        shape_out_dir = os.path.join(out_dir, shape_name)
        os.makedirs(shape_out_dir, exist_ok=True)
        out_path = os.path.join(shape_out_dir, 'coeff.npy')
        np.save(out_path, coeff)
        print(f"[✓] Saved coefficients for {shape_name} at {out_path}")

# ─── 스크립트 실행 ──────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute PCA shape coefficients for GPA-aligned meshes.'
    )
    parser.add_argument(
        '--input_dir', type=str,
        default=DEFAULT_INPUT_DIR,
        help=f'Folder with STL files (default: {DEFAULT_INPUT_DIR})'
    )
    parser.add_argument(
        '--mean', type=str,
        default=DEFAULT_MEAN,
        help=f'Mean shape vector file (default: {DEFAULT_MEAN})'
    )
    parser.add_argument(
        '--modes', type=str,
        default=DEFAULT_MODES,
        help=f'Modes matrix file (default: {DEFAULT_MODES})'
    )
    parser.add_argument(
        '--out_dir', type=str,
        default=DEFAULT_OUT_DIR,
        help=f'Output root for coefficients (default: {DEFAULT_OUT_DIR})'
    )
    args = parser.parse_args()
    main(args.input_dir, args.mean, args.modes, args.out_dir)