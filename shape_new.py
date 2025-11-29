import os
import numpy as np
import trimesh
from trimesh.registration import icp  

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

# make folder if not exist for save
input_dir = DATA_DIR / "good_stl"
output_dir = DATA_DIR / "output_new"
os.makedirs(output_dir, exist_ok=True)

# get stl mesh file from input folder
stl_files = [f for f in os.listdir(input_dir) if f.endswith(".stl")]

# prepare reference mesh
ref_filepath = input_dir / stl_files[0]
ref_mesh = trimesh.load_mesh(ref_filepath)
ref_vertices = ref_mesh.vertices.copy()
# adapt the reference centroid
ref_centroid = ref_vertices.mean(axis=0)
ref_centered = ref_vertices - ref_centroid
# calculate the range of z axis of the reference shape
ref_bbox_min = ref_centered.min(axis=0)
ref_bbox_max = ref_centered.max(axis=0)
ref_z_range = ref_bbox_max[2] - ref_bbox_min[2]

for filename in stl_files:
    filepath = os.path.join(input_dir, filename)
    mesh = trimesh.load_mesh(filepath)
    if not isinstance(mesh, trimesh.Trimesh):
        print(f"Skipping {filename}: not a valid mesh.")
        continue

    vertices = mesh.vertices.copy()  # shape is (K, 3) for K point

    #1. Move center to zero (centroid)
    # we find average point to be center
    centroid = vertices.mean(axis=0)
    # move all point so center is origin (0,0,0)
    verts_centered = vertices - centroid

    #2. Rotate to same axis direction (adapt to the angle)
    # returns 4×4 transformation T, and transformed points
    T, verts_icp, _ = icp(verts_centered, ref_centered,
                          threshold=1e-05, max_iterations=50)
    # apply only the rotation+translation (we've already centered meshes so translation ≈ 0)
    verts_rotated = verts_icp

    #3. Make all shape same size (Normalization process)
    # consider only the ratio of the z axis
    bbox_min = verts_rotated.min(axis=0)
    bbox_max = verts_rotated.max(axis=0)
    cur_z_range = bbox_max[2] - bbox_min[2]
    scale_z = ref_z_range / cur_z_range if cur_z_range != 0 else 1.0
    verts_normalized = verts_rotated.copy()
    verts_normalized[:, 2] *= scale_z

    # save new mesh with new point but same face
    new_mesh = trimesh.Trimesh(vertices=verts_normalized, faces=mesh.faces)
    save_path = output_dir / filename
    new_mesh.export(save_path)
    print(f"{filename} → aligned & normalized saved to {save_path}")
