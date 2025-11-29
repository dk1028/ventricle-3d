import os
import numpy as np
import trimesh
from sklearn.neighbors import NearestNeighbors

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

# 1. Set paths

# Set to the directory containing normalized STL files
INPUT_DIR  = DATA_DIR / "output_new"
# Set to the directory where GPA results will be saved
OUTPUT_DIR = DATA_DIR / "procrustes_new1"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sorted list of STL files
stl_files = sorted([
    INPUT_DIR / f
    for f in os.listdir(INPUT_DIR)
    if f.lower().endswith(".stl")
])
N = len(stl_files)
if N == 0:
    raise RuntimeError(f"No STL files found in {INPUT_DIR}")

# 2. Set reference mesh and establish correspondence

ref_mesh     = trimesh.load_mesh(stl_files[0])
ref_vertices = ref_mesh.vertices.copy()   # (K, 3)
K            = ref_vertices.shape[0]

# Allocate storage: N shapes, each with K points in 3D
all_shapes = np.zeros((N, K, 3), dtype=np.float64)

for i, path in enumerate(stl_files):
    mesh  = trimesh.load_mesh(path)
    verts = mesh.vertices    # (Mi,3), Mi may differ per mesh

    # 2-1) Center alignment using centroid
    verts = verts - verts.mean(axis=0)

    # 2-2) Establish correspondence between reference vertices and this mesh
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(verts)
    _, indices    = nbrs.kneighbors(ref_vertices)  # (K,1)
    corresponded  = verts[indices.flatten(), :]    # (K,3)

    all_shapes[i] = corresponded

print(f"Loaded {N} shapes, each with {K} points.")

# 3. GPA: iterative Procrustes alignment

def best_fit_transform(A, B):
    """
    Compute the optimal scale, rotation, and translation (s, R, t)
    that best align point set A to point set B in least-squares sense.
    Both A and B are arrays of shape (K, 3).
    """
    # Compute centroids
    mu_A = A.mean(axis=0)
    mu_B = B.mean(axis=0)
    A0   = A - mu_A
    B0   = B - mu_B

    # Compute variance of A for scaling
    var_A = np.sum(A0**2)

    # Compute optimal rotation using SVD
    H = A0.T @ B0
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Correct for reflection if necessary
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute scaling factor
    s = S.sum() / var_A

    # Compute translation vector
    t = mu_B - s * (R @ mu_A)

    return s, R, t

# Initialize mean shape as the simple average
mean_shape = all_shapes.mean(axis=0)

# Perform iterative Procrustes (GPA)
for iteration in range(1, 11):
    aligned = np.zeros_like(all_shapes)
    for i in range(N):
        s, R, t = best_fit_transform(all_shapes[i], mean_shape)
        aligned[i] = (s * (R @ all_shapes[i].T)).T + t

    new_mean = aligned.mean(axis=0)
    diff     = np.linalg.norm(mean_shape - new_mean)
    print(f"Iteration {iteration}: mean-shape diff = {diff:.6f}")

    mean_shape = new_mean
    if diff < 1e-6:
        print("Converged.")
        break

# 4. Save the results
for i, path in enumerate(stl_files):
    mesh     = trimesh.Trimesh(vertices=aligned[i], faces=ref_mesh.faces)
    out_name = os.path.basename(path).replace(".stl", "_gpa.stl")
    mesh.export(OUTPUT_DIR / out_name)
    print(f"Saved GPA-aligned shape: {out_name}")
