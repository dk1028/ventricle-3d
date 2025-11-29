import os
import numpy as np
import trimesh
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

# 1) make input folder and where to save result


# folder with input STL after align with GPA
INPUT_DIR = DATA_DIR / "procrustes_new1"
# folder to put result of shape model
OUTPUT_DIR = DATA_DIR / "ssm_new"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 2) load one mesh to know how many point (K) and which face connect


stl_files = sorted([
    INPUT_DIR / f
    for f in os.listdir(INPUT_DIR)
    if f.lower().endswith(".stl")
])
N = len(stl_files)
if N == 0:
    raise RuntimeError(f"No STL files found in {INPUT_DIR}")

ref_mesh = trimesh.load_mesh(stl_files[0])
ref_vertices = ref_mesh.vertices  # shape is (K, 3)
faces        = ref_mesh.faces
K            = ref_vertices.shape[0]
print(f"Reference mesh has K = {K} vertices.")


# 3) load all mesh and put inside one big 3D array
shapes_3d = np.zeros((N, K, 3), dtype=np.float64)
for i, path in enumerate(stl_files):
    mesh  = trimesh.load_mesh(path)
    verts = mesh.vertices         # shape maybe (Mi, 3), Mi not always K

    # 3-1) move shape to center so origin is (0,0,0)
    verts = verts - verts.mean(axis=0)

    # 3-2) if shape have different number of point, fix by nearest point match
    if verts.shape[0] != K:
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nbrs.fit(verts)
        _, idx = nbrs.kneighbors(ref_vertices)
        verts = verts[idx.flatten(), :]  # now size is (K, 3)
        print(f"{os.path.basename(path)}: remapped {mesh.vertices.shape[0]} -> {verts.shape[0]} vertices.")

    shapes_3d[i] = verts

print(f"All shapes loaded into array of shape {shapes_3d.shape}.")


# 4) make all shape flat so it become (N, 3K) for PCA
shapes_2d = shapes_3d.reshape(N, 3 * K)


# 5) do PCA to get average shape and shape change direction
pca = PCA()
pca.fit(shapes_2d)

mean_shape_vec   = pca.mean_                # average shape like 1 long line (3K,)
modes            = pca.components_          # how shape move direction (N, 3K)
eigenvalues      = pca.explained_variance_  # how big each mode
cum_var_ratio    = np.cumsum(pca.explained_variance_ratio_)
M = np.searchsorted(cum_var_ratio, 0.90) + 1  # how many mode need for 90% variation
print(f"Selected M = {M} modes to reach 90% variance.")


# 6) save result: number array and STL file

# save numbers
np.save(OUTPUT_DIR / "mean_shape.npy", mean_shape_vec)
np.save(OUTPUT_DIR / "modes.npy",        modes[:M])
np.save(OUTPUT_DIR / "eigenvalues.npy", eigenvalues[:M])

# save average shape as STL file
mean_verts = mean_shape_vec.reshape(K, 3)
mean_mesh  = trimesh.Trimesh(vertices=mean_verts, faces=faces)
mean_mesh.export(OUTPUT_DIR / "mean_shape.stl")

# save each mode as STL file, show how it change in + and - 3 sigma
for j in range(M):
    phi   = modes[j]
    lam   = eigenvalues[j]
    scale = 3 * np.sqrt(lam)
    for sign, tag in [(1, "plus"), (-1, "minus")]:
        vec    = mean_shape_vec + sign * scale * phi
        vertsj = vec.reshape(K, 3)
        meshj  = trimesh.Trimesh(vertices=vertsj, faces=faces)
        fname  = f"mode{j+1}_{tag}.stl"
        meshj.export(OUTPUT_DIR / fname)

print(f"PCA results saved to {OUTPUT_DIR}")
