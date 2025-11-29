# ventricle-3d

Code for single–view 3D reconstruction of cardiac/coronary anatomy from 2D silhouettes using

* a **statistical shape model (SSM)** built from aligned 3D meshes, and
* a **mesh deformation network (DeformNet)** that predicts shape coefficients from 2D CNN features.

This repository accompanies the manuscript

> **Single-View 3D Reconstruction via Mesh Deformation and Statistical Shape Models**
> *Dowoo Kim et al.*
> Preprint: `paper/3D_Reconstruction_paper.pdf`

The code is organized as a **reproducible pipeline**: from raw STL meshes, through alignment and shape modeling, to 2D rendering, feature extraction, neural network training, and 3D evaluation.

---

## 1. Repository structure

```text
ventricle-3d/
├── README.md
├── requirements.txt             # Python dependencies
├── paper/
│   └── 3D_Reconstruction_paper.pdf
├── data/
│   ├── good_stl/                # raw meshes (user-provided, not committed)
│   ├── output_new/              # roughly normalized meshes (shape_new.py)
│   ├── procrustes_new1/         # GPA-aligned meshes (procrustes_new.py)
│   ├── ssm_new/                 # statistical shape model (ssm.py)
│   ├── rendered_masks/          # PyTorch3D renderings (rendered_masks.py)
│   ├── features/                # CNN features for masks (extract_feature.py)
│   └── coefficients/            # PCA coefficients for meshes (compute_coefficients.py)
├── checkpoints/                 # trained DeformNet weights
│   └── deformnet.pth            # (optional) saved checkpoint
├── evaluation/                  # evaluation outputs (metrics, visualizations)
├── coronary_dataset.py          # PyTorch Dataset for (features, coeffs)
├── deformnet.py                 # DeformNet and mesh graph utilities
├── shape_new.py                 # initial alignment / normalization of STL meshes
├── procrustes_new.py            # generalized Procrustes alignment (GPA)
├── ssm.py                       # statistical shape model (mean shape + modes)
├── compute_coefficients.py      # project meshes onto SSM (PCA coefficients)
├── rendered_masks.py            # render silhouettes/RGB images via PyTorch3D
├── extract_feature.py           # extract CNN features from 2D masks
├── traindeformnet.py            # train DeformNet on (features → coefficients)
└── evaluation_deform.py         # evaluate and visualize reconstructions
```

> **Important.** Raw 3D data (STL meshes) are *not* included in the repository due to data-sharing constraints.
> To reproduce the experiments, you must provide your own meshes under `data/good_stl/` following the conventions below.

---

## 2. Installation

### 2.1. Clone the repository

```bash
git clone https://github.com/dk1028/ventricle-3d.git
cd ventricle-3d
```

### 2.2. Python environment

A minimal dependency set is provided in `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Main dependencies (simplified):

* `numpy`, `scipy`, `scikit-learn`
* `trimesh`
* `torch`, `torchvision`
* `pytorch3d`
* `matplotlib`, `imageio`, `Pillow`

> **Note on PyTorch3D.** Depending on your CUDA / PyTorch versions, installing `pytorch3d` may require following the official instructions. If `pip install pytorch3d` fails, please refer to the PyTorch3D documentation and adjust the command accordingly.

---

## 3. Data layout and assumptions

All scripts assume they are run from the **repository root** and use **paths relative to this folder** via `pathlib.Path`. The core data folders are:

* `data/good_stl/` – input 3D meshes (STL).
* `data/output_new/` – normalized & roughly aligned meshes (output of `shape_new.py`).
* `data/procrustes_new1/` – rigidly aligned meshes from generalized Procrustes analysis (GPA) (`procrustes_new.py`).
* `data/ssm_new/` – statistical shape model: mean shape, modes, eigenvalues, etc. (`ssm.py`).
* `data/rendered_masks/` – rendered binary masks and/or RGB images for different views (`rendered_masks.py`).
* `data/features/` – CNN feature vectors extracted from rendered images (`extract_feature.py`).
* `data/coefficients/` – PCA coefficients (shape parameters) for each mesh (`compute_coefficients.py`).

You must populate `data/good_stl/` with your own meshes before running the pipeline. The scripts expect **consistent vertex ordering** across meshes, or at least a consistent notion of correspondence enforced by the preprocessing pipeline.

---

## 4. Pipeline overview

The pipeline can be run in four conceptual stages:

1. **Shape preprocessing and alignment**
2. **Statistical shape model and coefficients**
3. **Rendering and feature extraction**
4. **DeformNet training and evaluation**

Each stage is described below with example commands.

---

## 5. Stage 1 — Preprocess and normalize meshes

### 5.1. Initial alignment and normalization (`shape_new.py`)

This script performs basic normalization of STL meshes (e.g., centering, scaling, optional ICP with a reference).

**Inputs**

* `data/good_stl/*.stl`

**Outputs**

* `data/output_new/*.stl`

**Usage**

```bash
python shape_new.py
```

The script automatically reads all `.stl` files under `data/good_stl` and writes the normalized versions to `data/output_new`.

---

### 5.2. Generalized Procrustes Analysis (GPA) (`procrustes_new.py`)

This script applies iterative Procrustes alignment across the normalized shapes to remove residual rigid transforms.

**Inputs**

* `data/output_new/*.stl`

**Outputs**

* `data/procrustes_new1/*_gpa.stl` – rigidly aligned versions of the meshes.

**Usage**

```bash
python procrustes_new.py
```

The code computes an average template and iteratively aligns each shape to that template using rigid transformations.

---

## 6. Stage 2 — Statistical Shape Model (SSM)

### 6.1. Build the SSM (`ssm.py`)

This script constructs a linear statistical shape model by applying PCA to the aligned meshes.

**Inputs**

* `data/procrustes_new1/*.stl`

**Outputs** (all in `data/ssm_new/`):

* `mean_shape.npy` – flattened mean shape (concatenated vertex coordinates).
* `modes.npy` – leading PCA modes (each mode is a deformation direction).
* `eigenvalues.npy` – corresponding eigenvalues.
* `mean_shape.stl` – mean mesh as STL.
* Optional: `mode{j}_plus.stl`, `mode{j}_minus.stl` – visualizations of ±k·√λ_j deformations.

**Usage**

```bash
python ssm.py
```

The number of retained modes and other thresholds are controlled within the script and can be adjusted depending on the dataset.

---

### 6.2. Project meshes into the SSM space (`compute_coefficients.py`)

This script computes PCA coefficients (shape parameters) for each aligned mesh.

**Inputs**

* `data/procrustes_new1/*.stl`
* `data/ssm_new/mean_shape.npy`
* `data/ssm_new/modes.npy`

**Outputs**

* `data/coefficients/<shape_id>/coeff.npy`

**Usage**

```bash
python compute_coefficients.py
```

Optional arguments:

```bash
python compute_coefficients.py \
  --input_dir data/procrustes_new1 \
  --mean      data/ssm_new/mean_shape.npy \
  --modes     data/ssm_new/modes.npy \
  --out_dir   data/coefficients
```

Each `coeff.npy` file stores the PCA coefficients for a given shape and is later used as the regression target for DeformNet.

---

## 7. Stage 3 — Rendering and feature extraction

### 7.1. Render silhouettes using PyTorch3D (`rendered_masks.py`)

This script uses PyTorch3D to render binary silhouettes (and optionally RGB images) from multiple viewpoints.

**Inputs**

* `data/output_new/*.stl`

**Outputs** (under `data/rendered_masks/`):

* For each mesh ID, a sub-folder containing:

  * `mask_view{i}.png` – silhouette masks.
  * `rgb_view{i}.png` – RGB renderings (optional).

**Usage**

```bash
python rendered_masks.py
```

Default parameters (number of views, camera distance, etc.) are defined in the script and can be modified.

---

### 7.2. Extract CNN features from masks (`extract_feature.py`)

This script feeds the rendered masks into a CNN (e.g., ResNet-based) to extract feature vectors.

**Inputs**

* `data/rendered_masks/`

**Outputs**

* `data/features/*.npy`

**Usage**

```bash
python extract_feature.py
```

You can customize the backbone network, layer from which features are taken, and pooling strategy inside the script.

---

## 8. Stage 4 — Train and evaluate DeformNet

### 8.1. Data interface (`coronary_dataset.py`)

`CoronaryDataset` provides a PyTorch `Dataset` that pairs CNN features with PCA coefficients.

Typical usage:

```python
from coronary_dataset import CoronaryDataset
from torch.utils.data import DataLoader

train_dataset = CoronaryDataset(
    features_dir="data/features",
    coeffs_dir="data/coefficients"
)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
```

The class assumes a consistent naming between feature files and coefficient files.

---

### 8.2. DeformNet architecture (`deformnet.py`)

`deformnet.py` defines:

* `DeformNet` – a network mapping features → PCA coefficients and/or vertex offsets.
* mesh graph utilities (`build_edge_index`, etc.) for graph-based operations.

You normally do **not** run this file directly; it is imported by `traindeformnet.py` and `evaluation_deform.py`.

---

### 8.3. Training script (`traindeformnet.py`)

This script trains DeformNet to predict PCA coefficients from CNN features.

**Default inputs**

* Features: `data/features/`
* Coefficients: `data/coefficients/`
* Shape model: `data/ssm_new/mean_shape.npy`, `data/ssm_new/modes.npy`, `data/ssm_new/mean_shape.stl`

**Default output**

* Checkpoint: `checkpoints/deformnet.pth`

**Usage**

```bash
python traindeformnet.py
```

Key arguments (with defaults):

```bash
python traindeformnet.py \
  --features_dir data/features \
  --coeffs_dir   data/coefficients \
  --mean_npy     data/ssm_new/mean_shape.npy \
  --modes_npy    data/ssm_new/modes.npy \
  --mean_stl     data/ssm_new/mean_shape.stl \
  --epochs       50 \
  --batch_size   8 \
  --lr           1e-3 \
  --checkpoint   checkpoints/deformnet.pth
```

The loss typically combines:

* L2 loss on PCA coefficients and/or vertex offsets.
* Chamfer distance between predicted and ground-truth meshes.

Details of the loss and architecture are documented in the manuscript and in the comments within `traindeformnet.py`.

---

### 8.4. Evaluation script (`evaluation_deform.py`)

This script evaluates the trained model and optionally visualizes reconstructions.

**Inputs**

* Features: `data/features/`
* Coefficients: `data/coefficients/`
* Shape model: `data/ssm_new/mean_shape.npy`, `data/ssm_new/modes.npy`, `data/ssm_new/mean_shape.stl`
* Checkpoint: `checkpoints/deformnet.pth`

**Outputs** (under `evaluation/`):

* `metrics.csv` – quantitative metrics (e.g., Chamfer distance).
* Optional visualizations (e.g., STL meshes, overlays, plots).

**Usage**

```bash
python evaluation_deform.py
```

You can adjust what is saved and which subset of shapes is evaluated by modifying arguments or the script internals.

---

## 9. Reproducibility notes

* All paths are handled via `pathlib.Path` and are **relative** to the repository root.
* For full reproducibility, you should:

  * Fix random seeds for NumPy and PyTorch (if not already done in the scripts).
  * Record exact package versions (e.g., via `pip freeze > env.txt`).
  * Document your GPU / CPU and CUDA versions when reporting timings.

Because the raw meshes are not distributed, results will numerically depend on your dataset, but the **code path and hyperparameters** should match the manuscript as closely as possible.


