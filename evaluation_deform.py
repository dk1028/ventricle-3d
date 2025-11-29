#!/usr/bin/env python3
"""
evaluation_visualization.py

Evaluate and visualize DeformNet reconstructions.

Usage:
  python evaluation_visualization.py \
    --features_dir <features root> \
    --coeffs_dir   <coefficients root> \
    --mean_npy     <mean_shape.npy> \
    --modes_npy    <modes.npy> \
    --mean_stl     <mean_shape.stl> \
    --checkpoint   <model checkpoint> \
    --output_dir   <output results>

Outputs:
  - metrics.csv: Chamfer, IoU, Dice for each test sample
  - visualizations/<sample>/: mesh renderings and silhouette overlays
"""
import os
import numpy as np
import torch
import csv
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from pytorch3d.loss import chamfer_distance
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLOrthographicCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader,
    BlendParams
)
from pytorch3d.structures import Meshes
from trimesh import load_mesh

from coronary_dataset import CoronaryDataset
from deformnet import DeformNet, build_edge_index

# Silhouette renderer

def get_sil_renderer(image_size=256):
    raster_settings = RasterizationSettings(image_size=image_size)
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
    return MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=SoftSilhouetteShader(blend_params=blend_params)
    )

# Metrics: IoU and Dice

def compute_iou_dice(mask_pred, mask_gt):
    mask_p = mask_pred.astype(bool)
    mask_g = mask_gt.astype(bool)
    inter = np.logical_and(mask_p, mask_g).sum()
    uni   = np.logical_or(mask_p, mask_g).sum()
    iou   = inter / uni if uni>0 else 0.0
    dice  = 2*inter / (mask_p.sum() + mask_g.sum()) if (mask_p.sum()+mask_g.sum())>0 else 0.0
    return iou, dice

# Main

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # Load PCA data
    mean_vec = np.load(args.mean_npy)        # (3K,)
    modes    = np.load(args.modes_npy)        # (M,3K)
    M, D     = modes.shape
    K        = D // 3
    mean_verts = torch.from_numpy(mean_vec.reshape(K,3)).float().to(device)

    # Load faces and edge_index
    mesh_mean = load_mesh(args.mean_stl)
    faces_np  = mesh_mean.faces           # (F,3)
    faces_t   = torch.from_numpy(faces_np).long().to(device)
    edge_index = build_edge_index(faces_np).to(device)

    # Dataset and test split
    full_ds = CoronaryDataset(args.features_dir, args.coeffs_dir)
    test_size = int(0.1 * len(full_ds))
    train_size = len(full_ds) - test_size
    _, test_ds = random_split(full_ds, [train_size, test_size])
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # Load model
    feat_dim = full_ds[0][0].shape[0]
    model = DeformNet(feature_dim=feat_dim,
                      hidden_dim=args.hidden_dim,
                      num_vertices=K,
                      edge_index=edge_index).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # Renderer & camera
    sil_renderer = get_sil_renderer(image_size=256)
    R, T = look_at_view_transform(dist=2.5, elev=10, azim=45)
    cameras = OpenGLOrthographicCameras(device=device, R=R, T=T)

    # Prepare CSV
    csv_path = os.path.join(args.output_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['idx','chamfer','iou','dice'])
        writer.writeheader()

        # Iterate test samples
        for idx, (feat, coeff) in enumerate(test_loader):
            feat = feat.to(device)       # (1, F)
            coeff = coeff.to(device)     # (1, M)
            # Reconstruct GT vertices
            diff = torch.matmul(coeff, torch.from_numpy(modes).float().to(device))  # (1, 3K)
            gt_delta = diff.view(1, K, 3)
            mean_batch = mean_verts.unsqueeze(0)  # (1, K, 3)

            # Predict
            with torch.no_grad():
                pred_delta = model(feat, mean_batch)  # (1,K,3)

            # Chamfer
            pred_verts = mean_batch + pred_delta
            gt_verts   = mean_batch + gt_delta
            chamf, _ = chamfer_distance(pred_verts, gt_verts)

                        # Build Meshes for rendering
            mesh_pred = Meshes(verts=[pred_verts[0]], faces=[faces_t])
            mesh_gt   = Meshes(verts=[gt_verts[0]],   faces=[faces_t])

            # Render silhouettes
            sil_pred = sil_renderer(mesh_pred, cameras=cameras)[0,...,3].cpu().numpy() > 0.5
            sil_gt   = sil_renderer(mesh_gt,   cameras=cameras)[0,...,3].cpu().numpy() > 0.5

            # Compute IoU, Dice
            iou, dice = compute_iou_dice(sil_pred, sil_gt)

            # Write CSV
            writer.writerow({'idx': idx, 'chamfer': float(chamf), 'iou': iou, 'dice': dice})

            # Visualization
            sample_dir = os.path.join(args.output_dir, f'sample_{idx:03d}')
            os.makedirs(sample_dir, exist_ok=True)
            fig, axes = plt.subplots(1,2,figsize=(6,3))
            axes[0].imshow(sil_gt, cmap='gray'); axes[0].set_title('GT Silhouette'); axes[0].axis('off')
            axes[1].imshow(sil_pred, cmap='gray'); axes[1].set_title('Pred Silhouette'); axes[1].axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(sample_dir, 'silhouettes.png'))
            plt.close(fig)

    print(f'Metrics and visualizations saved to {args.output_dir}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_dir', type=str, default=r"C:\Users\AV75950\Documents\features")
    parser.add_argument('--coeffs_dir', type=str, default=r"C:\Users\AV75950\Documents\coefficients")
    parser.add_argument('--mean_npy', type=str, default=r"C:\Users\AV75950\python\env\shapenet5\ssm_new\mean_shape.npy")
    parser.add_argument('--modes_npy', type=str, default=r"C:\Users\AV75950\python\env\shapenet5\ssm_new\modes.npy")
    parser.add_argument('--mean_stl', type=str, default=r"C:\Users\AV75950\python\env\shapenet5\ssm_new\mean_shape.stl")
    parser.add_argument('--checkpoint', type=str, default=r"C:\Users\AV75950\Documents\checkpoints\deformnet.pth")
    parser.add_argument('--output_dir', type=str, default=r"C:\Users\AV75950\Documents\evaluation")
    parser.add_argument('--hidden_dim', type=int, default=256)
    args = parser.parse_args()
    main(args)
