#!/usr/bin/env python3
"""
train_deformnet.py

Training script for DeformNet: reconstruct 3D coronary mesh from 2D mask features.
Loss: Chamfer Distance on vertex positions reconstructed from PCA modes.

Usage:
  python train_deformnet.py \
    --features_dir <features root> \
    --coeffs_dir   <coefficients root> \
    --mean_npy     <mean_shape.npy> \
    --modes_npy    <modes.npy> \
    --mean_stl     <mean_shape.stl> \
    [--epochs 50] [--batch_size 8] [--lr 1e-3] [--device auto]
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from pytorch3d.loss import chamfer_distance
from trimesh import load_mesh

# Import dataset and model
from coronary_dataset import CoronaryDataset
from deformnet import DeformNet, build_edge_index


def collate_fn(batch):
    feats, coeffs = zip(*batch)
    feats = torch.stack(feats, dim=0)
    coeffs = torch.stack(coeffs, dim=0)
    return feats, coeffs


def main(args):
    # Device selection (auto-detect if 'auto')
    if args.device.lower() == 'auto':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load PCA data
    mean_vec = np.load(args.mean_npy)        # (3K,)
    modes    = np.load(args.modes_npy)        # (M,3K)
    M, D     = modes.shape
    K        = D // 3
    mean_verts = torch.from_numpy(mean_vec.reshape(K,3)).float().to(device)

    # Load faces and build edge_index
    mesh_mean = load_mesh(args.mean_stl)
    faces = mesh_mean.faces
    edge_index = build_edge_index(faces).to(device)

    # Dataset
    full_dataset = CoronaryDataset(
        features_dir=args.features_dir,
        coeffs_dir=args.coeffs_dir
    )
    total_samples = len(full_dataset)
    val_size = int(0.1 * total_samples)
    train_size = total_samples - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn)

    # Model
    feat_dim = train_ds[0][0].shape[0]
    model = DeformNet(
        feature_dim=feat_dim,
        hidden_dim=args.hidden_dim,
        num_vertices=K,
        edge_index=edge_index
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        for feats, coeffs in train_loader:
            feats  = feats.to(device)
            coeffs = coeffs.to(device)
            # Reconstruct GT delta from PCA
            diff = torch.matmul(coeffs, torch.from_numpy(modes).float().to(device))  # (B,3K)
            gt_delta = diff.view(-1, K, 3)
            # Prepare mean batch
            mean_batch = mean_verts.unsqueeze(0).expand(feats.size(0), -1, -1)

            # Forward
            pred_delta = model(feats, mean_batch)
            pred_verts = mean_batch + pred_delta
            gt_verts   = mean_batch + gt_delta

            # Losses
            chamf_loss, _ = chamfer_distance(pred_verts, gt_verts)
            l2_loss = F.mse_loss(pred_delta, gt_delta)
            loss = chamf_loss + args.l2_weight * l2_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * feats.size(0)
        scheduler.step()
        avg_train_loss = total_loss / train_size

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for feats, coeffs in val_loader:
                feats  = feats.to(device)
                coeffs = coeffs.to(device)
                diff = torch.matmul(coeffs, torch.from_numpy(modes).float().to(device))
                gt_delta = diff.view(-1, K, 3)
                mean_batch = mean_verts.unsqueeze(0).expand(feats.size(0), -1, -1)
                pred_delta = model(feats, mean_batch)
                pred_verts = mean_batch + pred_delta
                gt_verts   = mean_batch + gt_delta
                chamf_loss, _ = chamfer_distance(pred_verts, gt_verts)
                l2_loss = F.mse_loss(pred_delta, gt_delta)
                loss = chamf_loss + args.l2_weight * l2_loss
                val_loss += loss.item() * feats.size(0)
        avg_val_loss = val_loss / val_size

        print(f"Epoch {epoch}/{args.epochs}  "
              f"Train Loss: {avg_train_loss:.6f}  "
              f"Val Loss: {avg_val_loss:.6f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), args.checkpoint)
            print(f"Best model saved to {args.checkpoint}")

    print("Training complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_dir', type=str,
                        default=r"C:\Users\AV75950\Documents\features")
    parser.add_argument('--coeffs_dir', type=str,
                        default=r"C:\Users\AV75950\Documents\coefficients")
    parser.add_argument('--mean_npy', type=str,
                        default=r"C:\Users\AV75950\python\env\shapenet5\ssm_new\mean_shape.npy")
    parser.add_argument('--modes_npy', type=str,
                        default=r"C:\Users\AV75950\python\env\shapenet5\ssm_new\modes.npy")
    parser.add_argument('--mean_stl', type=str,
                        default=r"C:\Users\AV75950\python\env\shapenet5\ssm_new\mean_shape.stl")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--l2_weight', type=float, default=1.0,
                        help='Weight for L2 delta loss')
    parser.add_argument('--device', type=str, default='auto',
                        help='Compute device: cuda:X, cpu, or auto')
    parser.add_argument('--checkpoint', type=str,
                        default=r"C:\Users\AV75950\Documents\checkpoints\deformnet.pth")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
    main(args)
