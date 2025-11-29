#!/usr/bin/env python3
r"""
coronary_dataset.py

Defines a PyTorch Dataset for pairing CNN features of 2D masks with PCA coefficients of 3D shapes.
Usage example:
    from coronary_dataset import CoronaryDataset
    dataset = CoronaryDataset(
        features_dir=r"C:\Users\AV75950\Documents\features",
        coeffs_dir=r"C:\Users\AV75950\Documents\coefficients"
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

class CoronaryDataset(Dataset):
    r"""
    Dataset for 2D mask features and 3D PCA coefficients.

    Expects directory structure:
      features_dir/
        <shape_name>/
          feat_00.npy
          feat_01.npy
          ...
      coeffs_dir/
        <coeff_folder>/
          coeff.npy
    coeff_folder should match shape_name or start with it (e.g., shape_name_gpa).
    """
    def __init__(self, features_dir: str, coeffs_dir: str, transform=None):
        super().__init__()
        self.features_dir = features_dir
        self.coeffs_dir = coeffs_dir
        self.transform = transform
        self.samples = []  # list of (feat_path, coeff_path)

        # Build sample index
        for shape_folder in sorted(os.listdir(self.features_dir)):
            feat_folder = os.path.join(self.features_dir, shape_folder)
            if not os.path.isdir(feat_folder):
                continue

            # Find matching coefficient folder
            coeff_candidates = [d for d in os.listdir(self.coeffs_dir)
                                if os.path.isdir(os.path.join(self.coeffs_dir, d))
                                and d.startswith(shape_folder)]
            if not coeff_candidates:
                print(f"[!] No coefficient folder for '{shape_folder}', skipping.")
                continue
            coeff_folder = coeff_candidates[0]
            coeff_path = os.path.join(self.coeffs_dir, coeff_folder, 'coeff.npy')
            if not os.path.exists(coeff_path):
                print(f"[!] Coefficient file not found at '{coeff_path}', skipping.")
                continue

            # Associate each feature with this coefficient
            for fname in sorted(os.listdir(feat_folder)):
                if fname.startswith('feat_') and fname.endswith('.npy'):
                    feat_path = os.path.join(feat_folder, fname)
                    self.samples.append((feat_path, coeff_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat_path, coeff_path = self.samples[idx]
        feat = np.load(feat_path)
        coeff = np.load(coeff_path)

        # Convert to torch.Tensor
        feat_tensor = torch.from_numpy(feat).float()
        coeff_tensor = torch.from_numpy(coeff).float()

        if self.transform is not None:
            feat_tensor = self.transform(feat_tensor)

        return feat_tensor, coeff_tensor

# Example usage
if __name__ == '__main__':
    import torch
    from torch.utils.data import DataLoader

    features_dir = r"C:\Users\AV75950\Documents\features"
    coeffs_dir   = r"C:\Users\AV75950\Documents\coefficients"
    dataset = CoronaryDataset(features_dir, coeffs_dir)
    print(f"Total samples: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    for feats, coeffs in loader:
        print('Batch features:', feats.shape)    # (B, F)
        print('Batch coeffs:',  coeffs.shape)   # (B, M)
        break
