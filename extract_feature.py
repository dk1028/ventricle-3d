#!/usr/bin/env python3
"""
extract_features_for_masks.py

Extracts CNN features from silhouette mask images for each rendered shape.
Usage:
  python extract_features_for_masks.py [--rendered_masks <path>] [--features_dir <path>] [--device <device>]

Default directories are set for convenience.
"""

import os
import argparse
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

# ─── Path setup ──────────────────────────────────────────
DEFAULT_RENDERED_MASKS_DIR = DATA_DIR / "rendered_masks"
DEFAULT_FEATURES_DIR       = DATA_DIR / "features"

# ─── Build model ─────────────────────────────────────────
def build_feature_extractor(device):
    backbone = models.resnet50(pretrained=True)
    backbone.fc = nn.Identity()  # Remove final FC layer
    model = backbone.to(device)
    model.eval()
    return model

# ─── Feature extraction ──────────────────────────────────
def extract_feature(image_path, model, transform, device):
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(tensor)
    return feat.squeeze(0).cpu().numpy()

# ─── Main function ───────────────────────────────────────
def main(rendered_masks_dir, features_dir, device_str):
    device = torch.device(device_str)
    extractor = build_feature_extractor(device)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    os.makedirs(features_dir, exist_ok=True)
    for shape_folder in sorted(os.listdir(rendered_masks_dir)):
        shape_path = os.path.join(rendered_masks_dir, shape_folder)
        if not os.path.isdir(shape_path):
            continue
        out_folder = os.path.join(features_dir, shape_folder)
        os.makedirs(out_folder, exist_ok=True)

        for fname in sorted(os.listdir(shape_path)):
            if fname.startswith('mask_') and fname.endswith('.png'):
                img_path = os.path.join(shape_path, fname)
                feat = extract_feature(img_path, extractor, transform, device)
                feat_fname = fname.replace('.png', '.npy').replace('mask_', 'feat_')
                np.save(os.path.join(out_folder, feat_fname), feat)
        print(f"[✓] Extracted features for {shape_folder}")

# ─── Script execution ─────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract CNN features from silhouette masks.'
    )
    parser.add_argument(
        '--rendered_masks', type=str,
        default=DEFAULT_RENDERED_MASKS_DIR,
        help=f'Path to rendered_masks root directory (default: {DEFAULT_RENDERED_MASKS_DIR})'
    )
    parser.add_argument(
        '--features_dir', type=str,
        default=DEFAULT_FEATURES_DIR,
        help=f'Output directory for features (default: {DEFAULT_FEATURES_DIR})'
    )
    parser.add_argument(
        '--device', type=str,
        default='cuda:0' if torch.cuda.is_available() else 'cpu',
        help='Compute device (default: auto-detect)'
    )
    args = parser.parse_args()
    main(args.rendered_masks, args.features_dir, args.device)
