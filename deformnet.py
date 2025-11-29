#!/usr/bin/env python3
"""
deformnet.py

Defines DeformNet: a CNN+GCN-based mesh deformation network to reconstruct 3D shapes from 2D features.
Usage:
    from deformnet import DeformNet, build_edge_index
    # Initialize:
    import torch
    from trimesh import load_mesh
    mesh = load_mesh(mean_stl_path)
    faces = mesh.faces
    edge_index = build_edge_index(faces)
    model = DeformNet(feature_dim=2048,
                      hidden_dim=256,
                      num_vertices=num_vertices,
                      edge_index=edge_index)
    delta = model(features, mean_vertices)
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


def build_edge_index(faces):
    """
    Build bidirectional edge_index from face list.
    faces: list, numpy.ndarray or torch.Tensor of shape (F,3)
    returns: edge_index Tensor of shape (2, E)
    """
    # Convert to torch.Tensor
    if isinstance(faces, (list, np.ndarray)):
        faces = torch.tensor(faces, dtype=torch.long)
    elif isinstance(faces, torch.Tensor):
        faces = faces.long()
    else:
        raise TypeError("Faces must be list, numpy.ndarray, or torch.Tensor")

    edges = []
    for tri in faces:
        i, j, k = tri.tolist()
        edges.extend([(i, j), (j, i), (j, k), (k, j), (i, k), (k, i)])
    # Unique edges
    edges = list(set(edges))
    src, dst = zip(*edges)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index


class DeformNet(nn.Module):
    """
    CNN features + GCN-based mesh deformation network.

    Inputs:
      features       Tensor of shape (B, feature_dim)
      mean_vertices  Tensor of shape (B, N, 3)
    Output:
      delta          Tensor of shape (B, N, 3)
    """
    def __init__(self, feature_dim: int, hidden_dim: int,
                 num_vertices: int, edge_index: torch.LongTensor):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_vertices = num_vertices
        # store edge_index
        self.register_buffer('edge_index', edge_index)

        # Encoder: feature_dim → hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # GCN layers
        self.conv1 = GCNConv(hidden_dim + 3, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, 3)

    def forward(self, features: torch.Tensor,
                mean_vertices: torch.Tensor) -> torch.Tensor:
        """
        features: (B, F)
        mean_vertices: (B, N, 3)
        returns delta: (B, N, 3)
        """
        B, N, _ = mean_vertices.shape
        # Encode features
        h = self.encoder(features)           # (B, hidden_dim)
        h = h.unsqueeze(1).expand(-1, N, -1) # (B, N, hidden_dim)
        # Concatenate with vertices
        x = torch.cat([h, mean_vertices], dim=-1)  # (B, N, hidden_dim+3)

        # Batch GCN
        deltas = []
        for b in range(B):
            xb = x[b]  # (N, hidden_dim+3)
            out = F.relu(self.conv1(xb, self.edge_index))
            out = F.relu(self.conv2(out, self.edge_index))
            delta = self.conv3(out, self.edge_index)  # (N,3)
            deltas.append(delta)
        delta = torch.stack(deltas, dim=0)  # (B,N,3)
        return delta
