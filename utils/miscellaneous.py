import torch
import numpy as np

def normalize_embeddings(embeddings):
    """
    Normalize embeddings to unit L2 norm per point.
    Input: embeddings (N, D)
    Output: normalized (N, D)
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    return embeddings / norms

def chamfer_distance(x, y):
    """
    x: (B, N, 3)
    y: (B, M, 3)
    Returns mean Chamfer distance per batch
    """
    x = x.unsqueeze(2)  # (B, N, 1, 3)
    y = y.unsqueeze(1)  # (B, 1, M, 3)
    dist = torch.norm(x - y, dim=-1)  # (B, N, M)

    min_dist_x, _ = dist.min(dim=2)  # (B, N)
    min_dist_y, _ = dist.min(dim=1)  # (B, M)

    return min_dist_x.mean(dim=1) + min_dist_y.mean(dim=1)  # (B,)

def collate_fn(batch):
    """
    Custom collate function to handle variable-length point clouds.
    Stacks padded point clouds into a batch.
    """
    point_clouds, labels = zip(*batch)

    # Convert to PyTorch tensors
    point_clouds = torch.stack(point_clouds)  # (batch_size, max_points, 3)
    labels = torch.stack(labels)  # (batch_size, max_points)

    return point_clouds, labels