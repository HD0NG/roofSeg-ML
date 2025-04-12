import torch
import open3d as o3d
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

# Positional Encoding: [x, y, z, r, theta, phi]
def add_positional_encoding(xyz):
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    r = torch.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-8
    theta = torch.atan2(y, x)
    phi = torch.acos(torch.clamp(z / r, min=-1.0, max=1.0))
    pe = torch.stack([x, y, z, r, theta, phi], dim=-1)
    return pe  # (B, N, 6)

def compute_normals(points_np, k=16):
    """
    Estimate normals for a point cloud using Open3D.
    Input: points_np (N, 3)
    Output: normals_np (N, 3)
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    pcd.normalize_normals()
    return np.asarray(pcd.normals)

def add_positional_encoding_with_normals(xyz):
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    r = torch.sqrt(x**2 + y**2 + z**2) + 1e-8
    theta = torch.atan2(y, x)
    phi = torch.acos(torch.clamp(z / r, min=-1.0, max=1.0))
    pe = torch.stack([x, y, z, r, theta, phi], dim=-1)

    # Convert to numpy for normal estimation
    xyz_np = xyz.detach().cpu().numpy()  # (B, N, 3)
    B = xyz_np.shape[0]
    normal_list = []

    for b in range(B):
        normals_np = compute_normals(xyz_np[b])  # (N, 3)
        normal_list.append(torch.tensor(normals_np, dtype=xyz.dtype, device=xyz.device))

    normals = torch.stack(normal_list, dim=0)  # (B, N, 3)

    full = torch.cat([pe, normals], dim=-1)  # (B, N, 9)
    return full


# FPS utility
def farthest_point_sampling(xyz, npoint):
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=2)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=1)[1]
    return centroids

# Contrastive Loss
def contrastive_loss(embeddings, labels, margin=1.0):
    """
    Contrastive loss to cluster points of the same roof face and separate different faces.
    """
    batch_size, num_points, emb_dim = embeddings.shape
    loss = 0.0
    count = 0

    for i in range(batch_size):
        for j in range(num_points):
            for k in range(num_points):
                if j == k:
                    continue
                d = torch.norm(embeddings[i, j] - embeddings[i, k], p=2)
                if labels[i, j] == labels[i, k]:  # Same face
                    loss += d ** 2
                else:  # Different faces
                    loss += max(0, margin - d) ** 2
                count += 1

    return loss / count

# Discriminative Loss
def discriminative_loss(embeddings, instance_labels, delta_v=0.5, delta_d=1.5,
                        alpha=1.0, beta=1.0, gamma=0.001):
    """
    Computes the discriminative loss for instance segmentation.

    Args:
        embeddings (Tensor): (B, N, D) - pointwise embeddings
        instance_labels (Tensor): (B, N) - instance IDs per point
        delta_v (float): margin for close loss
        delta_d (float): margin for apart loss
        alpha, beta, gamma (float): weights for each term

    Returns:
        loss (Tensor): scalar loss value
    """
    batch_size, num_points, emb_dim = embeddings.size()
    total_loss = 0.0

    for b in range(batch_size):
        embedding = embeddings[b]              # (N, D)
        labels = instance_labels[b]            # (N,)
        unique_labels = torch.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]  # ignore noise labels

        K = len(unique_labels)
        if K == 0:
            continue

        cluster_means = []
        l_close = 0.0
        for label in unique_labels:
            mask = labels == label
            if mask.sum() == 0:
                continue

            cluster = embedding[mask]  # Points of one instance
            mean = cluster.mean(dim=0)
            cluster_means.append(mean)

            # ℓ_close: Pull to mean if outside margin δ_v
            dist = torch.norm(cluster - mean, dim=1) - delta_v
            l_close += torch.mean(torch.clamp(dist, min=0.0) ** 2)

        l_close /= K
        cluster_means = torch.stack(cluster_means)  # (K, D)

        # ℓ_apart: Push means away if closer than margin δ_d
        l_apart = 0.0
        for i in range(K):
            for j in range(i + 1, K):
                dist = torch.norm(cluster_means[i] - cluster_means[j], p=1)
                l_apart += F.relu(2 * delta_d - dist) ** 2
        if K > 1:
            l_apart /= (K * (K - 1))

        # ℓ_reg: Pull all means toward origin (L1 norm)
        l_reg = torch.mean(torch.norm(cluster_means, p=1, dim=1))

        # Combine all
        loss = alpha * l_close + beta * l_apart + gamma * l_reg
        total_loss += loss

    return total_loss / batch_size

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
