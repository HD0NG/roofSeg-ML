import torch
import open3d as o3d
import numpy as np

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