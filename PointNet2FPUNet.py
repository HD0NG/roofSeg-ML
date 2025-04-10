
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, BatchNorm1d
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

# MLP block
def mlp_block(channels):
    layers = []
    for i in range(len(channels) - 1):
        layers.append(nn.Conv1d(channels[i], channels[i+1], 1))
        layers.append(nn.BatchNorm1d(channels[i+1]))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)

# PointNet++ Encoder with SA + FP, PE + FPS
class PointNetPPEncoderFP(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        self.sa1_npoint = 512
        self.sa2_npoint = 128

        self.sa1_mlp = mlp_block([9, 64, 64, 128])     # input dim = 9 after full PE
        self.sa2_mlp = mlp_block([128, 128, 128, emb_dim])

        self.fp2_mlp = mlp_block([emb_dim + 128, 256, 256])
        self.fp1_mlp = mlp_block([256 + 9, 128, emb_dim])

    def forward(self, xyz, return_skips=False):  # xyz: (B, N, 3)
        B, N, _ = xyz.shape

        # Add positional encoding
        pe_xyz = add_positional_encoding_with_normals(xyz)  # (B, N, 9)
        l0_features = pe_xyz.transpose(1, 2)   # (B, 9, N)

        # -------- SA 1 --------
        fps_idx1 = farthest_point_sampling(xyz, self.sa1_npoint)
        l1_xyz = torch.stack([xyz[b].index_select(0, fps_idx1[b]) for b in range(B)], dim=0)  # (B, 512, 3)
        l1_points = torch.stack([l0_features[b].transpose(0, 1).index_select(0, fps_idx1[b]) for b in range(B)], dim=0).transpose(1, 2)
        l1_features = self.sa1_mlp(l1_points)  # (B, 128, 512)

        # -------- SA 2 --------
        fps_idx2 = farthest_point_sampling(l1_xyz, self.sa2_npoint)
        l2_xyz = torch.stack([l1_xyz[b].index_select(0, fps_idx2[b]) for b in range(B)], dim=0)  # (B, 128, 3)
        l2_points = torch.stack([l1_features[b].transpose(0, 1).index_select(0, fps_idx2[b]) for b in range(B)], dim=0).transpose(1, 2)
        l2_features = self.sa2_mlp(l2_points)  # (B, emb_dim, 128)

        # -------- FP 2 (up: 128 → 512) --------
        l2_up = F.interpolate(l2_features, size=l1_features.shape[2], mode='nearest')
        l1_fp = self.fp2_mlp(torch.cat([l1_features, l2_up], dim=1))  # (B, 256, 512)

        # -------- FP 1 (up: 512 → N) --------
        l1_up = F.interpolate(l1_fp, size=N, mode='nearest')
        l0_fp = self.fp1_mlp(torch.cat([l1_up, l0_features], dim=1))  # (B, emb_dim, N)

        # return l0_fp.permute(0, 2, 1)  # (B, N, emb_dim)
        if return_skips:
            return l0_fp.permute(0, 2, 1), l1_fp.permute(0, 2, 1)  # (B, N, emb_dim), (B, N, 256)
        else:
            return l0_fp.permute(0, 2, 1)  # (B, N, emb_dim)
    
# UNet Decoder
# class UNetDecoder(nn.Module):
#     def __init__(self, emb_dim=128, output_dim=64):
#         super(UNetDecoder, self).__init__()
#         self.conv1 = nn.Conv1d(emb_dim * 2, 128, 1)
#         self.conv2 = nn.Conv1d(128, 64, 1)
#         self.conv3 = nn.Conv1d(64, output_dim, 1)
#         self.bn1 = nn.BatchNorm1d(128)
#         self.bn2 = nn.BatchNorm1d(64)
#         self.bn3 = nn.BatchNorm1d(output_dim)

#     def forward(self, x):
#         x = x.permute(0, 2, 1)  # Convert to (B, C, N)
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = self.bn3(self.conv3(x))
#         return x.permute(0, 2, 1)  # Back to (B, N, C)
# class UNetDecoder(nn.Module):
#     def __init__(self, emb_dim=128, output_dim=64):
#         super(UNetDecoder, self).__init__()
#         self.conv1 = nn.Conv1d(emb_dim, 128, 1)
#         self.conv2 = nn.Conv1d(128, 64, 1)
#         self.conv3 = nn.Conv1d(64, output_dim, 1)

#         self.bn1 = nn.BatchNorm1d(128)
#         self.bn2 = nn.BatchNorm1d(64)
#         self.bn3 = nn.BatchNorm1d(output_dim)

#     def forward(self, x):  # x: (B, N, emb_dim)
#         x = x.permute(0, 2, 1)            # → (B, emb_dim, N)
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = self.bn3(self.conv3(x))
#         return x.permute(0, 2, 1)         # → (B, N, output_dim)

# UNet Decoder
class UNetDecoder(nn.Module):
    def __init__(self, emb_dim=128, output_dim=64, dropout_rate=0.4):
        super(UNetDecoder, self).__init__()
        self.conv1 = nn.Conv1d(emb_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 64, 1)
        self.conv3 = nn.Conv1d(64, output_dim, 1)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(output_dim)

        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, x):  # x: (B, N, emb_dim)
        x = x.permute(0, 2, 1)                      # (B, emb_dim, N)
        x = self.dropout1(F.relu(self.bn1(self.conv1(x))))  # (B, 128, N)
        x = self.dropout2(F.relu(self.bn2(self.conv2(x))))  # (B, 64, N)
        x = self.bn3(self.conv3(x))                         # (B, output_dim, N)
        x = F.normalize(x, p=2, dim=1)                      # L2 normalize per point
        return x.permute(0, 2, 1)                           # (B, N, output_dim)



# PointNetPPUNe with PointNet++ Encoder and UNet Decoder
class PointNetPPUNet(nn.Module):
    def __init__(self, emb_dim=128, output_dim=64):
        super().__init__()
        self.encoder = PointNetPPEncoderFP(emb_dim=emb_dim)  # FPS + PE + FP
        self.decoder = UNetDecoder(emb_dim=emb_dim, output_dim=output_dim, dropout_rate=0.4)  # UNet Decoder

    def forward(self, x):  # x: (B, N, 3)
        features = self.encoder(x)         # (B, N, emb_dim)
        output = self.decoder(features)    # (B, N, output_dim)
        return output


class ReconstructionHead(nn.Module):
    def __init__(self, emb_dim):
        super(ReconstructionHead, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv1d(emb_dim, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)  # Output (x, y, z)
        )

    def forward(self, x):  # x: (B, N, emb_dim)
        x = x.permute(0, 2, 1)
        out = self.decoder(x)  # (B, 3, N)
        return out.permute(0, 2, 1)  # (B, N, 3)

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