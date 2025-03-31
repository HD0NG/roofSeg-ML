
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, BatchNorm1d

# Positional Encoding: [x, y, z, r, theta, phi]
def add_positional_encoding(xyz):
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    r = torch.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-8
    theta = torch.atan2(y, x)
    phi = torch.acos(torch.clamp(z / r, min=-1.0, max=1.0))
    pe = torch.stack([x, y, z, r, theta, phi], dim=-1)
    return pe  # (B, N, 6)

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

        self.sa1_mlp = mlp_block([6, 64, 64, 128])     # input dim = 6 after PE
        self.sa2_mlp = mlp_block([128, 128, 128, emb_dim])

        self.fp2_mlp = mlp_block([emb_dim + 128, 256, 256])
        self.fp1_mlp = mlp_block([256 + 6, 128, emb_dim])

    def forward(self, xyz, return_skips=False):  # xyz: (B, N, 3)
        B, N, _ = xyz.shape

        # Add positional encoding
        pe_xyz = add_positional_encoding(xyz)  # (B, N, 6)
        l0_features = pe_xyz.transpose(1, 2)   # (B, 6, N)

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
class UNetDecoder(nn.Module):
    def __init__(self, emb_dim=128, output_dim=64):
        super(UNetDecoder, self).__init__()
        self.conv1 = nn.Conv1d(emb_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 64, 1)
        self.conv3 = nn.Conv1d(64, output_dim, 1)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(output_dim)

    def forward(self, x):  # x: (B, N, emb_dim)
        x = x.permute(0, 2, 1)            # → (B, emb_dim, N)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return x.permute(0, 2, 1)         # → (B, N, output_dim)

# PointNetPPUNe with PointNet++ Encoder and UNet Decoder
class PointNetPPUNet(nn.Module):
    def __init__(self, emb_dim=128, output_dim=64):
        super().__init__()
        self.encoder = PointNetPPEncoderFP(emb_dim=emb_dim)  # FPS + PE + FP
        self.decoder = UNetDecoder(emb_dim=emb_dim, output_dim=output_dim)

    def forward(self, x):  # x: (B, N, 3)
        features = self.encoder(x)         # (B, N, emb_dim)
        output = self.decoder(features)    # (B, N, output_dim)
        return output


