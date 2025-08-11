import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import fps, knn_graph
from torch_scatter import scatter_max
from utils.positional_encoding import add_positional_encoding


class SharedMLP(nn.Sequential):
    def __init__(self, in_channels, out_channels_list):
        layers = []
        for out_c in out_channels_list:
            layers.append(nn.Conv2d(in_channels, out_c, kernel_size=1))
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_c
        super().__init__(*layers)


def ball_query(xyz, new_xyz, radius, k=32):
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape

    # Pairwise distance
    dists = torch.cdist(new_xyz, xyz)  # [B, S, N]
    group_idx = dists.argsort(dim=-1)[..., :k]  # [B, S, k]
    group_mask = dists.gather(-1, group_idx) < radius
    return group_idx * group_mask + (~group_mask) * -1  # mask out-of-radius


def group_points(points, idx):
    B, N, C = points.shape
    B, S, k = idx.shape

    device = points.device
    idx = idx.clone()
    idx[idx == -1] = 0  # Dummy idx for masked points

    idx_flat = idx.reshape(B, -1)
    grouped = torch.gather(points, 1, idx_flat.unsqueeze(-1).expand(-1, -1, C))
    grouped = grouped.reshape(B, S, k, C)
    return grouped


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, k, in_channels, mlp_channels):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.k = k
        self.mlp = SharedMLP(in_channels, mlp_channels)

    def forward(self, xyz, features):
        B, N, _ = xyz.shape

        # FPS sampling
        xyz_fl = xyz.transpose(1, 2).contiguous()
        sampled_idx = fps(xyz_fl, ratio=self.npoint / N).long()
        new_xyz = xyz[torch.arange(B)[:, None], sampled_idx]  # [B, S, 3]

        # Ball query & group
        idx = ball_query(xyz, new_xyz, self.radius, self.k)  # [B, S, k]
        grouped_xyz = group_points(xyz, idx) - new_xyz.unsqueeze(2)
        grouped_features = group_points(features, idx) if features is not None else None

        if grouped_features is not None:
            new_features = torch.cat([grouped_xyz, grouped_features], dim=-1)
        else:
            new_features = grouped_xyz

        new_features = new_features.permute(0, 3, 2, 1)  # [B, C, k, S]
        new_features = self.mlp(new_features)
        new_features = torch.max(new_features, 2)[0]  # Max over k → [B, C, S]

        return new_xyz, new_features.permute(0, 2, 1)  # [B, S, 3], [B, S, C]


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channels, mlp_channels):
        super().__init__()
        self.mlp = SharedMLP(in_channels, mlp_channels)

    def forward(self, xyz1, xyz2, features1, features2):
        dists = torch.cdist(xyz1, xyz2) + 1e-10
        idx = dists.argsort(dim=-1)[..., :3]  # 3-NN
        dists = torch.gather(dists, -1, idx)

        norm_dists = dists / torch.sum(1.0 / dists, dim=-1, keepdim=True)
        weights = 1.0 / dists
        weights = weights / weights.sum(dim=-1, keepdim=True)

        interpolated = torch.sum(
            torch.gather(features2, 1, idx.unsqueeze(-1).expand(-1, -1, -1, features2.shape[-1]))
            * weights.unsqueeze(-1), dim=2)

        if features1 is not None:
            new_features = torch.cat([interpolated, features1], dim=-1)
        else:
            new_features = interpolated

        return self.mlp(new_features.transpose(1, 2).unsqueeze(-1)).squeeze(-1).transpose(1, 2)


class PointNetPP(nn.Module):
    def __init__(self, output_dim=128, input_channels=3):
        super().__init__()

        self.sa1 = PointNetSetAbstraction(npoint=1024, radius=0.1, k=32,
                                          in_channels=input_channels + 6, mlp_channels=[64, 64, 128]) # 6 for positional encoding
        self.sa2 = PointNetSetAbstraction(npoint=256, radius=0.2, k=32,
                                          in_channels=128 + 6, mlp_channels=[128, 128, 256])
        self.sa3 = PointNetSetAbstraction(npoint=64, radius=0.4, k=32,
                                          in_channels=256 + 6, mlp_channels=[256, 256, 512])

        self.fp3 = PointNetFeaturePropagation(512 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(256 + 128, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128 + input_channels, [128, 128])

        self.classifier = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, output_dim, 1)
        )

    def forward(self, xyz, features, return_skips=False):
        # Add positional encoding
        pe = add_positional_encoding(xyz)  # (B, N, 6)
        if features is not None:
            features = torch.cat([features, pe], dim=-1)
        else:
            features = pe
        l0_xyz = xyz
        l0_points = features

        # Set abstraction layers
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Feature propagation layers
        fp2 = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        fp1 = self.fp2(l1_xyz, l2_xyz, l1_points, fp2)
        fp0 = self.fp1(l0_xyz, l1_xyz, l0_points, fp1)

        # out = self.classifier(fp0.transpose(1, 2))  # [B, num_classes, N]
        embedding = self.embedding(fp0.transpose(1, 2))  # [B, 128, N]
        embedding = F.normalize(embedding, p=2, dim=1)    # Normalize along feature dim

        if return_skips:
            return embedding, {
                "sa1": l1_points,
                "sa2": l2_points,
                "sa3": l3_points,
                "fp2": fp2,
                "fp1": fp1,
                "fp0": fp0,
            }

        return embedding  # [B, 128, N]