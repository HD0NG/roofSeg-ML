import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn import Sequential as Seq, Linear as Lin, BatchNorm1d as BN
from Archieve.AFunctions import *



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

# PointNet Model
class PointNetEncoder(nn.Module):
    def __init__(self, input_dim=3, emb_dim=128):
        super(PointNetEncoder, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, emb_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(emb_dim)

    # def forward(self, x):
    #     x = x.permute(0, 2, 1)  # Convert to (B, C, N)
    #     x = F.relu(self.bn1(self.conv1(x)))
    #     x = F.relu(self.bn2(self.conv2(x)))
    #     x = self.bn3(self.conv3(x))
    #     global_feature = torch.max(x, 2, keepdim=True)[0]
    #     global_feature = global_feature.repeat(1, 1, x.shape[2])
    #     x = torch.cat([x, global_feature], dim=1)
    #     return x.permute(0, 2, 1)  # Back to (B, N, C)
    def forward(self, x, return_skips=False):
        x = x.permute(0, 2, 1)  # (B, C, N)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))  # (B, C, N)

        global_feature = torch.max(x, 2, keepdim=True)[0]  # (B, C, 1)
        global_feature = global_feature.repeat(1, 1, x.shape[2])  # (B, C, N)

        x = torch.cat([x, global_feature], dim=1)  # (B, 2C, N)
        out = x.permute(0, 2, 1)  # (B, N, 2C)

        if return_skips:
            return out, [out]  # dummy skip structure
        return out

# UNet Model
class SipUNetDecoder(nn.Module):
    def __init__(self, emb_dim=128, output_dim=64):
        super(SipUNetDecoder, self).__init__()
        self.conv1 = nn.Conv1d(emb_dim * 2, 128, 1)
        self.conv2 = nn.Conv1d(128, 64, 1)
        self.conv3 = nn.Conv1d(64, output_dim, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Convert to (B, C, N)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return x.permute(0, 2, 1)  # Back to (B, N, C)

# PointNet-UNet Model
class PointUNet(nn.Module):
    def __init__(self, input_dim=3, emb_dim=128, output_dim=64):
        super(PointUNet, self).__init__()
        self.encoder = PointNetEncoder(input_dim, emb_dim)
        self.decoder = SipUNetDecoder(emb_dim, output_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


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