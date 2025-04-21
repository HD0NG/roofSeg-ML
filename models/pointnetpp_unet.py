from models.model_utils import mlp_block, UNetDecoder
from utils.positional_encoding import add_positional_encoding_with_normals, farthest_point_sampling
import torch.nn as nn
import torch
import torch.nn.functional as F

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
 
 # PointNetPPUNe with PointNet++ Encoder and UNet Decoder
class PointNetPPUNet(nn.Module):
    def __init__(self, emb_dim=128, output_dim=64):
        super().__init__()
        self.encoder = PointNetPPEncoderFP(emb_dim=emb_dim)  # FPS + PE + FP
        self.decoder = UNetDecoder(emb_dim=emb_dim, output_dim=output_dim, dropout_rate=0.4)  # UNet Decoder

        # Instance count prediction head
        self.count_head = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):  # x: (B, N, 3)
        features = self.encoder(x)         # (B, N, emb_dim)
        embeddings = self.decoder(features)    # (B, N, output_dim)

        # Predict count from pooled embeddings
        pooled = embeddings.mean(dim=1)        # (B, output_dim)
        count_pred = self.count_head(pooled).squeeze(-1)  # (B,)

        return embeddings, count_pred
        # return output