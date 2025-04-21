from models.model_utils import SipUNetDecoder
import torch.nn as nn
import torch
import torch.nn.functional as F

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