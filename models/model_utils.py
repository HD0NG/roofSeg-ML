import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch.nn import Conv1d
from torch.nn import Dropout

# MLP block
def mlp_block(channels):
    layers = []
    for i in range(len(channels) - 1):
        layers.append(nn.Conv1d(channels[i], channels[i+1], 1))
        layers.append(nn.BatchNorm1d(channels[i+1]))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)

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
