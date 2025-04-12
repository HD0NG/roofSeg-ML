import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn import Sequential as Seq, Linear as Lin, BatchNorm1d as BN
from AFunctions import *

class LiDARPointCloudDataset(Dataset):
    def __init__(self, point_folder, label_folder, max_points=512, mode="train"):
        """
        Args:
            point_folder (str): Path to the folder containing point cloud .txt files.
            label_folder (str): Path to the folder containing corresponding label .txt files.
            max_points (int): Maximum number of points per cloud (for subsampling or padding).
            mode (str): "train" or "test".
        """
        self.point_folder = point_folder
        self.label_folder = label_folder
        self.max_points = max_points
        self.mode = mode  # 'train' or 'test'

        # List all available point files
        self.point_files = sorted([f for f in os.listdir(point_folder) if f.endswith(".txt")])
        self.label_files = sorted([f for f in os.listdir(label_folder) if f.endswith(".txt")])

        # Ensure matching point and label files
        assert len(self.point_files) == len(self.label_files), "Mismatch in points and labels count."

    def load_txt_file(self, file_path, num_features=3):
        """
        Loads a .txt file, converting string lines into a NumPy array of type float64.
        Assumes space/comma-separated values.
        """
        data = []
        with open(file_path, "r") as file:
            for line in file:
                try:
                    values = np.array(line.strip().replace(',', ' ').split(), dtype=np.float64)
                    if len(values) == num_features:
                        data.append(values)
                except ValueError:
                    continue  # Skip lines that cannot be converted

        return np.array(data)

    def load_point_cloud(self, file_path):
        """Loads point cloud (XYZ) from a .txt file and returns a NumPy array."""
        return self.load_txt_file(file_path, num_features=3)  # Expecting [x, y, z]

    def load_labels(self, file_path, num_points):
        """Loads labels from a .txt file, ensuring it matches the number of points."""
        labels = self.load_txt_file(file_path, num_features=1).flatten()
        if len(labels) != num_points:
            print(f"Warning: {file_path} has {len(labels)} labels, expected {num_points}. Using zero-padding.")
            labels = np.pad(labels, (0, max(0, num_points - len(labels))), 'constant', constant_values=0)
        return labels

    def pad_or_subsample(self, points, labels):
        """Ensures a fixed number of points per cloud using padding or subsampling."""
        num_points = points.shape[0]

        if num_points > self.max_points:
            # Randomly sample points
            indices = np.random.choice(num_points, self.max_points, replace=False)
            points, labels = points[indices], labels[indices]
        elif num_points < self.max_points:
            # Pad with zeros
            pad_size = self.max_points - num_points
            pad_points = np.zeros((pad_size, 3), dtype=np.float64)
            pad_labels = np.zeros(pad_size, dtype=np.int64)
            points = np.vstack((points, pad_points))
            labels = np.hstack((labels, pad_labels))

        return points, labels

    def __len__(self):
        return len(self.point_files)

    def __getitem__(self, idx):
        point_path = os.path.join(self.point_folder, self.point_files[idx])
        label_path = os.path.join(self.label_folder, self.label_files[idx])

        point_cloud = self.load_point_cloud(point_path)
        labels = self.load_labels(label_path, num_points=point_cloud.shape[0])

        # Apply padding or subsampling
        point_cloud, labels = self.pad_or_subsample(point_cloud, labels)

        return torch.tensor(point_cloud, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


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