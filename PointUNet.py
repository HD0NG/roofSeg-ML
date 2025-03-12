# import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import open3d as o3d
# import laspy
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from sklearn.cluster import DBSCAN


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

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Convert to (B, C, N)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        global_feature = torch.max(x, 2, keepdim=True)[0]
        global_feature = global_feature.repeat(1, 1, x.shape[2])
        x = torch.cat([x, global_feature], dim=1)
        return x.permute(0, 2, 1)  # Back to (B, N, C)

# UNet Model
class UNetDecoder(nn.Module):
    def __init__(self, emb_dim=128, output_dim=64):
        super(UNetDecoder, self).__init__()
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
        self.decoder = UNetDecoder(emb_dim, output_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

"""
Cluster Contrastive Loss
Ensures same-face points are close, different faces are separated.
"""
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

# Train PointNet-UNet model
def train_model(model, train_loader, optimizer, num_epochs=50, device='cuda', save_model=True, directory="model", filename="pointnet_unet_trained.pth"):
    """
    Trains the PointNet-UNet model and visualizes training loss.

    Args:
        model: PointNet-UNet model.
        train_loader: DataLoader for training data.
        optimizer: Optimizer for training.
        num_epochs: Number of training epochs.
        device: 'cuda' or 'cpu'.
        save_model: Whether to save the trained model after training.

    Returns:
        loss_history: List of loss values per epoch.
    """
    model.to(device)
    model.train()
    loss_history = []

    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for data, labels in progress_bar:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = contrastive_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())  # Show live loss update

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Save model after training
    if save_model:
        torch.save(model.state_dict(), os.join(directory, filename))
        print("Model saved successfully!")

    return loss_history

# Post-Processing with DBSCAN for Instance Labels
def cluster_instances(embeddings, eps=0.3, min_samples=3):
    """
    Cluster points into roof faces using DBSCAN on embedding space.
    """
    embeddings_np = embeddings.detach().cpu().numpy()
    clustered_labels = []

    for batch_emb in embeddings_np:
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(batch_emb)
        clustered_labels.append(clustering.labels_)

    return np.array(clustered_labels)

# Test PointNet-UNet model
def test(model, test_loader, device='cuda'):
    model.eval()
    with torch.no_grad():
        for points, _ in test_loader:
            points = points.to(device)
            embeddings = model(points)  # (B, N, output_dim)
            instance_labels = cluster_instances(embeddings)
            print("Predicted Instance Labels:", instance_labels[0])  # Print for one batch
            break

class LiDARPointCloudDataset(Dataset):
    def __init__(self, point_folder, label_folder, max_points=256, mode="train"):
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

# Dynamic Batching with Collate Function
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

def visualize_lidar_with_plotly(points, labels, scatter_size=1):
    """
    Visualizes a LiDAR point cloud with color-coded segmentation labels using Plotly.

    Args:
        points (numpy array): Shape (N, 3) - XYZ coordinates.
        labels (numpy array): Shape (N,) - Segmentation labels.
    """
    # Convert points and labels into a Pandas DataFrame
    df = pd.DataFrame({
        "X": points[:, 0],
        "Y": points[:, 1],
        "Z": points[:, 2],
        "Label": labels
    })

    # Create 3D scatter plot
    fig = px.scatter_3d(
        df, x="X", y="Y", z="Z",
        color=df["Label"].astype(str),  # Color by label
        title="3D LiDAR Point Cloud Segmentation",
        labels={"Label": "Segmentation Label"},
        opacity=0.7
    )

    # Update marker size
    fig.update_traces(marker=dict(size=scatter_size))

    # Hide XYZ planes, grids, and background
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),  # Hide X-axis
            yaxis=dict(visible=False),  # Hide Y-axis
            zaxis=dict(visible=False),  # Hide Z-axis
            bgcolor="rgba(0,0,0,0)"  # Transparent background
        ),
        margin=dict(l=0, r=0, b=0, t=40),  # Reduce margin for better display
        legend=dict(
            title="Segmentation Label",  # Ensure correct legend title
            x=0.02,  # Move legend closer to the points
            y=0.98
        )
    )

    # Show figure
    fig.show()

# Get a batch from DataLoader
def inference_on_point_cloud(model, point_cloud, device='cuda', eps=0.3, min_samples=3, scatter_size=3):
    """
    Runs inference on a single point cloud and visualizes predictions in real-time.
    
    Args:
        model: Trained Point-U-Net model.
        point_cloud: Input point cloud (numpy array of shape (N, 3)).
        device: 'cuda' or 'cpu' for inference.
        eps: DBSCAN clustering epsilon (distance threshold).
        min_samples: Minimum points in a cluster for DBSCAN.
        scatter_size: Size of scatter points in visualization.
    """
    model.eval()
    point_cloud_tensor = torch.tensor(point_cloud, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        embeddings = model(point_cloud_tensor).squeeze(0).cpu().numpy()  # Run inference

    # Normalize embeddings before DBSCAN
    # embeddings = StandardScaler().fit_transform(embeddings)

    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)
    predicted_labels = clustering.labels_

    # Prepare data for visualization
    df = pd.DataFrame({
        "X": point_cloud[:, 0],
        "Y": point_cloud[:, 1],
        "Z": point_cloud[:, 2],
        "Predicted Instance": predicted_labels.astype(str)  # Convert labels to string for coloring
    })

    # Create 3D scatter plot
    fig = px.scatter_3d(
        df, x="X", y="Y", z="Z",
        color="Predicted Instance",
        title="Real-Time Roof Segmentation",
        opacity=1.0
    )

    # Update marker size for better visibility
    fig.update_traces(marker=dict(size=scatter_size))

    # Hide XYZ planes, grids, and background
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="rgba(0,0,0,0)"
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(
            title="Segmentation Label",
            x=0.2,  # Moves legend closer to the points
            y=0.90,
            font=dict(size=12)  # Adjust legend font size
        )
    )

    fig.show()

def load_clean_point_cloud(file_path, max_points=128):
    """
    Reads a point cloud file, removes non-numeric lines, and ensures exactly 128 points.
    
    Args:
        file_path (str): Path to the .txt file containing point cloud data.
        max_points (int): Fixed number of points per cloud (default: 128).
    
    Returns:
        numpy array: (128, 3) shape point cloud with valid XYZ values.
    """
    valid_points = []

    with open(file_path, "r") as file:
        for line in file:
            try:
                # Convert to float, handling both spaces & commas as delimiters
                values = np.array(line.strip().replace(',', ' ').split(), dtype=np.float64)
                if len(values) == 3:  # Ensure it's an (X, Y, Z) point
                    valid_points.append(values)
            except ValueError:
                continue  # Skip invalid lines

    valid_points = np.array(valid_points)

    num_points = valid_points.shape[0]

    if num_points > max_points:
        # Subsample to 128 points randomly
        indices = np.random.choice(num_points, max_points, replace=False)
        valid_points = valid_points[indices]
    elif num_points < max_points:
        # Pad with zeros to reach 128 points
        pad_size = max_points - num_points
        pad_points = np.zeros((pad_size, 3), dtype=np.float64)
        valid_points = np.vstack((valid_points, pad_points))

    return valid_points

# add if main block
if __name__ == "__main__":
    # Load the model
    model = PointUNet(input_dim=3, emb_dim=128, output_dim=64)
    model.load_state_dict(torch.load("model/pointnet_unet_trained.pth"))