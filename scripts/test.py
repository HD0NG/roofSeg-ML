import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.pointnet_unet import PointUNet
from models.pointnetpp_unet import PointNetPPUNet
from datasets.lidar_dataset import LiDARPointCloudDataset
from utils.evaluation import batch_evaluate

# Define dataset paths
points_folder = "data/roofNTNU/train_test_split/points_test_n"
labels_folder = "data/roofNTNU/train_test_split/labels_test_n"

# Create dataset instance
dataset = LiDARPointCloudDataset(points_folder, labels_folder, max_points=2048, mode="test")

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
model = PointUNet(emb_dim=128, output_dim=128).to(device)

# Load trained weights
model.load_state_dict(torch.load("model/pointunet_02_n.pth", map_location=device))
model.eval()  # Set model to evaluation mode
print("✅ Model loaded successfully!")

batch_evaluate(
    model=model,
    dataset=dataset,
    device=device,
    output_dir="test_results/ointunet_02_n"
)