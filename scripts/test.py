import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.pointnet_unet import PointUNet
from models.pointnetpp_unet import PointNetPPUNet
from datasets.lidar_dataset import LiDARPointCloudDataset
from torch.utils.data import DataLoader
from utils.evaluation import batch_evaluate
from utils.miscellaneous import collate_fn
# Define dataset paths
points_folder = "data/roofNTNU/train_test_split/points_test_n"
labels_folder = "data/roofNTNU/train_test_split/labels_test_n"

# Create dataset instance
dataset = LiDARPointCloudDataset(points_folder, labels_folder, max_points=2048, mode="test")

dataloader = DataLoader(
    dataset,
    batch_size=4,                # ← you still want per-scene evaluation
    shuffle=False,
    num_workers=8,              # Use multiple CPU cores for faster loading
    collate_fn=collate_fn        # ← the one that returns (points, labels, count)
)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
model = PointNetPPUNet(emb_dim=128, output_dim=128).to(device)

# Load trained weights
model.load_state_dict(torch.load("model/PointNetPPUNet_12_n_re.pth", map_location=device))
model.eval()  # Set model to evaluation mode
print("✅ Model loaded successfully!")

batch_evaluate(
    model=model,
    dataloader=dataloader,
    device=device,
    output_dir="test_results/PointNetPPUNet_12_n_re"
)