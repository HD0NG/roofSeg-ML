from PointUNet import *
import pickle

# Define dataset paths
points_folder = "data/roofNTNU/train/sample_points"
labels_folder = "data/roofNTNU/train/sample_labels"

# Create dataset instance
train_dataset = LiDARPointCloudDataset(points_folder, labels_folder, max_points=512, mode="train")

# Create DataLoader
# train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

train_loader = DataLoader(
    train_dataset, batch_size=4, shuffle=True,
    num_workers=8,  # Use multiple CPU cores for faster loading
    pin_memory=True,  # Optimizes GPU transfers
    collate_fn=collate_fn
)

# Check batch
for points, labels in train_loader:
    print("Batch Point Cloud Shape:", points.shape)  # Expected: (batch_size, max_points, 3)
    print("Batch Labels Shape:", labels.shape)  # Expected: (batch_size, max_points)
    break

# for points, labels in train_loader:
#     points_np = points[0].cpu().numpy()  # Convert first point cloud in batch to NumPy
#     labels_np = labels[0].cpu().numpy()  # Convert first label set in batch to NumPy
#     visualize_lidar_with_plotly(points_np, labels_np, scatter_size=4)  # Show visualization
#     break  # Only visualize one batch

# Initialize model & optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
print("Using device:", device)
model = PointUNet(input_dim=3, emb_dim=128, output_dim=64).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Train and visualize loss
loss_history = train_model_d(model, train_loader, optimizer, num_epochs=20, device=device, save_model=True, save_path="model/pointUnet_checkpoint_newloss_200s_20e.pth")
print("✅ Model trained!")

#save loss history
with open("model/loss_history.pkl", "wb") as f:
    pickle.dump(loss_history, f)
print("Loss history saved!")


# Plot loss curve
# plt.figure(figsize=(8, 5))
# plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', linestyle='-')
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training Loss Over Epochs")
# plt.grid()
# plt.show()

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
model = PointUNet(input_dim=3, emb_dim=128, output_dim=64).to(device)

# Load trained weights
model.load_state_dict(torch.load("model/pointnet_unet_checkpoint.pth", map_location=device))
model.eval()  # Set model to evaluation mode
print("✅ Model loaded successfully!")

# model.eval()

# Load a test point cloud
# test_point_cloud = load_clean_point_cloud("data/points/94939.txt")  # Assuming XYZ format

# Run real-time inference & visualization
# inference_on_point_cloud(model, test_point_cloud, device, eps=0.18, min_samples=3)

# torch.save(model.state_dict(), "model/pointnet_unet_checkpoint.pth")
print("Model checkpoint saved!")