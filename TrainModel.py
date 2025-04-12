# %%
from NetModel import *
from AFunctions import *
import torch.optim as optim
import pickle
import sys

# %%
# Define dataset paths
points_folder = "data/roofNTNU/train_test_split/points_train"
labels_folder = "data/roofNTNU/train_test_split/labels_train"

# Create dataset instance
train_dataset = LiDARPointCloudDataset(points_folder, labels_folder, max_points=2048, mode="train")

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

# %%
model = PointNetPPUNet(emb_dim=128, output_dim=64)
# recon_head = ReconstructionHead(emb_dim=64)

optimizer = optim.Adam(
    model.parameters(), 
    lr=0.001, 
    weight_decay=1e-4)


# loss_history = train_model(model, train_loader, optimizer, num_epochs=100, device='cuda', save_model=True, save_path="model/pointnetpp_unet_4.pth")
loss_history = train_model(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    recon_head=None,
    lambda_recon=0.1,
    num_epochs=50,
    save_model=True,
    save_path="model/pointnetpp_unet_6.pth",
    device='cuda'
)
print("✅ Model trained!")
# Save loss history
with open("model/loss_history_pointnetpp_unet_6.pkl", "wb") as f:
    pickle.dump(loss_history, f)
print("Loss history saved!")

# print("✅ Test complete!")
# add exit code
sys.exit(0)


