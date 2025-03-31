from PointUNet import *
from PointNet2FPUNet import *
import pickle
import sys

def train_model(model, train_loader, optimizer, num_epochs=10, device='cuda', save_model=True, save_path="pointnetpp_unet.pth"):
    model.to(device)
    model.train()
    loss_history = []

    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for points, labels in progress_bar:
            points, labels = points.to(device), labels.to(device)
            optimizer.zero_grad()

            embeddings = model(points)  # (B, N, output_dim)
            # Compute the discriminative loss
            # Reduce γ (reg term), Use δ_v = 0.3 and δ_d = 1.0
            loss = discriminative_loss(embeddings, labels, delta_v=0.3, delta_d=1.0,
                        alpha=1.0, beta=1.0, gamma=0.0001)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"✅ Epoch {epoch + 1}: Loss = {avg_loss:.4f}")

    if save_model:
        torch.save(model.state_dict(), save_path)
        print("💾 Model saved as "+save_path)

    return loss_history

# Define dataset paths
points_folder = "data/roofNTNU/train_test_split/points_train"
labels_folder = "data/roofNTNU/train_test_split/labels_train"

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

model = PointNetPPUNet(emb_dim=128, output_dim=64)

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)

loss_history = train_model(model, train_loader, optimizer, num_epochs=50, device='cuda', save_model=True, save_path="model/pointnetpp_unet.pth")
print("✅ Model trained!")
# Save loss history
with open("model/loss_history_pointnetpp_unet.pkl", "wb") as f:
    pickle.dump(loss_history, f)
print("Loss history saved!")

# print("✅ Test complete!")
# add exit code
sys.exit(0)