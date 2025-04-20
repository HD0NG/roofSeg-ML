from Archieve.PointUNet import *
from Archieve.PointNet2FPUNet import *
import pickle
import sys
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def normalize_embeddings(embeddings):
    """
    Normalize embeddings to unit L2 norm per point.
    Input: embeddings (N, D)
    Output: normalized (N, D)
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    return embeddings / norms

def train_model(model, train_loader, optimizer, 
                recon_head=None, lambda_recon=0.1,
                num_epochs=10, device='cuda', 
                save_model=True, save_path="pointnetpp_unet.pth"):
    
    model.to(device)
    if recon_head:
        recon_head.to(device)
        recon_head.train()

    model.train()
    loss_history = []

    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for points, labels in progress_bar:
            points, labels = points.to(device), labels.to(device)
            optimizer.zero_grad()

            embeddings = model(points)  # (B, N, emb_dim)

            # === Discriminative Loss ===
            emb_loss = discriminative_loss(
                embeddings, labels,
                delta_v=0.3, delta_d=2.5,
                alpha=1.0, beta=5.0, gamma=0.0001
            )

            # === Optional Reconstruction Loss ===
            if recon_head:
                recon_points = recon_head(embeddings)  # (B, N, 3)
                recon_loss = chamfer_distance(recon_points, points).mean()
                loss = emb_loss + lambda_recon * recon_loss
            else:
                loss = emb_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"✅ Epoch {epoch + 1}: Loss = {avg_loss:.4f}")

        # Optional Debug
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                example_embed = model(points[0:1]).squeeze(0).cpu().numpy()
                example_embed = normalize_embeddings(example_embed)
                variance = np.var(example_embed, axis=0).mean()
                print(f"📊 Embedding variance at epoch {epoch + 1}: {variance:.6f}")
            model.train()

    if save_model:
        torch.save(model.state_dict(), save_path)
        print("💾 Model saved as " + save_path)

    return loss_history


def visualize_clusters_matplotlib(points, labels, title="Instance Segmentation (Matplotlib)", save_path="segmentation_result.png"):
    """
    Visualize clustered point cloud using Matplotlib 3D scatter.
    
    Args:
        points: (N, 3) numpy array
        labels: (N,) numpy array of predicted cluster labels
        title: plot title
        save_path: path to save the figure
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap("tab20", len(unique_labels))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(points[mask, 0], points[mask, 1], points[mask, 2],
                   color=colors(i), label=f"Label {label}", s=4)

    ax.set_title(title)
    ax.set_axis_off()
    ax.view_init(elev=25, azim=120)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0), fontsize='small')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved cluster visualization to: {save_path}")

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
recon_head = ReconstructionHead(emb_dim=64)

optimizer = optim.Adam(
    list(model.parameters()) + list(recon_head.parameters()), 
    lr=1e-4,
    weight_decay=1e-5
)

# loss_history = train_model(model, train_loader, optimizer, num_epochs=100, device='cuda', save_model=True, save_path="model/pointnetpp_unet_4.pth")
loss_history = train_model(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    recon_head=recon_head,
    lambda_recon=0.1,
    num_epochs=100,
    save_model=True,
    save_path="model/pointnetpp_unet_5.pth",
    device='cuda'
)
print("✅ Model trained!")
# Save loss history
with open("model/loss_history_pointnetpp_unet_5.pkl", "wb") as f:
    pickle.dump(loss_history, f)
print("Loss history saved!")

# print("✅ Test complete!")
# add exit code
sys.exit(0)