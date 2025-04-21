import torch.optim as optim
import pickle
import sys
import json
from datetime import datetime
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.miscellaneous import collate_fn, normalize_embeddings, chamfer_distance
from datasets.lidar_dataset import LiDARPointCloudDataset
from models.pointnet_unet import PointUNet
from models.pointnetpp_unet import PointNetPPUNet
from losses.discriminative import discriminative_loss
from losses.contrastive import contrastive_loss, cosine_contrastive_loss
from utils.visualization import visualize_embeddings, compute_centroid_distances


# Define dataset paths
points_folder = "data/roofNTNU/train_test_split/points_train_n"
labels_folder = "data/roofNTNU/train_test_split/labels_train_n"

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

# Initialize model & optimizer
model = PointNetPPUNet(emb_dim=128, output_dim=128)
# model = PointUNet(emb_dim=128, output_dim=128)
# recon_head = ReconstructionHead(emb_dim=64)

optimizer = optim.Adam(
    model.parameters(), 
    lr=0.001, 
    weight_decay=1e-4)

save_path = "model/PointNetPPUNet_12_n_re.pth"
num_epochs = 50

log_data = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "save_model_path": save_path,
    "model_params": {
        "model_type": "PointNetPPUNet",
        "emb_dim": 128,
        "output_dim": 128,
        "num_layers": 4,
        "num_points": train_loader.dataset.max_points,
        "batch_size": train_loader.batch_size,
    },
    "loss_params" : {
        "alpha": 1.0,     
        "beta": 4.0,     
        "gamma": 0.0001,  
        "delta_v": 0.3,   
        "delta_d": 2.0,    
    },
    "training": {
        "epochs": num_epochs,
        "batch_size": train_loader.batch_size,
        "max_points": train_loader.dataset.max_points,
        "optimizer": type(optimizer).__name__,
        "learning_rate": optimizer.param_groups[0]['lr'],
        "reconstruction_head": False,
        "contrastive_loss": False,
        "lambda_recon": 0.1,
        "lambda_count": 0.1,
        "lambda_cos": 0.0,
        "contrastive_margin": 0.5,
    },
    "loss_history": [],
    "embedding_variance": []
}

def train_model(model, train_loader, optimizer, 
                recon_head=None, lambda_recon=0.1,
                contrastive_loss=False, lambda_cos = 0.3,
                lambda_count=0.1,
                num_epochs=10, device='cuda',
                alpha=1.0, beta=3.0, gamma=0.0001,
                delta_v=0.3, delta_d=1.5, 
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

        # for points, labels in progress_bar:
        for points, labels, instance_counts in progress_bar:
            points, labels = points.to(device), labels.to(device)
            optimizer.zero_grad()

            # embeddings = model(points)  # (B, N, emb_dim)
            embeddings, count_pred = model(points)  # new unpack
            # instance_counts = (labels != -1).int().unique(return_counts=True)[1].float()  # actual count per batch

            #  Add a Tiny Gaussian Noise to Embeddings
            embeddings = embeddings + torch.randn_like(embeddings) * 0.01

            # === Discriminative Loss ===
            emb_loss = discriminative_loss(
                embeddings, labels,
                delta_v=delta_v, delta_d=delta_d,
                alpha=alpha, beta=beta, gamma=gamma
            )

            # === Optional Reconstruction Loss ===
            if recon_head:
                recon_points = recon_head(embeddings)  # (B, N, 3)
                recon_loss = chamfer_distance(recon_points, points).mean()
                loss = emb_loss + lambda_recon * recon_loss
            else:
                loss = emb_loss
            
            # === Optional Contrastive Loss ===
            
            if contrastive_loss:
                contrastive = cosine_contrastive_loss(embeddings, labels, margin=0.5)
                if epoch < 5:
                    loss = lambda_cos * contrastive
                else:
                    loss = emb_loss + lambda_cos * contrastive

            # === Optional Count Loss ===
            mse_loss = F.mse_loss(count_pred, instance_counts.to(device))
            loss = emb_loss + lambda_count * mse_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        log_data["loss_history"].append(avg_loss)
        print(f"✅ Epoch {epoch + 1}: Loss = {avg_loss:.4f}")

        # Optional Debug
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                # Run one example (first in batch)
                # example_embed = model(points[0:1]).squeeze(0).cpu().numpy()  # [N, D]
                example_embed, _ = model(points[0:1])  # Ignore count
                example_embed = example_embed.squeeze(0).cpu().numpy()
                example_embed = normalize_embeddings(example_embed)
                example_labels = labels[0].cpu().numpy()  # [N]

                # --- Save Embedding Variance ---
                variance = float(np.var(example_embed, axis=0).mean())
                print(f"📊 Embedding variance at epoch {epoch + 1}: {variance:.6f}")

                # --- Save Visualizations ---
                vis_dir = f"model/debug/epoch_{epoch + 1}"
                os.makedirs(vis_dir, exist_ok=True)

                visualize_embeddings(example_embed, example_labels, f"{vis_dir}/pca_gt.png", title="GT", method="pca")
                visualize_embeddings(example_embed, example_labels, f"{vis_dir}/tsne_gt.png", title="GT", method="tsne")

                # --- Centroid Stats ---
                centroid_stats = compute_centroid_distances(example_embed, example_labels)
                print(f"📐 Centroid distances: min={centroid_stats['min']:.3f}, mean={centroid_stats['mean']:.3f}, max={centroid_stats['max']:.3f}")

                # --- Log All to JSON ---
                log_data["embedding_variance"].append({
                    "epoch": epoch + 1,
                    "variance": variance,
                    "centroid_distance": centroid_stats
                })

            model.train()

    if save_model:
        torch.save(model.state_dict(), save_path)
        print("💾 Model saved as " + save_path)

    return loss_history

# loss_history = train_model(model, train_loader, optimizer, num_epochs=100, device='cuda', save_model=True, save_path="model/pointnetpp_unet_4.pth")
loss_history = train_model(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    recon_head=None,
    lambda_recon=0.1,
    contrastive_loss=False,
    lambda_cos=0.0,
    alpha= 1.0,
    beta=4.0,
    gamma=0.0001,
    delta_v=0.3,
    delta_d=2.0,
    num_epochs=num_epochs,
    save_model=True,
    save_path=save_path,
    device='cuda'
)
print("✅ Model trained!")
# Save loss history
loss_history_path = save_path.replace(".pth", "_loss_history.pkl")
with open(loss_history_path, "wb") as f:
    pickle.dump(loss_history, f)
print("Loss history saved!")

log_path = save_path.replace(".pth", "_train_log.json")
# log_path = os.path.join(os.path.dirname(save_path), "train_log.json")
with open(log_path, "w") as f:
    json.dump(log_data, f, indent=4)
print(f"📒 Training log saved to {log_path}")

# print("✅ Test complete!")
# add exit code
sys.exit(0)