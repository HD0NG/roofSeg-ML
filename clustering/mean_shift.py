import torch
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import AgglomerativeClustering
from utils.miscellaneous import normalize_embeddings
from utils.positional_encoding import add_positional_encoding

def inference_on_point_cloud(model, point_cloud, device='cuda', clustering='meanshift', bandwidth=None):
    """
    Args:
        model: trained PointNet++ U-Net model
        point_cloud: (N, 3) numpy array of point cloud
        clustering: 'meanshift' or 'dbscan'
    Returns:
        predicted_labels: (N,) instance labels
        embeddings: (N, D) per-point feature embeddings
    """
    model.eval()
    point_cloud_tensor = torch.tensor(point_cloud, dtype=torch.float32).unsqueeze(0).to(device)  # (1, N, 3)

    with torch.no_grad():
        pe = add_positional_encoding(point_cloud_tensor)  # (1, N, 6)
        embeddings, skip = model.encoder(point_cloud_tensor, return_skips=True)  # (1, N, D)
        embeddings = embeddings.squeeze(0).cpu().numpy()  # → (N, D)
        embeddings = normalize_embeddings(embeddings)

    # Clustering
    if clustering == 'meanshift':
        if not bandwidth:
            bandwidth = estimate_bandwidth(embeddings, quantile=0.1)
            # clustering = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(embeddings)
        try:
            clustering = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(embeddings)
        except ValueError as e:
            print("Falling back to bin_seeding=False due to error:", e)
            clustering = MeanShift(bandwidth=bandwidth, bin_seeding=False).fit(embeddings)
    elif clustering == 'dbscan':
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=2, min_samples=5).fit(embeddings)
    elif clustering == 'agglomerative':
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1.0).fit(embeddings)
    else:
        raise ValueError("Clustering method must be 'meanshift', 'dbscan', or 'agglomerative'.")

    predicted_labels = clustering.labels_  # shape: (N,)
    return predicted_labels, embeddings

def cluster_embeddings(embeddings, bandwidth=None, min_bandwidth=0.25):
    if bandwidth is None:
        bandwidth = estimate_bandwidth(embeddings, quantile=0.1)
        if bandwidth < min_bandwidth:
            print(f"⚠️ Estimated bandwidth {bandwidth:.4f} too small — using fallback: {min_bandwidth}")
            bandwidth = min_bandwidth

    try:
        clustering = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(embeddings)
    except ValueError as e:
        print(f"❌ MeanShift failed (bin_seeding=True): {e}")
        clustering = MeanShift(bandwidth=bandwidth, bin_seeding=False).fit(embeddings)

    return clustering.labels_