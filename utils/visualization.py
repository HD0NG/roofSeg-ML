import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
from scipy.spatial.distance import pdist, squareform

def load_clean_point_cloud(file_path, max_points=512):
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

def visualize_embeddings(embeddings, labels=None, method='pca', title=None, save_path=None):
    """
    Visualize 2D projection of embeddings via PCA or t-SNE.
    """
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30)
    else:
        raise ValueError("Method must be 'pca' or 'tsne'.")
    
    if title is None:
        title = f'Embedding Projection ({method.upper()})'

    projected = reducer.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    if labels is not None:
        plt.scatter(projected[:, 0], projected[:, 1], c=labels, cmap='jet', s=3)
    else:
        plt.scatter(projected[:, 0], projected[:, 1], s=3)
    plt.title(f'Embedding Projection ({method.upper()})')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    if save_path:
        plt.savefig(save_path)


def visualize_clusters_plotly(points, labels, title="3D Instance Segmentation"):
    """
    Plot point cloud with instance labels using Plotly.
    Args:
        points: (N, 3)
        labels: (N,) array of ints
    """
    df = pd.DataFrame(points, columns=['x', 'y', 'z'])
    df['label'] = labels.astype(str)

    fig = px.scatter_3d(df, x='x', y='y', z='z', color='label',
                        title=title,
                        size_max=1, opacity=0.8)
    fig.update_traces(marker=dict(size=2))
    fig.update_layout(scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False))
    fig.show()


def plot_loss_from_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        loss_history = pickle.load(f)

    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, marker='o', linewidth=2)
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compute_centroid_distances(embeddings, labels):
    """Returns pairwise centroid distances and centroid variance."""
    unique_labels = np.unique(labels)
    centroids = []
    for lbl in unique_labels:
        mask = labels == lbl
        if np.sum(mask) == 0:
            continue
        centroid = embeddings[mask].mean(axis=0)
        centroids.append(centroid)
    
    if len(centroids) < 2:
        return {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'std': 0.0, 'num_clusters': len(centroids)}
    
    centroids = np.stack(centroids)
    dist_matrix = squareform(pdist(centroids))
    np.fill_diagonal(dist_matrix, np.nan)  # ignore self-distances

    return {
        'min': float(np.nanmin(dist_matrix)),
        'max': float(np.nanmax(dist_matrix)),
        'mean': float(np.nanmean(dist_matrix)),
        'std': float(np.nanstd(dist_matrix)),
        'num_clusters': len(centroids)
    }