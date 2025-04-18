import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import MeanShift
from scipy.spatial.distance import pdist, squareform
import torch


def visualize_embeddings(embeddings, labels, save_path, title='PCA', method='pca'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    embeddings = embeddings.reshape(-1, embeddings.shape[-1])
    labels = labels.flatten()

    if method == 'pca':
        reduced = PCA(n_components=2).fit_transform(embeddings)
    elif method == 'tsne':
        reduced = TSNE(n_components=2, perplexity=30, init='random').fit_transform(embeddings)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    plt.figure(figsize=(6, 5))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab20', s=2)
    plt.title(f'{title} - {method.upper()}')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


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