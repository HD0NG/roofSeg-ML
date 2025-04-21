import torch
import torch.nn as nn
import torch.nn.functional as F

# Discriminative Loss
def discriminative_loss(embeddings, instance_labels, delta_v=0.5, delta_d=1.5,
                        alpha=1.0, beta=1.0, gamma=0.001):
    """
    Computes the discriminative loss for instance segmentation.

    Args:
        embeddings (Tensor): (B, N, D) - pointwise embeddings
        instance_labels (Tensor): (B, N) - instance IDs per point
        delta_v (float): margin for close loss
        delta_d (float): margin for apart loss
        alpha, beta, gamma (float): weights for each term

    Returns:
        loss (Tensor): scalar loss value
    """
    batch_size, num_points, emb_dim = embeddings.size()
    total_loss = 0.0

    for b in range(batch_size):
        embedding = embeddings[b]              # (N, D)
        labels = instance_labels[b]            # (N,)
        unique_labels = torch.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]  # ignore noise labels

        K = len(unique_labels)
        if K == 0:
            continue

        cluster_means = []
        l_close = 0.0
        for label in unique_labels:
            mask = labels == label
            if mask.sum() == 0:
                continue

            cluster = embedding[mask]  # Points of one instance
            mean = cluster.mean(dim=0)
            cluster_means.append(mean)

            # ℓ_close: Pull to mean if outside margin δ_v
            dist = torch.norm(cluster - mean, dim=1) - delta_v
            l_close += torch.mean(torch.clamp(dist, min=0.0) ** 2)

        l_close /= K
        cluster_means = torch.stack(cluster_means)  # (K, D)

        # ℓ_apart: Push means away if closer than margin δ_d
        l_apart = 0.0
        for i in range(K):
            for j in range(i + 1, K):
                dist = torch.norm(cluster_means[i] - cluster_means[j], p=1)
                l_apart += F.relu(2 * delta_d - dist) ** 2
        if K > 1:
            l_apart /= (K * (K - 1))

        # ℓ_reg: Pull all means toward origin (L1 norm)
        l_reg = torch.mean(torch.norm(cluster_means, p=1, dim=1))

        # Combine all
        loss = alpha * l_close + beta * l_apart + gamma * l_reg
        total_loss += loss

    return total_loss / batch_size