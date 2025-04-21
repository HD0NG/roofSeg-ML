import torch
import torch.nn as nn
import torch.nn.functional as F


# Contrastive Loss
def contrastive_loss(embeddings, labels, margin=1.0):
    """
    Contrastive loss to cluster points of the same roof face and separate different faces.
    """
    batch_size, num_points, emb_dim = embeddings.shape
    loss = 0.0
    count = 0

    for i in range(batch_size):
        for j in range(num_points):
            for k in range(num_points):
                if j == k:
                    continue
                d = torch.norm(embeddings[i, j] - embeddings[i, k], p=2)
                if labels[i, j] == labels[i, k]:  # Same face
                    loss += d ** 2
                else:  # Different faces
                    loss += max(0, margin - d) ** 2
                count += 1

    return loss / count

def cosine_contrastive_loss(embeddings, labels, margin=0.5):
    """
    embeddings: (B, N, D)
    labels: (B, N)
    """
    B, N, D = embeddings.shape
    embeddings = F.normalize(embeddings, p=2, dim=-1)  # Normalize for cosine

    total_loss = 0
    for b in range(B):
        emb = embeddings[b]  # (N, D)
        lbl = labels[b]      # (N,)

        sim = torch.matmul(emb, emb.T)  # (N, N)
        label_matrix = lbl.unsqueeze(0) == lbl.unsqueeze(1)  # (N, N)

        pos_mask = label_matrix.float()
        neg_mask = 1.0 - pos_mask

        pos_loss = (1 - sim) * pos_mask
        neg_loss = F.relu(sim - margin) * neg_mask

        total_loss += (pos_loss.sum() + neg_loss.sum()) / (N * N)

    return total_loss / B
