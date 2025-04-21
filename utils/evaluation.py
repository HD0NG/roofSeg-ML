import torch
import os
import json
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from clustering.mean_shift import cluster_embeddings

def adjusted_rand_index(pred_labels, gt_labels):
    return adjusted_rand_score(gt_labels, pred_labels)


def instance_mean_iou(gt_labels, pred_labels):
    gt_ids = np.unique(gt_labels)
    pred_ids = np.unique(pred_labels)
    iou_matrix = np.zeros((len(gt_ids), len(pred_ids)))

    for i, gt_id in enumerate(gt_ids):
        gt_mask = gt_labels == gt_id
        for j, pred_id in enumerate(pred_ids):
            pred_mask = pred_labels == pred_id
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()
            if union > 0:
                iou_matrix[i, j] = intersection / union

    row_ind, col_ind = linear_sum_assignment(-iou_matrix)  # maximize total IoU
    matched_ious = iou_matrix[row_ind, col_ind]
    return matched_ious.mean()


def save_cluster_plot(points, labels, path):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=labels, cmap='tab20', s=2)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def run_inference_on_scene(model, pointcloud, gt_labels, device, save_path, scene_name):
    model.eval()
    with torch.no_grad():
        # pc_tensor = torch.tensor(pointcloud, dtype=torch.float32).unsqueeze(0).to(device)
        if isinstance(pointcloud, torch.Tensor):
            pc_tensor = pointcloud.clone().detach().unsqueeze(0).to(device)
        else:
            pc_tensor = torch.tensor(pointcloud, dtype=torch.float32).unsqueeze(0).to(device)
        embeddings = model(pc_tensor).squeeze(0).cpu().numpy()

        pred_labels = cluster_embeddings(embeddings)
        ari = adjusted_rand_index(pred_labels, gt_labels)
        miou = instance_mean_iou(gt_labels, pred_labels)

        print(f"{scene_name} — ARI: {ari:.3f}, mIoU: {miou:.3f}")

        vis_path = os.path.join(save_path, f"{scene_name}_clusters.png")
        save_cluster_plot(pointcloud, pred_labels, vis_path)

        return {
            "scene": scene_name,
            "ARI": ari,
            "mean_IoU": miou,
            "num_pred_instances": len(np.unique(pred_labels)),
            "num_gt_instances": len(np.unique(gt_labels))
        }


def batch_evaluate(model, dataset, device, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    all_metrics = []

    for i, (points, labels) in enumerate(dataset):
        scene_name = f"scene_{i:03d}"
        result = run_inference_on_scene(
            model, points, labels, device, output_dir, scene_name
        )
        all_metrics.append(result)

    # Save all metrics to JSON
    with open(os.path.join(output_dir, "evaluation_summary.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n✅ Batch evaluation complete. Summary written to evaluation_summary.json")
