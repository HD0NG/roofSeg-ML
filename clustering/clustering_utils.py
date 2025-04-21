import numpy as np

def filter_clusters(labels, min_cluster_size=20):
    """
    Removes small clusters by relabeling them as -1 (noise).
    """
    labels = np.array(labels)
    final_labels = labels.copy()
    for cluster_id in np.unique(labels):
        if cluster_id == -1:
            continue
        mask = labels == cluster_id
        if np.sum(mask) < min_cluster_size:
            final_labels[mask] = -1
    return final_labels

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

