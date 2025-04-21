from torch.utils.data import Dataset
import os
import numpy as np
import torch

class LiDARPointCloudDataset(Dataset):
    def __init__(self, point_folder, label_folder, max_points=512, mode="train"):
        """
        Args:
            point_folder (str): Path to the folder containing point cloud .txt files.
            label_folder (str): Path to the folder containing corresponding label .txt files.
            max_points (int): Maximum number of points per cloud (for subsampling or padding).
            mode (str): "train" or "test".
        """
        self.point_folder = point_folder
        self.label_folder = label_folder
        self.max_points = max_points
        self.mode = mode  # 'train' or 'test'

        # List all available point files
        self.point_files = sorted([f for f in os.listdir(point_folder) if f.endswith(".txt")])
        self.label_files = sorted([f for f in os.listdir(label_folder) if f.endswith(".txt")])

        # Ensure matching point and label files
        assert len(self.point_files) == len(self.label_files), "Mismatch in points and labels count."

    def load_txt_file(self, file_path, num_features=3):
        """
        Loads a .txt file, converting string lines into a NumPy array of type float64.
        Assumes space/comma-separated values.
        """
        data = []
        with open(file_path, "r") as file:
            for line in file:
                try:
                    values = np.array(line.strip().replace(',', ' ').split(), dtype=np.float64)
                    if len(values) == num_features:
                        data.append(values)
                except ValueError:
                    continue  # Skip lines that cannot be converted

        return np.array(data)

    def load_point_cloud(self, file_path):
        """Loads point cloud (XYZ) from a .txt file and returns a NumPy array."""
        return self.load_txt_file(file_path, num_features=3)  # Expecting [x, y, z]

    def load_labels(self, file_path, num_points):
        """Loads labels from a .txt file, ensuring it matches the number of points."""
        labels = self.load_txt_file(file_path, num_features=1).flatten()
        if len(labels) != num_points:
            print(f"Warning: {file_path} has {len(labels)} labels, expected {num_points}. Using zero-padding.")
            labels = np.pad(labels, (0, max(0, num_points - len(labels))), 'constant', constant_values=0)
        return labels

    def pad_or_subsample(self, points, labels):
        """Ensures a fixed number of points per cloud using padding or subsampling."""
        num_points = points.shape[0]

        if num_points > self.max_points:
            # Randomly sample points
            indices = np.random.choice(num_points, self.max_points, replace=False)
            points, labels = points[indices], labels[indices]
        elif num_points < self.max_points:
            # Pad with zeros
            pad_size = self.max_points - num_points
            pad_points = np.zeros((pad_size, 3), dtype=np.float64)
            # pad_labels = np.zeros(pad_size, dtype=np.int64)
            pad_labels = np.full(pad_size, -1, dtype=np.int64)
            points = np.vstack((points, pad_points))
            labels = np.hstack((labels, pad_labels))

        return points, labels

    def __len__(self):
        return len(self.point_files)

    def __getitem__(self, idx):
        point_path = os.path.join(self.point_folder, self.point_files[idx])
        label_path = os.path.join(self.label_folder, self.label_files[idx])

        point_cloud = self.load_point_cloud(point_path)
        labels = self.load_labels(label_path, num_points=point_cloud.shape[0])

        # Apply padding or subsampling
        point_cloud, labels = self.pad_or_subsample(point_cloud, labels)

        return torch.tensor(point_cloud, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)
