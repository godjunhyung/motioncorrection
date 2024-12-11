import os
import numpy as np
import pandas as pd
import torch

def bin_iqm(df, column_name):
    df_copy = df.copy()

    data = df_copy[column_name]
    
    counts, bin_edges = np.histogram(data, bins=100, range=(data.min(), data.max()))
    bin_indices = np.digitize(data, bins=bin_edges[:-1], right=False) - 1

    bin_indices = np.clip(bin_indices, 0, 99)

    df_copy[column_name] = bin_indices

    return df_copy

def load_data_and_labels(csv_path, iqm_label):
    """
    Load data, labels, and sequence information for all predefined sequences.

    Parameters:
        csv_path (str): Path to the CSV file containing slice information.
        iqm_label (str): The label (e.g., "Haarpsi", "VSI", "VIF", "NQM") to use.

    Returns:
        data (list): List of loaded NumPy arrays for slices.
        labels (list): Corresponding labels for the slices.
        sequences (list): Sequence information for each slice.
    """
    # Define the four sequences
    sequences_to_process = ["t1", "t2", "t1post", "flair"]

    # Load the CSV
    df = pd.read_csv(csv_path)
    df = df[(df["Slice Index"] < 10)]
    binned_df = bin_iqm(df, iqm_label)

    # Prepare lists to store data, labels, and sequences
    data = []
    labels = []
    sequences = []

    for sequence in sequences_to_process:
        # Filter the DataFrame based on sequence
        filtered_df = binned_df[(binned_df["sequence"] == sequence)]

        # Process motion sequences
        for _, row in filtered_df.iterrows():
            subject_id = row["Subject ID"]
            slice_idx = row["Slice Index"]
            motion_level = row["Motion_Level"]

            # Construct the path to the motion data
            data_path = f"/root/motioncorrection/data/{sequence}_g{motion_level}/{subject_id}_motion.npy"
            if os.path.exists(data_path):
                slice_data = np.load(data_path)[slice_idx]  # Load specific slice
                data.append(slice_data)
                labels.append(row[iqm_label])
                sequences.append(sequence)

        # Process clear sequences (label = 100)
        unique_subjects = filtered_df["Subject ID"].unique()
        for subject_id in unique_subjects:
            clear_path = f"/root/motioncorrection/data/{sequence}_clear/{subject_id}.npy"
            if os.path.exists(clear_path):
                clear_data = np.load(clear_path)
                for slice_idx in range(min(10, clear_data.shape[0])):  # Use slices < 10
                    data.append(clear_data[slice_idx])
                    labels.append(100)  # Clear sequences have label = 100
                    sequences.append(sequence)
    
    data = np.array(data)
    data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(1)
    print(data_tensor.shape)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)  # Assuming labels are float

    return data_tensor, labels_tensor, sequences

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, iqm_label, load_feats=None):

        self.X, self.y, self.sequence = load_data_and_labels(csv_path, iqm_label)
        
        self.bias_feats = None

        if load_feats:
            print("Loading biased features", load_feats)
            self.bias_feats = torch.load(load_feats, map_location="cpu")
        
        print(f"Read {len(self.X)} records")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):

        x = self.X[index]
        y = self.y[index]
        sequence = self.sequence[index]
        
        if self.bias_feats is not None:
            return x, y, self.bias_feats[index]
        else:
            return x, y, sequence