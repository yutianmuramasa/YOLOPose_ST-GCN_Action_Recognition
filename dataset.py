import os
import numpy as np
import torch
from torch.utils.data import Dataset

class ActionDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.labels = []

        self.class_names = sorted(os.listdir(root_dir))
        self.class_to_idx = {
            name: i for i, name in enumerate(self.class_names)
        }

        for cls in self.class_names:
            cls_dir = os.path.join(root_dir, cls)
            for file in os.listdir(cls_dir):
                if file.endswith(".npy"):
                    self.samples.append(os.path.join(cls_dir, file))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = np.load(self.samples[idx])          # [3,30,17]
        data = torch.tensor(data, dtype=torch.float32)
        label = self.labels[idx]
        return data, label
