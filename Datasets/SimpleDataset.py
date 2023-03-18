import torch
from torch.utils.data import Dataset

class SimpleDataset(Dataset):
    def __init__(self, datapoints, labels):
        self.datapoints = datapoints
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        features = self.datapoints[idx]
        label = self.labels[idx]
        return features, label
