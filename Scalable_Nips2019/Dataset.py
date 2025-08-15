import os
import numpy as np
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """Time series dataset for unsupervised representation learning"""
    
    def __init__(self, data):
        """
        Args:
            data: torch.Tensor of shape (N, C, L) where N is number of samples,
                  C is number of channels, L is sequence length
        """
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def load_mock_data(n=1024, c=5, l=256, seed=0):
    """Generate mock time series data for testing"""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 8*np.pi, l)
    data = []
    for i in range(n):
        waves = []
        base_phase = rng.uniform(0, 2*np.pi)
        for ch in range(c):
            freq = 0.5 + 0.2*ch + 0.05*rng.standard_normal()
            amp  = 1.0 + 0.3*rng.standard_normal()
            trend = 0.01 * ch * np.linspace(0, 1, l)
            waves.append(amp*np.sin(freq*t + base_phase) + trend)
        x = np.stack(waves, axis=0).astype(np.float32)  # (C,L)
        x += 0.1 * rng.standard_normal(size=(c, l)).astype(np.float32)
        data.append(x)
    return torch.from_numpy(np.stack(data, axis=0)).float()  # (N,C,L)


def get_dataset(dataset_name, data_path=None):
    """
    Get dataset based on dataset name and path
    
    Args:
        dataset_name: Name of the dataset (e.g., 'ucr', 'mock')
        data_path: Path to the dataset file
        **kwargs: Additional arguments for dataset creation
    
    Returns:
        TimeSeriesDataset instance
    """
    if dataset_name == "mock":
        # Use mock data for testing
        data = load_mock_data()
        return TimeSeriesDataset(data)
    
    elif dataset_name == "ucr":
        # Load UCR dataset
        if data_path and os.path.isfile(data_path):
            arr = np.load(data_path)
            assert arr.ndim == 3, f"UCR data must be (N, C, L), got shape {arr.shape}"
            data = torch.from_numpy(arr).float()
            return TimeSeriesDataset(data)

    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
