import os
import numpy as np
import pandas as pd
import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    """Time series dataset for unsupervised representation learning"""
    
    def __init__(self, data, transform=None, to_tensor=True):
        """
        Args:
            data: numpy array or torch.Tensor of shape (N, C, L) where N is number of samples,
                  C is number of channels, L is sequence length
            transform: Optional transform to be applied on a sample
            to_tensor: Whether to convert data to tensor automatically
        """
        if to_tensor and not isinstance(data, torch.Tensor):
            self.data = torch.from_numpy(data).float()
        else:
            self.data = data
            
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample


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

def load_ucr_data(data_path):
    sub_data = data_path.split('/')[-1]
    train_file = os.path.join(data_path, sub_data + "_TRAIN.tsv")
    test_file = os.path.join(data_path, sub_data + "_TEST.tsv")
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    train_array = np.array(train_df)
    test_array = np.array(test_df)

    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_array[:, 0])   # get all class labels
    transform = {}                          # organize dataset based on labels
    for i, l in enumerate(labels):
        transform[l] = i

    train = np.expand_dims(train_array[:, 1:], 1).astype(np.float32)
    train_labels = np.vectorize(transform.get)(train_array[:, 0])
    test = np.expand_dims(test_array[:, 1:], 1).astype(np.float32)
    test_labels = np.vectorize(transform.get)(test_array[:, 0])

    mean = np.nanmean(np.concatenate([train, test]))
    var = np.nanvar(np.concatenate([train, test]))
    train = (train - mean) / math.sqrt(var)
    test = (test - mean) / math.sqrt(var)
    return train, train_labels, test, test_labels

def load_finance_data(data_path, normalize=True):
    """
    Load finance data and convert from (N, T, C) to (N, C, T) format
    
    Args:
        data_path: Path to the saved numpy file containing finance data
        normalize: Whether to normalize the data
        
    Returns:
        numpy.ndarray of shape (N, C, T)
    """
    # Load data (expected shape: N, T, C)
    data = np.load(data_path).astype(np.float32)
    print(f"Original data shape: {data.shape}")
    
    # Transpose from (N, T, C) to (N, C, T)
    data_transposed = np.transpose(data, (0, 2, 1))
    print(f"Transposed data shape: {data_transposed.shape}")
    
    if normalize:
        # Normalize the data
        mean = np.nanmean(data_transposed)
        var = np.nanvar(data_transposed) 
        data_transposed = (data_transposed - mean) / np.sqrt(var)
    
    return data_transposed


def get_dataset(dataset_name, data_path=None, transform=None, to_tensor=True, **kwargs):
    """
    Get dataset based on dataset name and path
    
    Args:
        dataset_name: Name of the dataset (e.g., 'ucr', 'mock', 'finance')
        data_path: Path to the dataset file
        transform: Optional transform to be applied on samples
        to_tensor: Whether to convert data to tensor automatically
        **kwargs: Additional arguments for dataset creation
    
    Returns:
        TimeSeriesDataset instance
    """
    if dataset_name == "mock":
        # Use mock data for testing - already returns tensor
        data = load_mock_data(**kwargs)
        # Convert back to numpy if we want to handle tensor conversion in dataset
        if not to_tensor:
            data = data.numpy()
        return TimeSeriesDataset(data, transform=transform, to_tensor=to_tensor)
    
    elif dataset_name == "ucr":
        # Load UCR dataset
        data, train_label, test, test_label = load_ucr_data(data_path)
        return TimeSeriesDataset(data, transform=transform, to_tensor=to_tensor)

    elif dataset_name == 'finance':
        # Load finance dataset
        data = load_finance_data(data_path)
        return TimeSeriesDataset(data, transform=transform, to_tensor=to_tensor)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
