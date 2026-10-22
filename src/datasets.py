#MNIST and Fashion MNIST loaders and train/val/test splits

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np

def load_data(dataset_name='mnist', data_dir='./data'):
    """
    Loads MNIST or Fashion-MNIST, normalizes pixels to [0, 1], 
    splits into 55k Train, 5k Validation, and 10k Test sets.
    Returns flattened Numpy arrays (N, 784) for linear models & PyTorch Tensors.
    """
    transform = transforms.ToTensor() # Automatically scales pixels to [0, 1]
    
    if dataset_name.lower() == 'mnist':
        full_train = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    elif dataset_name.lower() == 'fashion-mnist':
        full_train = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    # Standard 55,000 / 5,000 split
    train_subset, val_subset = torch.utils.data.random_split(
        full_train, [55000, 5000], generator=torch.Generator().manual_seed(42)
    )
    
    # Convert to Numpy vectors (N, 784) for PCA/NMF
    def extract_and_flatten(subset):
        loader = torch.utils.data.DataLoader(subset, batch_size=len(subset), shuffle=False)
        x_tensor, y_tensor = next(iter(loader))
        x_flat = x_tensor.view(x_tensor.size(0), -1).numpy()
        y_np = y_tensor.numpy()
        return x_flat, y_np

    x_train, y_train = extract_and_flatten(train_subset)
    x_val, y_val = extract_and_flatten(val_subset)
    x_test, y_test = extract_and_flatten(test_set)
    
    return {
        'x_train': x_train, 'y_train': y_train,
        'x_val': x_val,     'y_val': y_val,
        'x_test': x_test,   'y_test': y_test
    }