import torch
import torch.nn as nn
import torch.nn.functional as F

class TiedLinearAE(nn.Module):
    """
    Tied-weight linear autoencoder: x_hat = W * W^T * (x - mu) + mu.
    Uses a single weight matrix W of shape (784, k).
    """
    def __init__(self, in_features=784, k=16):
        super().__init__()
        self.k = k
        # W is initialized with shape (784, k)
        self.W = nn.Parameter(torch.randn(in_features, k) * 0.01)
        self.register_buffer('mu', torch.zeros(1, in_features))

    def set_mean(self, x_train_tensor):
        """Sets data mean vector from training data."""
        self.mu.copy_(x_train_tensor.mean(dim=0, keepdim=True))

    def encode(self, x):
        x_centered = x - self.mu
        return torch.matmul(x_centered, self.W) # (N, k)

    def decode(self, h):
        return torch.matmul(h, self.W.t()) + self.mu # (N, 784)

    def forward(self, x):
        h = self.encode(x)
        x_hat = self.decode(h)
        return x_hat, h


class NonlinearAE(nn.Module):
    """
    Deep Feedforward Autoencoder:
    Encoder: 784 -> 512 -> 128 -> k (Linear bottleneck)
    Decoder: k -> 128 -> 512 -> 784 (Sigmoid output)
    """
    def __init__(self, in_features=784, k=16):
        super().__init__()
        self.k = k
        
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, k) # Linear bottleneck
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(k, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, in_features),
            nn.Sigmoid() # Keeps pixel outputs in [0, 1]
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, h):
        return self.decoder(h)

    def forward(self, x):
        h = self.encode(x)
        x_hat = self.decode(h)
        return x_hat, h


class DenoisingAE(NonlinearAE):
    """
    Inherits NonlinearAE architecture.
    Corrupts inputs with Gaussian noise (sigma_tr) during training forward pass.
    """
    def __init__(self, in_features=784, k=16, noise_std=0.3):
        super().__init__(in_features=in_features, k=k)
        self.noise_std = noise_std

    def forward_train(self, x):
        """Corrupts input x during training, evaluates against clean x target."""
        noise = torch.randn_like(x) * self.noise_std
        x_noisy = torch.clamp(x + noise, 0.0, 1.0)
        
        h = self.encode(x_noisy)
        x_hat = self.decode(h)
        return x_hat, h