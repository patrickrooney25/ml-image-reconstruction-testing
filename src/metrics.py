#MSE, PSNR, Hoyer Sparsity, SSIM

import numpy as np

def compute_mse(x_true, x_pred):
    """Per-pixel Mean Squared Error across all samples and dimensions."""
    return np.mean((x_true - x_pred) ** 2)

def compute_psnr(mse, max_val=1.0):
    """Peak Signal-to-Noise Ratio (PSNR) for images bounded in [0, 1]."""
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10((max_val ** 2) / mse)

def hoyer_sparsity(h):
    """
    Computes average Hoyer Sparsity S(h) across a matrix of code vectors h (N, k).
    S(h) = (sqrt(k) - L1/L2) / (sqrt(k) - 1)
    """
    h_abs = np.abs(h)
    N, k = h_abs.shape
    if k == 1:
        return 0.0
    
    l1 = np.sum(h_abs, axis=1)
    l2 = np.linalg.norm(h_abs, axis=1)
    
    # Prevent division by zero for all-zero latent codes
    l2[l2 == 0] = 1e-12 
    
    sparsity = (np.sqrt(k) - (l1 / l2)) / (np.sqrt(k) - 1.0)
    return np.mean(sparsity)