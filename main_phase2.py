import torch
from src.datasets import load_data
from src.models import TiedLinearAE, NonlinearAE, DenoisingAE
from src.train import train_autoencoder, predict_autoencoder
from src.metrics import compute_mse, compute_psnr, hoyer_sparsity

def run_phase2():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    print("Loading MNIST dataset...")
    data = load_data('mnist')
    x_train, x_val, x_test = data['x_train'], data['x_val'], data['x_test']
    
    bottlenecks = [2, 4, 8, 16, 32, 64]
    
    # -------------------------------------------------------------------------
    print("\n--- RUNNING TIED LINEAR AUTOENCODER ---")
    for k in bottlenecks:
        model = TiedLinearAE(in_features=784, k=k)
        model = train_autoencoder(model, x_train, x_val, lr=1e-3, max_epochs=30, device=device)
        x_hat, h_test = predict_autoencoder(model, x_test, device=device)
        
        mse = compute_mse(x_test, x_hat)
        psnr = compute_psnr(mse)
        sparsity = hoyer_sparsity(h_test)
        print(f"TiedLinearAE (k={k:2d}) -> Test MSE: {mse:.5f} | PSNR: {psnr:.2f} dB | Sparsity: {sparsity:.3f}")

    # -------------------------------------------------------------------------
    print("\n--- RUNNING NON-LINEAR AUTOENCODER ---")
    for k in bottlenecks:
        model = NonlinearAE(in_features=784, k=k)
        model = train_autoencoder(model, x_train, x_val, lr=1e-3, max_epochs=30, device=device)
        x_hat, h_test = predict_autoencoder(model, x_test, device=device)
        
        mse = compute_mse(x_test, x_hat)
        psnr = compute_psnr(mse)
        sparsity = hoyer_sparsity(h_test)
        print(f"NonlinearAE  (k={k:2d}) -> Test MSE: {mse:.5f} | PSNR: {psnr:.2f} dB | Sparsity: {sparsity:.3f}")

    # -------------------------------------------------------------------------
    print("\n--- RUNNING DENOISING AUTOENCODER ---")
    for k in bottlenecks:
        model = DenoisingAE(in_features=784, k=k, noise_std=0.3)
        model = train_autoencoder(model, x_train, x_val, lr=1e-3, max_epochs=30, device=device)
        x_hat, h_test = predict_autoencoder(model, x_test, device=device)
        
        mse = compute_mse(x_test, x_hat)
        psnr = compute_psnr(mse)
        sparsity = hoyer_sparsity(h_test)
        print(f"DenoisingAE  (k={k:2d}) -> Test MSE: {mse:.5f} | PSNR: {psnr:.2f} dB | Sparsity: {sparsity:.3f}")

if __name__ == "__main__":
    run_phase2()