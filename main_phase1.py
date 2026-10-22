#pipeline script to run phase 1 baseline experiments

from src.datasets import load_data
from src.baselines import PCAModel, NMFModel
from src.metrics import compute_mse, compute_psnr, hoyer_sparsity

def run_phase1():
    print("Loading MNIST dataset...")
    data = load_data('mnist')
    x_train, x_test = data['x_train'], data['x_test']
    
    bottlenecks = [2, 4, 8, 16, 32, 64]
    
    print("\n--- RUNNING PCA BASELINE ---")
    for k in bottlenecks:
        pca = PCAModel(k=k)
        pca.fit(x_train)
        x_hat, h_test = pca.reconstruct(x_test)
        
        mse = compute_mse(x_test, x_hat)
        psnr = compute_psnr(mse)
        sparsity = hoyer_sparsity(h_test)
        print(f"PCA (k={k:2d}) -> Test MSE: {mse:.5f} | PSNR: {psnr:.2f} dB | Sparsity: {sparsity:.3f}")
        
    print("\n--- RUNNING NMF BASELINE ---")
    for k in bottlenecks:
        nmf = NMFModel(k=k)
        nmf.fit(x_train)
        x_hat, h_test = nmf.reconstruct(x_test)
        
        mse = compute_mse(x_test, x_hat)
        psnr = compute_psnr(mse)
        sparsity = hoyer_sparsity(h_test)
        print(f"NMF (k={k:2d}) -> Test MSE: {mse:.5f} | PSNR: {psnr:.2f} dB | Sparsity: {sparsity:.3f}")

if __name__ == "__main__":
    run_phase1()