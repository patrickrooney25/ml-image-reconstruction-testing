import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datasets import load_data
from src.baselines import PCAModel, NMFModel

import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.datasets import load_data
from src.baselines import PCAModel, NMFModel
from src.models import TiedLinearAE, NonlinearAE, DenoisingAE
from src.train import train_autoencoder, predict_autoencoder
from src.metrics import compute_mse, compute_psnr, hoyer_sparsity

def run_experiment_1(dataset_name='mnist', output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running Experiment 1 on {dataset_name.upper()} using device: {device}\n")
    
    # 1. Load Data
    data = load_data(dataset_name)
    x_train, x_val, x_test = data['x_train'], data['x_val'], data['x_test']
    
    bottlenecks = [2, 4, 8, 16, 32, 64]
    models = ['PCA', 'NMF', 'TiedLinearAE', 'NonlinearAE', 'DenoisingAE']
    
    # Container for all logged metrics
    results = {
        m: {'mse': [], 'psnr': [], 'sparsity': []} for m in models
    }
    
    # 2. Run Baselines (PCA & NMF)
    print("--- Running PCA Baseline ---")
    for k in bottlenecks:
        pca = PCAModel(k=k)
        pca.fit(x_train)
        x_hat, h = pca.reconstruct(x_test)
        
        mse = compute_mse(x_test, x_hat)
        psnr = compute_psnr(mse)
        sp = hoyer_sparsity(h)
        
        results['PCA']['mse'].append(float(mse))
        results['PCA']['psnr'].append(float(psnr))
        results['PCA']['sparsity'].append(float(sp))
        print(f"PCA (k={k:2d}) -> MSE: {mse:.5f} | PSNR: {psnr:.2f} dB")
        
    print("\n--- Running NMF Baseline ---")
    for k in bottlenecks:
        nmf = NMFModel(k=k)
        nmf.fit(x_train)
        x_hat, h = nmf.reconstruct(x_test)
        
        mse = compute_mse(x_test, x_hat)
        psnr = compute_psnr(mse)
        sp = hoyer_sparsity(h)
        
        results['NMF']['mse'].append(float(mse))
        results['NMF']['psnr'].append(float(psnr))
        results['NMF']['sparsity'].append(float(sp))
        print(f"NMF (k={k:2d}) -> MSE: {mse:.5f} | PSNR: {psnr:.2f} dB")

    # 3. Run PyTorch Autoencoders
    print("\n--- Running PyTorch Autoencoders ---")
    for k in bottlenecks:
        print(f"\nTraining models for bottleneck k={k}...")
        
        # Tied Linear Autoencoder
        tied_ae = TiedLinearAE(in_features=784, k=k)
        tied_ae = train_autoencoder(tied_ae, x_train, x_val, lr=1e-3, max_epochs=30, device=device)
        x_hat, h = predict_autoencoder(tied_ae, x_test, device=device)
        results['TiedLinearAE']['mse'].append(float(compute_mse(x_test, x_hat)))
        results['TiedLinearAE']['psnr'].append(float(compute_psnr(results['TiedLinearAE']['mse'][-1])))
        results['TiedLinearAE']['sparsity'].append(float(hoyer_sparsity(h)))

        # Nonlinear Autoencoder
        nonlin_ae = NonlinearAE(in_features=784, k=k)
        nonlin_ae = train_autoencoder(nonlin_ae, x_train, x_val, lr=1e-3, max_epochs=30, device=device)
        x_hat, h = predict_autoencoder(nonlin_ae, x_test, device=device)
        results['NonlinearAE']['mse'].append(float(compute_mse(x_test, x_hat)))
        results['NonlinearAE']['psnr'].append(float(compute_psnr(results['NonlinearAE']['mse'][-1])))
        results['NonlinearAE']['sparsity'].append(float(hoyer_sparsity(h)))

        # Denoising Autoencoder
        denoise_ae = DenoisingAE(in_features=784, k=k, noise_std=0.3)
        denoise_ae = train_autoencoder(denoise_ae, x_train, x_val, lr=1e-3, max_epochs=30, device=device)
        x_hat, h = predict_autoencoder(denoise_ae, x_test, device=device)
        results['DenoisingAE']['mse'].append(float(compute_mse(x_test, x_hat)))
        results['DenoisingAE']['psnr'].append(float(compute_psnr(results['DenoisingAE']['mse'][-1])))
        results['DenoisingAE']['sparsity'].append(float(hoyer_sparsity(h)))

    # 4. Save JSON of Metrics
    results_path = os.path.join(output_dir, f'exp1_{dataset_name}_metrics.json')
    with open(results_path, 'w') as f:
        json.dump({'bottlenecks': bottlenecks, 'results': results}, f, indent=4)
    print(f"\n[Saved] Metrics saved to {results_path}")

    # 5. Generate Figures
    generate_plots(bottlenecks, results, dataset_name, output_dir)

def generate_plots(bottlenecks, results, dataset_name, output_dir):
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    colors = {
        'PCA': '#1f77b4',
        'NMF': '#ff7f0e',
        'TiedLinearAE': '#2ca02c',
        'NonlinearAE': '#d62728',
        'DenoisingAE': '#9467bd'
    }
    markers = {'PCA': 'o', 'NMF': 's', 'TiedLinearAE': '^', 'NonlinearAE': 'D', 'DenoisingAE': 'v'}

    # -------------------------------------------------------------
    # Figure 1: Test MSE vs Bottleneck Size (k)
    # -------------------------------------------------------------
    plt.figure(figsize=(7, 5), dpi=300)
    for model_name, metric_dict in results.items():
        linestyle = '--' if 'Linear' in model_name or model_name == 'PCA' else '-'
        plt.plot(bottlenecks, metric_dict['mse'], marker=markers[model_name],
                 color=colors[model_name], linestyle=linestyle, label=model_name, linewidth=1.8)

    plt.xlabel('Bottleneck Size ($k$)', fontsize=12)
    plt.ylabel('Test Reconstruction Error (MSE)', fontsize=12)
    plt.title(f'Reconstruction Error vs. Bottleneck Capacity ({dataset_name.upper()})', fontsize=13, fontweight='bold')
    plt.xticks(bottlenecks)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    
    fig1_path = os.path.join(output_dir, f'fig1_reconstruction_mse_{dataset_name}.png')
    plt.savefig(fig1_path)
    plt.close()
    print(f"[Saved] Figure 1 saved to {fig1_path}")

    # -------------------------------------------------------------
    # Figure 2: The Nonlinear Advantage Gap: Delta(k) = MSE_PCA - MSE_AE
    # -------------------------------------------------------------
    plt.figure(figsize=(7, 4.5), dpi=300)
    pca_mse = np.array(results['PCA']['mse'])
    nonlin_mse = np.array(results['NonlinearAE']['mse'])
    delta_k = pca_mse - nonlin_mse

    plt.plot(bottlenecks, delta_k, marker='D', color='#d62728', linewidth=2.0, label=r'$\Delta(k) = \mathrm{MSE}_{\mathrm{PCA}} - \mathrm{MSE}_{\mathrm{AE}}$')
    plt.axhline(0, color='gray', linestyle=':', alpha=0.7)

    plt.xlabel('Bottleneck Size ($k$)', fontsize=12)
    plt.ylabel(r'Nonlinear Advantage $\Delta(k)$', fontsize=12)
    plt.title(r'Nonlinear Advantage Over Linear Baseline ($\Delta(k)$ vs. $k$)', fontsize=13, fontweight='bold')
    plt.xticks(bottlenecks)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    fig2_path = os.path.join(output_dir, f'fig2_nonlinear_gap_{dataset_name}.png')
    plt.savefig(fig2_path)
    plt.close()
    print(f"[Saved] Figure 2 saved to {fig2_path}")

if __name__ == '__main__':
    run_experiment_1('mnist')