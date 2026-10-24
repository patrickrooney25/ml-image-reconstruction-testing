import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def train_autoencoder(model, x_train, x_val, lr=1e-3, batch_size=128, max_epochs=50, patience=5, device='cpu'):
    """
    Trains a PyTorch autoencoder using Adam and MSE loss with early stopping.
    Returns the trained model and loss history.
    """
    model = model.to(device)
    
    # Convert numpy arrays to PyTorch Tensors
    train_tensor = torch.tensor(x_train, dtype=torch.float32)
    val_tensor = torch.tensor(x_val, dtype=torch.float32)
    
    # Set mean for TiedLinearAE
    if hasattr(model, 'set_mean'):
        model.set_mean(train_tensor)
        
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensor), batch_size=batch_size, shuffle=False)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    best_weights = None
    patience_counter = 0
    
    for epoch in range(max_epochs):
        # --- Training Loop ---
        model.train()
        train_loss = 0.0
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            
            # Use corrupted inputs for Denoising AE training pass
            if hasattr(model, 'forward_train'):
                x_hat, _ = model.forward_train(batch_x)
            else:
                x_hat, _ = model(batch_x)
                
            loss = criterion(x_hat, batch_x) # Reconstruct clean target
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(train_tensor)
        
        # --- Validation Loop ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch_x,) in val_loader:
                batch_x = batch_x.to(device)
                x_hat, _ = model(batch_x) # Evaluate clean val image
                loss = criterion(x_hat, batch_x)
                val_loss += loss.item() * batch_x.size(0)
                
        val_loss /= len(val_tensor)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
                
    # Restore best weights
    model.load_state_dict(best_weights)
    return model

def predict_autoencoder(model, x_test, device='cpu'):
    """Encodes and reconstructs numpy test data using a trained PyTorch model."""
    model.eval()
    model.to(device)
    test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        x_hat_tensor, h_tensor = model(test_tensor)
        
    return x_hat_tensor.cpu().numpy(), h_tensor.cpu().numpy()