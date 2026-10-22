#Scikit-learn wrappers for PCA and NMF

import numpy as np
from sklearn.decomposition import PCA, NMF

class PCAModel:
    def __init__(self, k):
        self.k = k
        self.model = PCA(n_components=k, random_state=42)
        
    def fit(self, x_train):
        self.model.fit(x_train)
        
    def encode(self, x):
        return self.model.transform(x)
        
    def decode(self, h):
        return self.model.inverse_transform(h)
        
    def reconstruct(self, x):
        h = self.encode(x)
        x_hat = self.decode(h)
        return x_hat, h

class NMFModel:
    def __init__(self, k):
        self.k = k
        # max_iter=200 matches Lee & Seung setting
        self.model = NMF(n_components=k, init='nndsvda', max_iter=200, random_state=42)
        
    def fit(self, x_train):
        # Learns dictionary W and training activations H
        self.model.fit(x_train)
        
    def encode(self, x):
        # Holds W fixed, optimizes H for new input
        return self.model.transform(x)
        
    def decode(self, h):
        # x_hat = H * W
        return np.dot(h, self.model.components_)
        
    def reconstruct(self, x):
        h = self.encode(x)
        x_hat = self.decode(h)
        return x_hat, h