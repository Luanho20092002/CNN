import numpy as np

class Dense:

    def __init__(self, units, active="relu") -> None:
        self.units = units
        self.active = active
        self.has_weight = True
        self.is_init_mv = True

    def forward(self, X):
        if self.has_weight:
            self.has_weight = False
            dim = X.shape[1]
            self.w = 0.01 * np.random.randn(dim, self.units)
            self.b = 0.01 * np.random.randn(1, self.units)
        self.X = X
        Z = X @ self.w + self.b
        if self.active == "relu":
            self.mask = (Z > 0)
            return Z * self.mask
        elif self.active == "softmax":
            return self.softmax(Z)
    
    def backward(self, dZ, optimizer):
        if self.active == "relu":
            dZ = dZ * self.mask
        dX = dZ @ self.w.T
        dw = self.X.T @ dZ
        db = np.sum(dZ, axis=0, keepdims=True)
        optimizer.update(self, dw, db)
        return dX

    def softmax(self, Z):
        eZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return eZ/eZ.sum(axis=1).reshape(-1, 1)
