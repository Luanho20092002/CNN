import numpy as np

class Dense:

    def __init__(self, units, active) -> None:
        self.units = units
        self.active = active
        self.w = 0
        self.isHasFilter = True

    def forward(self, Xin):
        self.Xin = Xin
        if self.isHasFilter:
            dim = Xin.shape[1]
            self.w = 0.1* np.random.randn(self.units, dim)
            self.b = 0.1* np.random.randn(1, self.units)
            self.isHasFilter = False
        Z = Xin.dot(self.w.T) + self.b
        if self.active == "relu":
            A = self.relu(Z)
        elif self.active == "softmax":
            A = self.softmax(Z)
            self.A = A
        return A

    def backward(self, dL, eta):
        if self.active == "relu":
            dLdZ = dL["dLdZ"]
            w_next = dL["w_next"]
            dLdZ = dLdZ.dot(w_next) 
            dLdZ[dLdZ<0] = 0
            w_curr = self.w
            dw = dLdZ.T.dot(self.Xin)
            db = np.sum(dLdZ, axis=0)
            self.w = self.w - eta*dw
            self.b = self.b - eta*db
            return {"dLdZ":dLdZ, "w_next":w_curr}
        elif self.active == "tanh":
            pass
        elif self.active == "sigmoid":
            pass

    # Tính backward tại ouput layer
    def dLdZ(self, y, eta, N):
        if self.active == "softmax":
            w_curr = self.w
            dLdZ = 1/N * (self.A - y)
            dw = dLdZ.T.dot(self.Xin)
            db = np.sum(dLdZ, axis=0)
            self.w = self.w - eta*dw
            self.b = self.b - eta*db
            return {"dLdZ":dLdZ, "w_next":w_curr}
        elif self.active == "sigmoid":
            return

    def relu(self, Z):
        return np.maximum(0, Z)
    
    def softmax(self, Z):
        eZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return eZ/eZ.sum(axis=1).reshape(-1, 1)
