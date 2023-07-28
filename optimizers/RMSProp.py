import numpy as np

class RMSProp:

    def __init__(self, lr=0.001, gamma=0.9, eps=1e-8) -> None:
        self.lr = lr
        self.gamma = gamma
        self.eps = eps

    def update(self, this, dw, db):
        if this.is_init_mv:
            this.is_init_mv = False
            this.v_w = np.zeros_like(this.w)
            this.v_b = np.zeros_like(this.b)
        
        this.v_w = self.gamma * this.v_w + (1 - self.gamma) * (np.array(dw)**2)
        this.w -= self.lr * dw / (np.sqrt(this.v_w) + self.eps)

        this.v_b = self.gamma * this.v_b + (1 - self.gamma) * (np.array(db)**2)
        this.b -= self.lr * db / (np.sqrt(this.v_b) + self.eps)