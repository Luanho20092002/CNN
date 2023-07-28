import numpy as np

class Adam:

    def __init__(self, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
    
    def update(self, this, dw, db):
        if this.is_init_mv:
            this.is_init_mv = False
            this.w_m, this.w_v = np.zeros_like(this.w), np.zeros_like(this.w)
            this.b_m, this.b_v = np.zeros_like(this.b), np.zeros_like(this.b)
            this.i = 0
        #Increase iter
        this.i += 1
        #Update weight
        this.w_m = self.beta1 * this.w_m + (1 - self.beta1) * (dw)
        this.w_v = self.beta2 * this.w_v + (1 - self.beta2) * (dw**2)
        w_m_hat = this.w_m / (1 - self.beta1**this.i)
        w_v_hat = this.w_v / (1 - self.beta2**this.i)
        this.w -= self.lr / (np.sqrt(w_v_hat) + self.eps) * w_m_hat
        #Update bias
        this.b_m = self.beta1 * this.b_m + (1 - self.beta1) * (db)
        this.b_v = self.beta2 * this.b_v + (1 - self.beta2) * (db**2)
        b_m_hat = this.b_m / (1 - self.beta1 ** this.i)
        b_v_hat = this.b_v / (1 - self.beta2 ** this.i)
        this.b -= self.lr / (np.sqrt(b_v_hat) + self.eps) * b_m_hat
