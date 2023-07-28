import numpy as np

class Conv2D:

    def __init__(self, kernel_size, filter=32, stride=1, padding="valid", active="relu") -> None:
        self.filter = filter
        self.stride = stride
        self.active = active
        self.has_weight = True
        self.is_init_mv = True
        self.padding = padding
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
    
    def forward(self, X):
        self.X = X
        nX, cX, hX, wX = X.shape
        #Init w and b
        if self.has_weight:
            self.has_weight = False
            self.w = np.random.randn(self.filter, cX, self.kernel_size[0], self.kernel_size[1])
            self.b = np.zeros((self.filter, 1))
            #Caculate padding
            self.h_pad, self.w_pad = 0, 0
            if self.padding == 'same':
                self.h_pad = int((self.kernel_size[0] - self.stride) / 2)
                self.w_pad = int((self.kernel_size[1] - self.stride) / 2)
                
        nf, _, hf, wf = self.w.shape
        self.X_pad = np.pad(X, ((0,0),(0,0),(self.h_pad,self.h_pad),(self.w_pad,self.w_pad)))
        h_new = (hX + 2*self.h_pad - hf) // self.stride + 1
        w_new = (wX + 2*self.w_pad - wf) // self.stride + 1
        Z = np.zeros((nX, nf, h_new, w_new))
        for n in range(nX):
            for c in range(nf):
                for h in range(h_new):
                    for w in range(w_new):
                        h_start = h * self.stride
                        h_end = h_start + hf
                        w_start = w * self.stride
                        w_end = w_start + wf
                        Z[n, c, h, w] = np.sum(self.X_pad[n, :, h_start:h_end, w_start:w_end] * self.w[c]) + self.b[c]  
        if self.active == "relu":
            self.mask = (Z > 0)
            return Z * self.mask

    def backward(self, dZ, optimizer):
        if self.active == "relu":
            dZ = dZ * self.mask
        ndZ, cdZ, hdZ, wdZ = dZ.shape
        _, _, hf, wf = self.w.shape

        dw = np.zeros_like(self.w)
        db = np.zeros_like(self.b)
        dX = np.zeros_like(self.X)
        dX_pad = np.zeros_like(self.X_pad)
        for n in range(ndZ):
            x_pad = self.X_pad[n]
            dx_pad = dX_pad[n]
            for c in range(cdZ):
                for h in range(hdZ):
                    for w in range(wdZ):
                        h_start = h * self.stride
                        h_end = h_start + hf
                        w_start = w * self.stride
                        w_end = w_start + wf
                        dx_pad[:, h_start:h_end, w_start:w_end] += self.w[c] * dZ[n, c, h, w]
                        dw[c] += x_pad[:, h_start:h_end, w_start:w_end] * dZ[n, c, h, w]
                        db[c] += dZ[n, c, h, w]
            dX[n] = dx_pad[:, self.h_pad:-self.h_pad, self.w_pad:-self.w_pad]  
        optimizer.update(self, dw, db)       
        return dX

    def get_weight(self):
        return self.w