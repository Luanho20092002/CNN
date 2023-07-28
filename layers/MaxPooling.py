import numpy as np

class MaxPooling:

    def __init__(self, kernel_size=2, stride=2) -> None:
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, X):
        self.X = X
        w = X.shape[2] # Size ảnh gốc
        h = X.shape[3]
        w_new = int(((w - self.kernel_size)/self.stride) + 1)
        h_new = int(((h - self.kernel_size)/self.stride) + 1)
        Xout = np.zeros((X.shape[0], X.shape[1], w_new, h_new))
        for v in range(X.shape[0]):
            for c in range(X.shape[1]):
                for w in range(w_new):
                    for h in range(h_new):
                        w_start = w*self.stride
                        w_end = w_start + self.kernel_size
                        h_start = h*self.stride
                        h_end = h_start + self.kernel_size
                        slice_arr = X[v, c, w_start:w_end, h_start:h_end]
                        Xout[v, c, w, h] = np.max(slice_arr)
        return Xout

    def backward(self, dZ, optimizer):
        stride = self.stride
        kernel = self.kernel_size
        dX = np.zeros(self.X.shape)
        
        for v in range(dZ.shape[0]):
            for c in range(dZ.shape[1]):
                for w in range(dZ.shape[2]):
                    for h in range(dZ.shape[3]):
                        #Tim index cua phan tu lon nhat trong khu vuc tich chap
                        a, b = self.nanargmax(self.X[v, c, w*stride:w*stride+kernel, h*stride:h*stride+kernel])
                        dX[v, c, w*stride+a, h*stride+b] = dZ[v, c, w, h]
        return dX
    
    def nanargmax(self, arr):
        idx = np.nanargmax(arr)
        idxs = np.unravel_index(idx, arr.shape)
        return idxs 


