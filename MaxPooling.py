import numpy as np

class MaxPooling:

    def __init__(self, kernel_size, stride=1) -> None:
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, Xin):
        self.Xin = Xin
        #Kiểm tra 4d
        if len(Xin.shape) != 4: print("Error!"); return
        w = Xin.shape[2] # Size ảnh gốc
        h = Xin.shape[3]
        w_new = int(((w - self.kernel_size)/self.stride) + 1)
        h_new = int(((h - self.kernel_size)/self.stride) + 1)

        Xout = np.zeros((Xin.shape[0], Xin.shape[1], w_new, h_new))
        for v in range(Xin.shape[0]):
            for c in range(Xin.shape[1]):
                for w in range(w_new):
                    for h in range(h_new):
                        w_start = w*self.stride
                        w_end = w_start + self.kernel_size
                        h_start = h*self.stride
                        h_end = h_start + self.kernel_size
                        slice_arr = Xin[v, c, w_start:w_end, h_start:h_end]
                        Xout[v, c, w, h] = np.max(slice_arr)
        return Xout

    def backward(self, dL, eta):
        dX = dL["dX"]
        stride = self.stride
        kernel = self.kernel_size
        dXconv = np.zeros(self.Xin.shape)
        
        for v in range(dX.shape[0]):
            for c in range(dX.shape[1]):
                for w in range(dX.shape[2]):
                    for h in range(dX.shape[3]):
                        #Tim index cua phan tu lon nhat trong khu vuc tich chap
                        a, b = self.nanargmax(self.Xin[v, c, w*stride:w*stride+kernel, h*stride:h*stride+kernel])
                        dXconv[v, c, w*stride+a, h*stride+b] = dX[v, c, w, h]
        return {"dX":dXconv}
    
    def nanargmax(self, arr):
        idx = np.nanargmax(arr)
        idxs = np.unravel_index(idx, arr.shape)
        return idxs 
   
