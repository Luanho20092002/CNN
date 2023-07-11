import numpy as np

class Conv2D:

    def __init__(self, kernel_size, filter=1, pad=0, stride=1, b=0) -> None:
        self.filter = filter
        self.kernel_size = kernel_size
        self.pad = pad
        self.stride = stride
        self.isHasFilter = True
    
    def forward(self, Xin):
        # Chuyển sang dạng đầy đủ 4d
        if len(Xin.shape) == 2:
            Xin = Xin.reshape(1, 1, Xin.shape[0], Xin.shape[1])
        elif len(Xin.shape) == 3:
            Xin = Xin.reshape(Xin.shape[0], 1, Xin.shape[1], Xin.shape[2])
        self.Xin = Xin
        #Kiểm tra xem đã từng tạo filter chưa
        if self.isHasFilter:
            c = Xin.shape[1]
            self.F = 0.1* np.random.randn(self.filter, c, self.kernel_size, self.kernel_size) # Tạo bộ lọc F ngẫu nhiên
            self.b = 0.1* np.random.randn(self.filter, 1)
            self.isHasFilter = False
        Xout = self.conv2d(Xin, f=self.F, pad=self.pad, stride=self.stride, b=self.b)
        Xout = self.relu(Xout) 
        return Xout

    def backward(self, dL, eta):
        dXconv = dL["dX"]
        df = self.dFilter(Xin=self.Xin, dXconv=dXconv, pad=self.pad, stride=self.stride)
        self.F = self.F - eta*df
        dbias = np.sum(dXconv, axis=(0,2,3)).reshape(-1, 1)
        self.b = self.b - eta*dbias
        flip_F = self.flip(self.F)
        dXin = self.conv2d(Xin=dXconv, f=flip_F, pad=self.pad, stride=self.stride, backward=True)
        dXin[dXin<0] = 0
        return {"dX": dXin}

    def conv2d(self, Xin, f, pad=0, stride=1, b=[0], backward=False):
        vol, channels, h, w = Xin.shape
        num, cha, h_f, w_f = f.shape
        w_new = int(((w + 2*pad - w_f)/stride)+1)
        h_new = int(((h + 2*pad - h_f)/stride)+1)

        Xout = np.zeros((vol, num, h_new, w_new))
        for v in range(vol):
            pad_arr = np.pad(Xin[v, :], ((0,0), (pad,pad), (pad,pad)), constant_values=0)
            for c in range(num):
                for h in range(h_new):
                    for w in range(w_new):
                        h_start = h*stride
                        h_end = h_start + h_f
                        w_start = w*stride
                        w_end = w_start + w_f
                        Xout[v, c, h, w] = np.sum(pad_arr[:, h_start:h_end, w_start:w_end] * f[c]) + b[c]
        return Xout

    # Tính đạo hàm theo filter
    def dFilter(self, Xin, dXconv, pad=0, stride=1):
        volX, chaX, heiX, widX = Xin.shape
        voldX, chadX, heidX, widdX = dXconv.shape
        h_new = int(((heiX + 2*pad - heidX)/stride)+1)
        w_new = int(((widX + 2*pad - widdX)/stride)+1)

        out = np.zeros((chadX, chaX, h_new, w_new))
        for v in range(volX):
            rs = np.zeros_like(out)
            for cd in range(chadX):
                for c in range(chaX):
                    arr_pad = np.pad(Xin[v, c], ((1,1), (1, 1)))
                    #print(arr_pad.shape)
                    #print("dXconv: ("+str(v)+", "+str(cd)+") | Xin: ("+str(v)+", "+str(c)+")")
                    for h in range(h_new):
                        for w in range(w_new):
                            h_start = h*stride
                            h_end = h_start + heidX
                            w_start = w*stride
                            w_end = w_start + widdX
                            rs[cd, c, h, w] = np.sum(arr_pad[h_start:h_end, w_start:w_end] * dXconv[v, cd])    
            out = out + rs
        return out

    # Lật ma trận 180 độ theo chiều dọc, ngang
    # Ma trận đầu vào 4d (VOLUME, CHANNEL, HEIGHT, WIDTH) gồm có VOLUME ma trận 3 chiều
    # Trong mỗi ma trận 3 chiều, lấy lớp đầu tiên ra, xếp lại chúng thành 1 ma trận 3d mới (lặp lại cho đến hết ma trận 3d). Ta sẽ có CHANNEL ma trận 3d mới
    # Gộp tất cả ma trận 3d mới lại. Ta được 1 ma trận 4d mới (CHANNEL, VOLUME, HEIGHT, WIDTH)
    def flip(self, matrix):
        matrix = np.flip(matrix, axis=(3, 2))
        vol, cha, hei, wid = matrix.shape
        out = np.zeros((cha, vol, hei, wid))
        for c in range(cha):
            for v in range(vol):
                out[c, v] = matrix[v, c] 
        return out
    
    def relu(self, Z):
        return np.maximum(0, Z)
