import numpy as np

class Flatten:

    def forward(self, Xin):
        self.Xin = Xin
        if len(Xin.shape) != 4: print("Error!"); return
        out = Xin.reshape(Xin.shape[0], Xin.shape[1] * Xin.shape[2] * Xin.shape[3])
        return out

    def backward(self, dL, eta):
        w_next = dL["w_next"]
        dLdZ = dL["dLdZ"]
        dflat = dLdZ.dot(w_next)
        dXpool = np.array(dflat).reshape(self.Xin.shape)
        return {"dX":dXpool}
