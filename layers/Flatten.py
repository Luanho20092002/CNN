class Flatten:

    def forward(self, X):
        self.X = X
        output_shape = (X.shape[0], -1)
        return X.reshape(output_shape)

    def backward(self, dZ, optimizer):
        dX = dZ.reshape(self.X.shape)
        return dX
    
