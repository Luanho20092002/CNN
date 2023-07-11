import numpy as np

class Sequential:

    def __init__(self, *args) -> None:
        self.layer = args

    def fit(self, Xtrain, ytrain, batch_size=1, max_epoch=10, eta=1, validation=(None, None)) -> None:
        N = Xtrain.shape[0]
        epoch = 0
        while epoch < max_epoch: # epoch
            for b in range(0, N, batch_size): # batch
                X = Xtrain[b:b+batch_size]
                y = ytrain[b:b+batch_size]
                #Forward
                for i in range(len(self.layer)):
                    X = self.layer[i].forward(X) 
                #Backpropagation & update weight
                dLdZ = self.layer[-1].dLdZ(y, eta, batch_size)
                for i in reversed(range(len(self.layer[:-1]))):
                    dLdZ = self.layer[i].backward(dLdZ, eta)
            print("Loss:", self.cost(Xtrain, ytrain))
            epoch += 1
            
    def add(self, l):
        self.layer = self.layer + (l,)

    def cost(self, X, y):
        N = ytest.shape[0]
        for layer in self.layer:
            X = layer.forward(X)
        return -1/N * np.sum(y * np.log(X))
        
    def predict(self, X):
        for md in self.layer:
            X = md.forward(X)
        rs = np.argmax(X, axis=1)
        return rs
    
    def evaluate(self, Xtest, ytest):
        rs = self.predict(Xtest)
        ytest = np.argmax(ytest, axis=1)
        score = np.mean(rs == ytest)
        return score
      
