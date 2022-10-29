import numpy as np

class logistic_regression(object):
    
    def __init__(self, lr, N_iter):
        self.lr = lr
        self.N_iter = N_iter
        self.w = None
        self.loss = None
        
    def predict(self, X):
        if self.w is None:
            self.w = np.random.uniform(size = X.shape[1] + 1)
        z = X.dot(self.w[0:-1]) + self.w[-1]
        return 1/(1 + np.exp(-z))
    
    def gradient_and_sqloss(self, X, Y):
        
        grad_w = 0
        grad_b = 0
        sq_loss = 0
        for i in range(X.shape[0]):
            est = self.predict(X[i])
            e = est - Y[i] 
            sq_loss += e**2
            grad_b += e
            grad_w += e*X[i]
        sq_loss = sq_loss/X.shape[0]
        return grad_w, grad_b, sq_loss
    
    def gradient(self, X, Y):
        
        est = self.predict(X)
        e = est - Y
        grad_b = e
        grad_w = e*X
        
        return grad_w, grad_b
    
    def fit(self, X, Y, opt = None):
        self.loss = np.zeros(self.N_iter)
        self.w = np.random.uniform(size = X.shape[1] + 1)
        for i in range(self.N_iter):
            if opt == 'batch' or opt is None:
                grad_w, grad_b, self.loss[i] = self.gradient_and_sqloss(X, Y)
            elif opt == 'sgd':
                ind = np.random.uniform(low = 0.0, high = X.shape[0], size = 1)
                ind = int(np.floor(ind))
                grad_w, grad_b = self.gradient(X[ind, :], Y[ind])
            self.w[-1] = self.w[-1] - self.lr*grad_b
            self.w[0:-1] = self.w[0:-1] - self.lr*grad_w
        
