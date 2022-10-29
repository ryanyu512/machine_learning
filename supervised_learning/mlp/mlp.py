'''
mainly for neural network related machine learning algorithms

update: 2022/10/29
'''

import numpy as np
from scipy.special import expit

class logistic_regression(object):
    
    def __init__(self, lr, N_iter):
        #learning rate
        self.lr = lr
        #number of iterations
        self.N_iter = N_iter
        #weights 
        self.w = None
        #loss 
        self.loss = None
        
    def predict(self, X):
        '''
        purpose: predict the probability of class 1 
        X    (in): data for training
        est (out): probability of class 1
        '''
        if self.w is None:
            self.w = np.random.uniform(size = X.shape[1] + 1)
        z = X.dot(self.w[0:-1]) + self.w[-1]
        #est = 1/(1 + np.exp(-z))
        #return est
        return expit(z)
        
    def batch_gradient_and_loss(self, X, Y):
        '''
        purpose: compute batch gradients of weights + bias and cross entropy loss
        X       (in): data for training
        Y       (in): labels for training
        grad_w (out): gradients of weights
        grad_b (out): gradients of bias
        sq_loss(out): squared loss 
        '''
        grad_w = 0
        grad_b = 0
        loss = 0
        for i in range(X.shape[0]):
            est = self.predict(X[i])
            e = est - Y[i] 
            loss += -Y[i]*np.log(est) - (1 - Y[i])*np.log(1 - est)
            grad_b += e
            grad_w += e*X[i]
        loss = loss/X.shape[0]
        return grad_w, grad_b, loss
    
    def gradient(self, X, Y):
        '''
        purpose: compute gradients of weights + bias
        X       (in): data for training
        Y       (in): labels for training
        grad_w (out): gradients of weights
        grad_b (out): gradients of bias
        '''
        est = self.predict(X)
        e = est - Y
        grad_b = e
        grad_w = e*X
        
        return grad_w, grad_b
    
    def fit(self, X, Y, opt = None):
        '''
        purpose: optimize weights and bias based on training data
        X       (in): data for training
        Y       (in): labels for training
        opt     (in): choice of batch training or sgd training
        '''
        self.loss = np.zeros(self.N_iter)
        self.w = np.random.uniform(size = X.shape[1] + 1)
        for i in range(self.N_iter):
            if opt == 'batch' or opt is None:
                grad_w, grad_b, self.loss[i] = self.batch_gradient_and_loss(X, Y)
            elif opt == 'sgd':
                ind = np.random.uniform(low = 0.0, high = X.shape[0], size = 1)
                ind = int(np.floor(ind))
                grad_w, grad_b = self.gradient(X[ind, :], Y[ind])
            self.w[-1] = self.w[-1] - self.lr*grad_b
            self.w[0:-1] = self.w[0:-1] - self.lr*grad_w
        
