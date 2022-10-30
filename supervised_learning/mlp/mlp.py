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
    
    def sigmoid(self, z):
        return expit(z)
    
    def predict(self, X):
        '''
        purpose: predict the probability of class 1 
        X    (in): data for training
        est (out): probability of class 1
        '''
        if self.w is None:
            self.w = np.random.uniform(size = X.shape[1] + 1)
        z = X.dot(self.w[0:-1]) + self.w[-1]
        
        return self.sigmoid(z)
        
    def batch_gradient_and_loss(self, X, Y):
        '''
        purpose: compute batch gradients of weights + bias and cross entropy loss
        X       (in): data for training
        Y       (in): labels for training
        grad_w (out): gradients of weights
        grad_b (out): gradients of bias
        sq_loss(out): squared loss 
        '''

        est = self.predict(X)
        e = est - Y
        grad_w = np.transpose(X).dot(e)
        grad_b = np.sum(e, axis = 0)
        loss = -np.transpose(Y).dot(np.log(est)) - np.transpose(1 - Y).dot(np.log(1 - est))
        loss = loss / X.shape[0]

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

class multi_logistic_regression(object):
    def __init__(self, lr, N_iter):
        #learning rate
        self.lr = lr
        #number of iterations
        self.N_iter = N_iter
        #weights 
        self.w = None
        #loss 
        self.loss = None
        #number of classes
        self.N_class = None
    
    def softmax(self, z):
        '''
        purpose: compute non-linear result from softmax function
        z       (in): linear outputs (N, C)
        est    (out): non - linear outputs (N, C)
        '''
        est = np.exp(z) # (N x C)
        s_est = np.sum(est, axis = 1, keepdims = True)
        est = est/s_est
        
        return est
      
    def predict(self, X):
        '''
        purpose: predict the probability of class 1 
        X      (in): data for training      (N, D)
        est   (out): probability of classes (N, C)
        '''
        if self.w is None:
            self.w = np.random.uniform(size = (X.shape[1] + 1)*self.N_class)
            self.w.shape = (X.shape[1] + 1, self.N_class)
        
        z = X.dot(self.w[0:-1,:]) + self.w[-1,:]
        
        return self.softmax(z) 
    
    def batch_gradient_and_loss(self, X, Y):
        '''
        purpose: compute batch gradients of weights + bias and cross entropy loss
        X       (in): data for training    (N, D)
        Y       (in): labels for training  (N, C)
        grad_w (out): gradients of weights (D, C)
        grad_b (out): gradients of bias    (1, C)
        sq_loss(out): squared loss 
        '''

        est = self.predict(X) 
        e = est - Y
        grad_w = np.transpose(X).dot(e)
        grad_b = np.sum(e, axis = 0)
        loss = np.sum(np.square(np.sum(e, axis = 1)), axis = 0)
        loss = loss / X.shape[0]
        
        return grad_w, grad_b, loss
    
    def fit(self, X, Y, opt = None):
        '''
        purpose: optimize weights and bias based on training data
        X       (in): data for training
        Y       (in): labels for training
        opt     (in): choice of batch training or sgd training
        '''
        self.N_class = len(np.unique(Y))
        self.loss = np.zeros(self.N_iter)
        self.w = np.random.uniform(size = (X.shape[1] + 1)*self.N_class)
        self.w.shape = (X.shape[1] + 1, self.N_class)
        
        Y_one_shot = np.zeros((Y.shape[0], self.N_class))
        for i in range(Y_one_shot.shape[0]):
            ind = int(Y[i])
            Y_one_shot[i, ind] = 1
        
        for i in range(self.N_iter):
            grad_w, grad_b, self.loss[i] = self.batch_gradient_and_loss(X, Y_one_shot)
            self.w[0:-1, :] = self.w[0:-1, :] - self.lr*grad_w
            self.w[  -1, :] = self.w[  -1, :] - self.lr*grad_b