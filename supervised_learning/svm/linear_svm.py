#%%
import numpy as np
from numpy.random import default_rng

#%% 
def gaussian_sample_data(mu, cov, N = 1000):
    
    '''
    purpose: sampling data from gaussian distribution
    mu   (in): mean of gaussian distribution
    cov  (in): covariance matrix of multi-variate gaussian distribution
    data(out): sampling data from gaussian distribution
    '''
    
    #initialize random number generator
    rng = default_rng()
    #sampling data based on data distribution 
    data = rng.multivariate_normal(mu, cov, size = N)

    return data

#%%
class linear_svm(object):
    def __init__(self, lr, l, N_iter):
        #learning rate
        self.lr = lr
        #number of iterations
        self.N_iter = N_iter
        #regularization factor
        self.l = l
        #weights
        self.w = None
        #loss value
        self.loss = None

        
    def gradient_and_loss(self, X, y):
        '''
        purpose: compute gradient and loss for weight optimization
        X    (in): matrix of features in shape of N (# of samples) x M (# of features)
        y    (in): labels of sample
        grad(out): steepest gradient of linear svm
        loss(out): the loss value of one epoch 
        
        #please note that hinge loss is used as loss function
        '''
        
        grad = 0
        loss = 0
        rs, cs = X.shape
        for i in range(rs):
            est = y[i]*(X[i,:].dot(self.w))
            if est < 1:
                grad += -y[i]*np.transpose(X[i,:])
                loss += (1 - est)
        
        grad += 2*self.l*self.w
        loss = loss/rs
        return grad, loss

    def update_w(self, grad):
        '''
        purpose: update weight based on gradient from steepest gradient
        w      (in): matrix of features in shape of N (# of samples) x M (# of features)
        grad   (in): steepest gradient of linear svm
        '''
        
        self.w = self.w - self.lr*grad
    
    def fit(self, sample, labels):
        '''
        purpose: fit training data
        sample (in): training data
        labels (in): training labels
        '''
        self.loss = np.zeros(self.N_iter)
        self.w = np.random.uniform(size = sample.shape[1])
        for i in range(self.N_iter):
            grad, self.loss[i] = self.gradient_and_loss(sample, labels)
            self.update_w(grad)
            
    def predict(self, sample):
        return sample.dot(self.w)
