#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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
        new_w (out): updated weights of linear svm 
        '''
        
        self.w = self.w - self.lr*grad
    
    def fit(self, sample, labels):
        self.loss = np.zeros(self.N_iter)
        self.w = np.random.uniform(size = sample.shape[1])
        for i in range(self.N_iter):
            grad, self.loss[i] = self.gradient_and_loss(sample, labels)
            self.update_w(grad)
            
    def predict(self, sample):
        return sample.dot(self.w)

#%% 
def plot_decision_regions(sample, labels, clf, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(labels))])

    # plot the decision surface
    x_min, x_max = sample[:, 0].min() - 1, sample[:, 0].max() + 1
    y_min, y_max = sample[:, 1].min() - 1, sample[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x_min, x_max, resolution), np.arange(x_min, x_max, resolution))
    Z = clf.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = np.where(Z >= 0, 1, -1)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(labels)):
        plt.scatter(x=sample[labels == cl, 0], y=sample[labels == cl, 1],
        alpha=0.8, c=cmap(idx),
        marker=markers[idx], label=cl)

#%%
#purpose: sampling two classes based on gaussian_sample_data

#define means of two clusters
mu1 = [-10, -5]
mu2 = [ 5,  5]

#define covariance matrix
#eig values affects the spread of data along axis i
eig_val1 = np.sqrt(1)
eig_val2 = np.sqrt(5)

#eig vectors represents the orientation of data distribution
dir1 = np.array([5,  2])
dir2 = np.array([2, -5])

cov1 = eig_val1*np.outer(dir1, dir1) + \
       eig_val2*np.outer(dir2, dir2)

cov2 = cov1 

sample1 = gaussian_sample_data(mu1, cov1, 100)
sample2 = gaussian_sample_data(mu2, cov2, 100)

#integrate two class
sample = np.concatenate((sample1, sample2), axis = 0)
labels = np.zeros(sample.shape[0])
labels[0:int(sample.shape[0]/2)] = -1
labels[int(sample.shape[0]/2):sample.shape[0]] = 1
print("shape of sample: {}".format(sample.shape))
print("shape of labels: {}".format(labels.shape))
print("# of -1s: {}".format(len(labels[labels == -1])))
print("# of  1s: {}".format(len(labels[labels ==  1])))
#%% 
#purpose: plot two classes for checking correctness

plt.scatter(sample1[:,0], sample1[:,1], c = 'r')
plt.scatter(sample2[:,0], sample2[:,1], c = 'g')
plt.show()
# %% 
# define hyper - parameters of linear SVM 
# number of iterations
N_iter = 1000
#learning rate 
lr = 0.0001
l  = 1
# %%
# fit linear svm
lin_svm = linear_svm(lr, l, N_iter)
lin_svm.fit(sample, labels)

# %%
# plot loss curve 
plt.plot(lin_svm.loss)

# %%
plot_decision_regions(sample, labels, lin_svm)
# %%
