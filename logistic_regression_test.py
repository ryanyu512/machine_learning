'''
This file aims to test the functions in logistic_regression class

update: 2022/10/29
'''

#%%
import numpy as np
import matplotlib.pyplot as plt
from visualization.visualize import plot_decision_regions
from sampling.sampling import gaussian_sample_data
from supervised_learning.mlp.mlp import logistic_regression
#%%
'''
purpose: sampling two classes based on gaussian_sample_data
'''

#define means of two clusters
mu1 = [-50, -50]
mu2 = [-30, -30]

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
labels[0:int(sample.shape[0]/2)] = 0
labels[int(sample.shape[0]/2):sample.shape[0]] = 1
print("shape of sample: {}".format(sample.shape))
print("shape of labels: {}".format(labels.shape))
print("# of  0s: {}".format(len(labels[labels ==  0])))
print("# of  1s: {}".format(len(labels[labels ==  1])))

#%% 
'''
purpose: plot two classes for checking correctness
'''

plt.scatter(sample1[:,0], sample1[:,1], c = 'r')
plt.scatter(sample2[:,0], sample2[:,1], c = 'g')
plt.legend(['class 0', 'class 1'])
plt.show()

# %%
'''
purpose: 
1) define hyper - parameters of logistic regression
2) train the logistic regression based on batch training
3) evaluate the training results
'''
#define learning rate
lr = 1e-4
#define number of iterations
N_iter = 10000
#train logistic regression
net = logistic_regression(lr, N_iter)
net.fit(sample, labels, 'batch')
#evaluate
print("weightings after training: {}".format(net.w))
plt.plot(net.loss)
plt.xlabel("number of iterations")
plt.ylabel("loss")

#%%
threshold = 0.5
plot_decision_regions(sample, labels, net, threshold)

# %%
'''
purpose: 
1) define hyper - parameters of logistic regression
2) train the logistic regression based on sgd training
3) evaluate the training results
'''
#define learning rate
lr = 1e-2
#define number of iterations
N_iter = 10000
#train logistic regression
net_sgd = logistic_regression(lr, N_iter)
net_sgd.fit(sample, labels)
#evaluate
print("weightings after training: {}".format(net_sgd.w))
plt.xlabel("number of iterations")
plt.ylabel("loss")

threshold = 0.5
plot_decision_regions(sample, labels, net_sgd, threshold)
# %%
