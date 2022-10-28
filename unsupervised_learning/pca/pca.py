#%% import modules
import numpy as np
import matplotlib.pylab as plt
from numpy.random import default_rng

#%%
'''
purpose     : obtain data with zero mean
data    (in): data to be offset
c_data (out): data with zero mean
'''
def center_data(data):
    data_mean = np.mean(data, axis = 0)
    c_data = data - data_mean 
    
    return c_data

'''
purpose     : estimate covariance matrix
data    (in): data distribution
cov    (out): covariance matrix of input data distribution
'''
def est_cov(data):
    offset_data = center_data(data)   
    
    rs, cs = data.shape
    cov = np.zeros((cs, cs))

    for x in offset_data:
        cov += np.outer(x, x)
    cov = cov/rs    
    
    return cov

'''
purpose      : estimate eigen values and eigen vectors
cov      (in): covariance matrix
eig_vec (out): matrix of eigen vectors
eig_val (out): eigen values
'''
def est_eig(cov):
    eig_vec, eig_val, _ = np.linalg.svd(cov)

    return eig_vec, eig_val

'''
purpose: sort the column vector of eig_vec based on magnitude of eig_val
eig_vec    (in): matrix of eigen vectors
eig_val    (in): eig values
s_eig_vec (out): sorted matrix of eigen vectors based on 
magnitude of eigen values
s_ind     (out): sorted column index of eig_vec
'''
def sort_eig_vec(eig_vec, eig_val):
    
    s_ind = np.argsort(-eig_val)
    s_eig_vec = eig_vec[:,s_ind]
    
    return s_eig_vec, s_ind

'''
purpose: reduce dimension of data 
data         (in): data to be transformed
k            (in): number of features to be extracted
reduced_data(out): data after dimension reduction
'''
def transform_data(data, k):
    '''
    #procedures of pca
    #step 1: center data 
    #step 2: compute covariance matrix of centered data
    #step 3: compute eigen vectors and eigen values of covariance matrix
             by SVD
    #step 4: sort eigen vectors based on magnitude of eigen values 
             (from larger to smaller)
    #step 5: select top k eigen vectors as principle components to represent 
             data in a lower - dimensional space
    #step 6: transform data using top k eigen vectors 
    '''
    
    cov = est_cov(data)
    eig_vec, eig_val = est_eig(cov)
    s_eig_vec, s_ind = sort_eig_vec(eig_vec, eig_val)
    reduced_data = center_data(data).dot(s_eig_vec[:,0:k])
    
    return reduced_data

'''
purpose: sampling data from gaussian distribution
mu   (in): mean of gaussian distribution
cov  (in): covariance matrix of multi-variate gaussian distribution
data(out): sampling data from gaussian distribution
'''
def gaussian_sample_data(mu, cov, N = 1000):
    #initialize random number generator
    rng = default_rng()
    #sampling data based on data distribution 
    data = rng.multivariate_normal(mu, cov, size = N)

    return data

#%%
#purpose: sampling data from multi-gaussian distribution

#define center of gaussian
mu = np.array([2, 2])

#define covariance matrix
#eig values affects the spread of data along axis i
eig_val1 = np.sqrt(5)
eig_val2 = np.sqrt(1)

#eig vectors represents the orientation of data distribution
dir1 = np.array([5,  2])
dir2 = np.array([2, -5])

cov = eig_val1*np.outer(dir1, dir1) + \
      eig_val2*np.outer(dir2, dir2)

#sampling data
data = gaussian_sample_data(mu, cov, 1000)
print(data.shape)

plt.scatter(data[:,0], data[:,1], c = 'r')
plt.show()

#%%
#purpose: reduce features
#define the number of features to be extracted
#k <= data.shape[1]
k = 2
reduced_data = transform_data(data, k)
print("shape of reduced data: {}".format(reduced_data.shape))

#%%
#purpose: plot data
plt.plot(reduced_data, '.r')
plt.show()
