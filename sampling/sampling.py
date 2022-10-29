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

# %%
