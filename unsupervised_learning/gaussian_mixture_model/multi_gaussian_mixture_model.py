#%%
import numpy as np
import tensorflow as tf 
import tensorflow_probability as tfp
from tqdm import tqdm 
import matplotlib.pyplot as plt 

#%% 
def em(dataset, n_clusters, n_iter = 100):
    #infer from the dataset
    n_samples, n_dims = dataset.shape
    
    #draw initial guesses
    cluster_probs = tfp.distributions.Dirichlet(tf.ones(n_clusters)).sample(seed = 42)
    mus = tfp.distributions.Normal(loc=0.0, scale = 3.0).sample((n_clusters, n_dims), seed = 42)
    covs = tfp.distributions.WishartTriL(df = 3, scale_tril=tf.eye(n_dims)).sample(n_clusters, seed = 42)

    for _ in tqdm(range(n_iter)):
        # batched cholesky factorization
        
        ls = tf.linalg.cholesky(covs).numpy()
        normals = tfp.distributions.MultivariateNormalTriL(
            loc = mus,
            scale_tril = ls
        )
        ### E - step
        # (1) resp is of shape (n_samples x n_clusters)
        # batched multicariate normal is of shape (n_clusters x n_dims)
        unnormalized_responsibilities = (
            tf.reshape(cluster_probs, (1, n_clusters)) * normals.prob(tf.reshape(dataset, (n_samples, 1, n_dims)).numpy())
        )
    
        #normals.prob(tf.reshape(dataset, (n_samples, 1, n_dims)).numpy())
        # (2) 
        responsibilities = unnormalized_responsibilities/tf.reduce_sum(unnormalized_responsibilities, axis = 1, keepdims = True)
        responsibilities = tf.cast(responsibilities, "float32")
        # (3)
        class_responsibilities = tf.reduce_sum(responsibilities, axis = 0)
        class_responsibilities = tf.cast(class_responsibilities, "float32")
        ### M - step
        #(1)
        cluster_probs = class_responsibilities/n_samples 
        cluster_probs = tf.cast(cluster_probs, "float32")
        #(2)
        # class_responsibilities is of shape (n_clusters)
        # responsibilities is of shape (n_samples, n_clusters)
        # dataset is of shape (n_samples, n_dims)
        # mus is of shape (n_clusters, n_dims)
        # summation has to occur over the sample axis

        mus = tf.reduce_sum(
            tf.reshape(tf.cast(responsibilities, "float32"), (n_samples, n_clusters, 1))*tf.reshape(tf.cast(dataset, "float32"), (n_samples, 1, n_dims)),
            axis = 0
            )/tf.reshape(class_responsibilities, (n_clusters, 1))
        mus = tf.cast(mus, "float32")
        # (3)
        # class_responsibilities is of shape (n_clusters)
        # dataset is of shape (n_samples, n_dims)
        # mus is of shape (n_clusters, n_dims)
        # responsibilities is of shape (n_samples, n_clusters)
        
        # covs is of shape (n_clusters, n_dims, n_dims)
        # (n_clusters, n_samples, n_dims)
        centered_datasets = tf.reshape(tf.cast(dataset, "float64"), (1, n_samples, n_dims))  - tf.reshape(tf.cast(mus, "float64") , (n_clusters, 1, n_dims))
        #continue
        centered_dataset_with_responsibilities = centered_datasets * tf.reshape(tf.cast(tf.transpose(responsibilities), dtype="float64"), (n_clusters, n_samples, 1))
        #batched matrix multiplication
        #(n_clusters, n_dims, n_dims)
        sample_covs = tf.matmul(centered_datasets, centered_dataset_with_responsibilities, transpose_a = True)
        sample_covs = tf.cast(sample_covs, "float32")
        covs = sample_covs / tf.reshape(class_responsibilities, (n_clusters, 1, 1))
        #ensure positive definiteness by adding a "small amount" to the diagonal 
        covs = covs + 1.0e-8 * tf.eye(n_dims, batch_shape=(n_clusters, ))
        
    return cluster_probs, mus, covs

#%%
def main():

    n_clusters = 2
    cluster_probs = [0.3, 0.7]

    mus_true = np.array([
                         [ 5.0,  5.0],
                         [-3.0, -2.0]
                        ])

    covs_true = np.array(
                    [
                        [
                            [1.5, 0.5],
                            [0.5, 2.0]
                        ],
                        [
                            [1.5, 0.0],
                            [0.0, 1.8]
                        ]
                    ]
                )
   
    n_samples = 5000

    #batched cholesky factorization of the covariance matrices
    ls_true = tf.linalg.cholesky(covs_true)

    # the true guassian mixture model (we want to use for sampling some 
    # some artificial data)
    cat = tfp.distributions.Categorical(
        probs = cluster_probs,
    )
    
    m_normals = tfp.distributions.MultivariateNormalTriL(
        loc = mus_true,
        scale_tril = ls_true,
    )
    
    tfd = tfp.distributions
    gmm_true = tfd.MixtureSameFamily(
        mixture_distribution=cat, 
        components_distribution=m_normals,
    )

    data = gmm_true.sample(n_samples, seed = 42)
    #data = tf.cast(data, "float64")
    #plt.scatter(data.numpy()[:,0],
    #            data.numpy()[:,1])
    #plt.show()
    
    #return 
    class_probs_approx, mus_approx, covs_approx = em(data, n_clusters, 300)
    
    print("=======")
    print("class probabilities:")
    print(class_probs_approx)
    print("=======")
    print("mus:")
    print(mus_approx)
    print("=======")
    print("covariance matrix:")
    print(covs_approx)
    
if __name__ == "__main__":
    main()

# %%
