from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

def plot_decision_regions(sample, labels, clf, threshold, resolution=0.02):
    '''
    purpose    (in): plot decision bounary
    sample     (in): data to be plotted
    labels     (in): the label of data
    clf        (in): classifier object
    resolution (in): the resolution of meshgrid 
    '''
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(labels))])

    # plot the decision surface
    x_min, x_max = sample[:, 0].min() - 1, sample[:, 0].max() + 1
    y_min, y_max = sample[:, 1].min() - 1, sample[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x_min, x_max, resolution), np.arange(x_min, x_max, resolution))
    Z = clf.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = np.where(Z >= threshold, 1, 0)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(labels)):
        plt.scatter(x=sample[labels == cl, 0], y=sample[labels == cl, 1],
        alpha=0.8, c=cmap(idx),
        marker=markers[idx], label=cl)

