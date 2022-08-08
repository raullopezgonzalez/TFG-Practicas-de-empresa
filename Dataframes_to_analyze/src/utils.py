# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse

# sklearn
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.metrics import silhouette_samples, silhouette_score


def load_examples():
    # case 1: general case
    X1, y1 = make_blobs(random_state=1)

    # case 2: clusters are of different densities and sizes
    X2, y2 = make_blobs(n_samples=200, cluster_std=[1.0, 2.5, 0.5], random_state=170)

    # case 3: Anisotropic distribution
    X3, y3 = make_blobs(random_state=170, n_samples=600)
    rng = np.random.RandomState(74)
    transformation = rng.normal(size=(2, 2))
    X3 = np.dot(X3, transformation)

    # case 4: two_moons data 
    X4, y4 = make_moons(n_samples=200, noise=0.05, random_state=0)
    
    return X1, y1, X2, y2, X3, y3, X4, y4

def plot_scatter(X,title, cluster_labels = None, centroids = None):
    
    eps = X.std() / 5.

    x_min, x_max = X[:, 0].min() - eps, X[:, 0].max() + eps
    y_min, y_max = X[:, 1].min() - eps, X[:, 1].max() + eps
    
    if cluster_labels is not None:
        plt.scatter(X[:, 0], X[:, 1], s = 100, alpha = 0.5, c = cluster_labels)
        if centroids is not None:
            plt.scatter(centroids[:,0],centroids[:,1],marker = '^', c = 'r', s = 200)
    else:    
        plt.scatter(X[:, 0], X[:, 1], s = 100, alpha = 0.5)
        
    plt.xlabel('$x_1$',fontsize=16)
    plt.ylabel('$x_2$',fontsize=16)
    plt.axis('equal')
    #plt.axis([x_min, x_max, y_min, y_max])
    #ax = plt.gca()
    #ax.set_aspect('equal', 'box')
    plt.xticks([])
    plt.yticks([])
    plt.title(title)



def plot_silhouette(X, k, cluster_labels, centroids):
    
    # silhouette_score
    silhouette_avg = silhouette_score(X, cluster_labels)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    #--- Do the ploting
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (k + 1) * 10])

    y_lower = 10
    for i in range(k):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / k)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    colors = cm.nipy_spectral(cluster_labels.astype(float) / k)
    ax2.scatter(X[:, 0], X[:, 1], marker='o', s=100, alpha=0.5, c=colors)
    ax2.scatter(centroids[:,0],centroids[:,1],marker = 'x', c = 'k', s = 200)
    ax2.set_xlabel('$x_1$',fontsize=16)
    ax2.set_ylabel('$x_2$',fontsize=16)

    plt.show()
    
#https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html
def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    elif covariance.shape == (2,): # this was added from the original code
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    else: 
        angle = 0
        width = 2*covariance 
        height= 2*covariance
        
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
        
def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)