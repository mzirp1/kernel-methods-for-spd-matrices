##### Euclidean Metric #####

import numpy as np

def linear_kernel(X, Y=None):
    """
    Compute the linear kernel between two sets of matrices.

    Parameters:
    - X: array of shape (n_samples_X, n, n), the first dataset (e.g., covariance matrices).
    - Y: array of shape (n_samples_Y, n, n), the second dataset. If None, Y = X.

    Returns:
    - K: array of shape (n_samples_X, n_samples_Y), the linear kernel matrix.
    """
    if Y is None:
        Y = X
    
    n_samples_X = X.shape[0]
    n_samples_Y = Y.shape[0]
    K = np.zeros((n_samples_X, n_samples_Y))
    
    for i in range(n_samples_X):
        for j in range(n_samples_Y):
            K[i, j] = np.trace(np.dot(X[i].T, Y[j]))
    
    return K

def polynomial_kernel(X, Y=None, c=0, d=1):
    """
    Compute the polynomial kernel between two sets of matrices.

    Parameters:
    - X: array of shape (n_samples_X, n, n), the first dataset (e.g., covariance matrices).
    - Y: array of shape (n_samples_Y, n, n), the second dataset. If None, Y = X.
    - c: float, the additive constant in the kernel formula (default: 0).
    - d: int, the degree of the polynomial (default: 1).

    Returns:
    - K: array of shape (n_samples_X, n_samples_Y), the polynomial kernel matrix.
    """
    if Y is None:
        Y = X
    
    n_samples_X = X.shape[0]
    n_samples_Y = Y.shape[0]
    K = np.zeros((n_samples_X, n_samples_Y))
    
    for i in range(n_samples_X):
        for j in range(n_samples_Y):
            K[i, j] = np.trace(np.dot(X[i].T, Y[j]))
    
    return (K + c) ** d


def gaussian_kernel(X, Y=None, sigma=1):
    """
    Compute the Gaussian kernel between sets of matrices based on Frobenius distance.

    Parameters:
    - X: array of shape (n_samples_X, n, n), the first dataset.
    - Y: array of shape (n_samples_Y, n, n), the second dataset. If None, Y = X.
    - sigma: float, the bandwidth of the Gaussian kernel.

    Returns:
    - K: array of shape (n_samples_X, n_samples_Y), the Gaussian kernel matrix.
    """
    if Y is None:
        Y = X
    
    n_samples_X = X.shape[0]
    n_samples_Y = Y.shape[0]
    K = np.zeros((n_samples_X, n_samples_Y))
    
    for i in range(n_samples_X):
        for j in range(n_samples_Y):
            # Compute Frobenius norm as the distance between matrices
            frobenius_distance = np.linalg.norm(X[i] - Y[j], ord='fro')
            # Apply the Gaussian kernel formula
            K[i, j] = np.exp(-frobenius_distance**2 / (2 * sigma**2))
    
    return K

def gen_gaussian_kernel(X, Y=None, sigma=1, q=2):
    """
    Compute a generalized Gaussian kernel between sets of matrices based on Frobenius distance.

    Parameters:
    - X: array of shape (n_samples_X, n, n), the first dataset.
    - Y: array of shape (n_samples_Y, n, n), the second dataset. If None, Y = X.
    - sigma: float, the bandwidth of the Gaussian kernel.
    - q: float, the power applied to the Frobenius distance (0<q<2) (default: 2).

    Returns:
    - K: array of shape (n_samples_X, n_samples_Y), the generalized Gaussian kernel matrix.
    """
    if Y is None:
        Y = X
    
    n_samples_X = X.shape[0]
    n_samples_Y = Y.shape[0]
    K = np.zeros((n_samples_X, n_samples_Y))
    
    for i in range(n_samples_X):
        for j in range(n_samples_Y):
            # Compute Frobenius norm as the distance between matrices
            frobenius_distance = np.linalg.norm(X[i] - Y[j], ord='fro')
            # Apply the generalized Gaussian kernel formula
            K[i, j] = np.exp(-frobenius_distance**q / (2 * sigma**2))
    
    return K
