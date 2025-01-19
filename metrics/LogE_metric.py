import numpy as np
from scipy.linalg import logm

# Log-Euclidean Distance
def LogE_dist(A, B):
    """
    Computes the Log-Euclidean distance between two matrices A and B.

    Parameters:
    A, B (array): Positive-definite matrices.

    Returns:
    distance (float): Log-Euclidean distance.
    """
    log_A = logm(A)
    log_B = logm(B)
    distance = np.linalg.norm(log_A - log_B, ord='fro')
    return distance

# Log-Euclidean Linear Kernel
def loge_linear_kernel(X, Y=None):
    """
    Computes the linear kernel between matrices in the log-Euclidean space.

    Parameters:
    X (array): An array (n, p, p).
    Y (array, optional): An optional array (m, p, p).
    
    Returns:
    K: A kernel matrix.
    """
    if Y is None:
        Y = X  # If no Y is provided, compute the kernel between matrices in X

    # Compute the matrix logarithms of X and Y
    log_X = np.array([logm(X[i]) for i in range(X.shape[0])])  # Shape: (n, p, p)
    log_Y = np.array([logm(Y[j]) for j in range(Y.shape[0])])  # Shape: (m, p, p)

    n, m = log_X.shape[0], log_Y.shape[0]
    K = np.zeros((n, m))

    # Compute the kernel matrix
    for i in range(n):
        for j in range(m):
            K[i, j] = np.trace(log_X[i].T @ log_Y[j])  # Frobenius inner product

    return K

# Log-Euclidean Polynomial Kernel
def loge_polynomial_kernel(X, Y=None, c=0, d=1):
    """
    Computes the polynomial kernel between two sets of matrices using the Log-Euclidean metric.

    Parameters:
    X (array): Array of shape (n, p, p), positive-definite matrices.
    Y (array, optional): Array of shape (m, p, p), positive-definite matrices.
    c (float): A constant added to the kernel values before raising to the power d.
    d (float): The degree of the polynomial kernel.

    Returns:
    K: Kernel matrix.
    """
    if Y is None:
        Y = X  # If no Y is provided, compute the kernel between matrices in X

    # Compute the matrix logarithms of X and Y
    log_X = np.array([logm(X[i]) for i in range(X.shape[0])])  # Shape: (n, p, p)
    log_Y = np.array([logm(Y[j]) for j in range(Y.shape[0])])  # Shape: (m, p, p)

    n, m = log_X.shape[0], log_Y.shape[0]
    K = np.zeros((n, m))

    # Compute the kernel matrix
    for i in range(n):
        for j in range(m):
            K[i, j] = np.trace(log_X[i].T @ log_Y[j])  # Frobenius inner product

    return (K + c)**d

# Log-Euclidean Gaussian Kernel
def loge_gaussian_kernel(X, Y=None, sigma=1):
    """
    Computes the Gaussian kernel between two sets of matrices using the Log-Euclidean metric.

    Parameters:
    X (array): Array of shape (n, p, p).
    Y (array, optional): Array of shape (m, p, p). If None, Y=X.
    sigma (float): Scale parameter for the Gaussian kernel.

    Returns:
    K: A kernel matrix.
    """
    if Y is None:
        Y = X  # If no Y is provided, compute the kernel between matrices in X
      
    distances = LogE_dist_matrix(X, Y)
    K = np.exp(-distances**2 / (2 * sigma**2))
    return K

# Log-Euclidean Generalized Gaussian Kernel
def loge_gen_gaussian_kernel(X, Y=None, sigma=1, q=2):
    """
    Computes the generalized Gaussian kernel between two sets of matrices using the Log-Euclidean metric.

    Parameters:
    X (array): Array of shape (n, p, p).
    Y (ndarray, optional): Array of shape (m, p, p). If None, Y=X.
    sigma (float): Scale parameter for the Gaussian kernel.
    q (float): Exponent parameter for the generalized Gaussian kernel, 0<q<2.

    Returns:
    K: A kernel matrix.
    """
    if Y is None:
        Y = X  # If no Y is provided, compute the kernel between matrices in X
      
    distances = LogE_dist_matrix(X, Y)
    K = np.exp(-distances**q / (2 * sigma**2))
    return K
