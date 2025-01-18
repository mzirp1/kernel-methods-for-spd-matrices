import numpy as np
from scipy.linalg import sqrtm
from scipy.linalg import logm
from scipy.linalg import expm

# Bures-Wasserstein Distance
def BW_dist(A, B):
    """
    Computes the Bures-Wasserstein distance between two positive-definite matrices A and B.
    
    Parameters:
    A (array): A positive-definite matrix.
    B (array): A positive-definite matrix.
    
    Returns:
    output (float): The Bures-Wasserstein distance between A and B.
    """
    temp = np.real(sqrtm(np.dot(A, B)))
    output = np.real(np.sqrt((np.trace(A + B) - 2 * np.trace(temp)) + 0j))
    return output

# Bures-Wasserstein Logarithm Map
def BW_log(A, B):
    """
    Computes the Bures-Wasserstein logarithm between two positive-definite matrices A and B.

    Parameters:
    A (array): A positive-definite matrix.
    B (array): A positive-definite matrix.
    
    Returns:
    output (array): The matrix representing the logarithm the result of the exponential map.
    """
    temp1 = np.real(sqrtm(np.dot(A, B)))
    temp2 = np.real(sqrtm(np.dot(B, A)))
    output = temp1 + temp2 - 2 * A
    return output

# Bures-Wasserstein Exponential Map
def BW_exp(A, B):
    """
    Computes the Bures-Wasserstein exponential map that interpolates between two matrices A and B.
    
    Parameters:
    A (array): A positive-definite matrix.
    B (array): A matrix representing the tangent vector (logarithm) at A.
    
    Returns:
    output (array): A positive-definite matrix representing the result of the exponential map.
    """
    D, U = np.linalg.eigh(A)
    Xu = np.transpose(U).dot(B).dot(U)
    W = np.zeros((len(D), len(D)))
    for i in range(len(D)):
        for j in range(len(D)):
            W[i, j] = 1 / (D[i] + D[j])
    output = A + B + U.dot(W * Xu).dot(np.diag(D)).dot(W * Xu).dot(np.transpose(U))
    return output

# Bures-Wasserstein Projection Mean
def bw_projection_mean(x, eps=1e-6, verbose=False):
    """
    Computes the Bures-Wasserstein mean (barycenter) of a set of positive-definite matrices.

    Parameters:
    x (ndarray): A set of positive-definite matrices of shape (n, p, p).
    eps (float): The stopping criteria, where the iteration stops if the distance between successive means is less than eps.
    verbose (bool): If True, print the progress of the iterations.

    Returns:
    mean_new (array): The Bures-Wasserstein mean of the input matrices.
    """
    mean_new = np.mean(x, axis=0)
    dist_mean = 10  # Initial value larger than eps
    k = 0

    while dist_mean > eps:
        # SVD and symmetric part of the matrix
        U, s, VT = np.linalg.svd(mean_new, full_matrices=True)
        mean_old = (np.transpose(VT).dot(np.diag(s)).dot(VT) + np.transpose(np.transpose(VT).dot(np.diag(s)).dot(VT))) / 2

        mean_tangent = np.mean(np.array([BW_log(mean_old, x[i]) for i in range(x.shape[0])]), axis=0)
        mean_new = np.real(BW_exp(mean_old, mean_tangent))
        dist_mean = BW_dist(mean_new, mean_old)

        k += 1
        mean_old = mean_new

        if verbose:
            print(f'Iter {k}, dist_mean={round(dist_mean, 7)}')

    return mean_new

# Bures-Wasserstein Induced Linear Kernel
def bw_linear_kernel(X, Y=None):
    """
    Computes the linear kernel between two sets of covariance matrices induced by the Bures-Wasserstein metric.
    
    Parameters:
    X (array): A list of covariance matrices (N,p,p).
    Y (array, optional): A list of covariance matrices. Defaults to None.
    
    Returns:
    K: A kernel matrix (p,p).
    """
    # Use X as both sets if Y is not provided
    if Y is None:
        Y = X
    
    n = len(X)
    m = len(Y)

    # Compute C_ref (barycenter) for both sets
    C_ref_X = bw_projection_mean(X)
    C_ref_Y = bw_projection_mean(Y)

    # Initialize kernel matrix
    K = np.zeros((n, m))

    # Loop over pairs of covariance matrices from X and Y
    for i in range(n):
        for j in range(m):
            B_i = BW_log(X[i], C_ref_X)
            B_j = BW_log(Y[j], C_ref_Y)

            # Solve H_i and H_j
            H_i = np.linalg.inv(C_ref_X) @ B_i
            H_j = np.linalg.inv(C_ref_Y) @ B_j

            # Compute kernel as Re(tr(H_j @ C_ref @ H_i))
            K[i, j] = np.real(np.trace(H_j @ C_ref_Y @ H_i))

    return K

# Bures-Wasserstein Polynomial Kernel
def bw_polynomial_kernel(X, Y=None, c=0, d=1):
    """
    Computes a polynomial kernel between two sets of covariance matrices induced by the Bures-Wasserstein metric.
    
    Parameters:
    X (array): A list of covariance matrices (N,p,p).
    Y (array, optional): A list of covariance matrices. Defaults to None.
    c (float): A constant added to the kernel before raising it to the power d.
    d (float): The exponent applied to the kernel values.
    
    Returns:
    K: A kernel matrix (p,p).
    """
    # Use X as both sets if Y is not provided
    if Y is None:
        Y = X
    
    n = len(X)
    m = len(Y)

    # Compute C_ref (barycenter) for both sets
    C_ref_X = bw_projection_mean(X)
    C_ref_Y = bw_projection_mean(Y)

    # Initialize kernel matrix
    K = np.zeros((n, m))

    # Loop over pairs of covariance matrices from X and Y
    for i in range(n):
        for j in range(m):
            B_i = BW_log(X[i], C_ref_X)
            B_j = BW_log(Y[j], C_ref_Y)

            # Solve H_i and H_j
            H_i = np.linalg.inv(C_ref_X) @ B_i
            H_j = np.linalg.inv(C_ref_Y) @ B_j

            # Compute kernel as Re(tr(H_j @ C_ref @ H_i))
            K[i, j] = np.real(np.trace(H_j @ C_ref_Y @ H_i))

    return (K + c) ** d

# Bures-Wasserstein Gaussian Kernel
def bw_gaussian_kernel(X, Y=None, sigma=1):
    """
    Computes the Gaussian kernel between two sets of matrices using the Bures-Wasserstein distance.
    
    Parameters:
    X (array): An array of covariance matrices (N,p,p).
    Y (array, optional): An array of covariance matrices. Defaults to None.
    sigma (float): The scale parameter for the Gaussian kernel.
    
    Returns:
    K: A kernel matrix (p,p)
    """
    # Use X as both sets if Y is not provided
    if Y is None:
        Y = X
    
    n = X.shape[0]
    m = Y.shape[0]
    distances = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            distances[i, j] = BW_dist(X[i, :, :], Y[j, :, :])
    
    K = np.exp(-distances**2 / (2 * sigma**2))
    return K

# Bures-Wasserstein Gaussian Kernel (Generalized)
def bw_gen_gaussian_kernel(X, Y=None, sigma=1, q=2):
    """
    Computes a generalized Gaussian kernel between two sets of matrices using the Bures-Wasserstein distance.
    
    Parameters:
    X (array): An array of covariance matrices (N,p,p).
    Y (array, optional): An array of covariance matrices. Defaults to None.
    sigma (float): The scale parameter for the Gaussian kernel.
    q (float): The exponent parameter for the generalized Gaussian kernel, where 0 < q < 2.
    
    Returns:
    K: A kernel matrix (p,p)
    """
    # Use X as both sets if Y is not provided
    if Y is None:
        Y = X
    
    n = X.shape[0]
    m = Y.shape[0]
    distances = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            distances[i, j] = BW_dist(X[i, :, :], Y[j, :, :])
    
    K = np.exp(-distances**q / (2 * sigma**2))
    return K
