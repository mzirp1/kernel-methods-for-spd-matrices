##### Affine-Invariant Metric #####

import numpy as np
from scipy.linalg import sqrtm
from scipy.linalg import logm
from scipy.linalg import expm

# Affine Invariant Distance
def AI_dist(A, B):
    """
    Compute the affine-invariant distance between two positive-definite matrices A and B.

    Parameters:
    - A: array of shape (p, p), the first positive-definite matrix.
    - B: array of shape (p, p), the second positive-definite matrix.

    Returns:
    - output: float, the affine-invariant distance between matrices A and B.
    """
    # Compute the matrix product of A^(-1) * B
    temp = np.dot(np.linalg.inv(A), B)
    
    # Compute the eigenvalues of the product matrix
    lambdas, _ = np.linalg.eig(temp)
    
    # Affine-Invariant distance is based on the logarithms of the eigenvalues
    output = np.real(np.sqrt(np.sum(np.log(np.real(lambdas))**2) + 0j))
    
    return output


# Affine Invariant Logarithm Map
def AI_log(A, B):
    """
    Compute the affine-invariant logarithm map from matrix B to A.

    Parameters:
    - A: array of shape (p, p), the base matrix (positive-definite).
    - B: array of shape (p, p), the matrix to be mapped to the tangent space of A.

    Returns:
    - output: array of shape (p, p), the affine-invariant logarithm map from B to A.
    """
    # Compute the logarithm of the matrix after applying the affine transformation
    temp = logm(sqrtm(np.linalg.inv(A)).dot(B).dot(sqrtm(np.linalg.inv(A))))
    
    # Apply the square root of A to both sides of the logarithm
    output = sqrtm(A).dot(temp).dot(sqrtm(A))
    
    return output


# Affine Invariant Exponential Map
def AI_exp(A, B):
    """
    Compute the affine-invariant exponential map from matrix A to B.

    Parameters:
    - A: array of shape (p, p), the base matrix (positive-definite).
    - B: array of shape (p, p), the matrix to be mapped to the manifold.

    Returns:
    - output: array of shape (p, p), the affine-invariant exponential map from A to B.
    """
    # Compute the matrix exponential after applying the affine transformation
    temp = expm(sqrtm(np.linalg.inv(A)).dot(B).dot(sqrtm(np.linalg.inv(A))))
    
    # Apply the square root of A to both sides of the exponential
    output = sqrtm(A).dot(temp).dot(sqrtm(A))
    
    return output


# AI Projection Mean
def ai_projection_mean(x, eps, verbose=False):
    """
    Compute the affine-invariant projection mean of a set of positive-definite matrices.

    Parameters:
    - x: array of shape (N, p, p), a collection of N positive-definite matrices of dimension p x p.
    - eps: float, the convergence tolerance for the distance between successive means.
    - verbose: bool, optional, default is False. If True, prints the iteration process and convergence details.

    Returns:
    - mean_old: array of shape (p, p), the affine-invariant mean matrix, which is also positive-definite.
    """
    # Compute the initial guess for the mean matrix by averaging the input matrices
    mean_new = np.mean(x, axis=0)
    dist_mean = 10  # Initial distance larger than the threshold eps
    smaller_step = True
    last_dist = 0
    k = 0
    
    # Iterate until convergence (distance between means is smaller than eps or step size starts increasing)
    while dist_mean > eps and smaller_step:
        # Perform singular value decomposition on the current mean matrix
        U, s, VT = np.linalg.svd(mean_new, full_matrices=True)
        
        # Compute the symmetric part of the mean using SVD
        mean_old = (np.transpose(VT).dot(np.diag(s)).dot(VT) + np.transpose(np.transpose(VT).dot(np.diag(s)).dot(VT))) / 2
        
        # Compute the mean tangent vector in the affine-invariant geometry
        mean_tangent = np.mean(np.array([AI_log(mean_old, x[i]) for i in range(x.shape[0])]), axis=0)
        
        # Update the mean using the affine-invariant exponential map
        mean_new = np.real(AI_exp(mean_old, mean_tangent))
        
        # Compute the distance between the new and old means
        dist_mean = AI_dist(mean_new, mean_old)
        
        # Track whether the step size is decreasing
        smaller_step = True if k == 0 else (last_dist - dist_mean) > 0.0001

        # Verbose output for tracking the convergence process
        if verbose:
            print("Iter", k, 
                  ", Smaller step? ", "NA" if k == 0 else smaller_step, 
                  ", dist_mean=", np.round(dist_mean, 7), 
                  sep="")

        # Increment iteration count
        k += 1
        last_dist = dist_mean
        
        # Only update the mean when the step size is decreasing
        mean_old = mean_new if smaller_step else mean_old

    return mean_old


def ai_gaussian_kernel(X, Y=None, sigma=1):
    """
    Compute the AI Gaussian kernel between two sets of matrices based on AI distance.

    Parameters:
    - X: array of shape (n_samples_X, p, p), the first dataset (e.g., matrices).
    - Y: array of shape (n_samples_Y, p, p), the second dataset. If None, Y = X.
    - sigma: float, the bandwidth of the Gaussian kernel.

    Returns:
    - K: array of shape (n_samples_X, n_samples_Y), the AI Gaussian kernel matrix.
    """
    if Y is None:
        Y = X
    
    n_samples_X = X.shape[0]
    n_samples_Y = Y.shape[0]
    K = np.zeros((n_samples_X, n_samples_Y))
    
    for i in range(n_samples_X):
        for j in range(n_samples_Y):
            # Compute the AI distance between matrices
            K[i, j] = AI_dist(X[i, :, :], Y[j, :, :])
    
    # Apply the Gaussian kernel formula
    K = np.exp(-K**2 / (2 * sigma**2))
    
    return K


def ai_gen_gaussian_kernel(X, Y=None, sigma=1, q=2):
    """
    Compute the generalized AI Gaussian kernel between two sets of matrices based on AI distance.

    Parameters:
    - X: array of shape (n_samples_X, p, p), the first dataset.
    - Y: array of shape (n_samples_Y, p, p), the second dataset. If None, Y = X.
    - sigma: float, the bandwidth of the Gaussian kernel.
    - q: float, the power applied to the AI distance (0<q<2) (default: 2).

    Returns:
    - K: array of shape (n_samples_X, n_samples_Y), the generalized AI Gaussian kernel matrix.
    """
    if Y is None:
        Y = X
    
    n_samples_X = X.shape[0]
    n_samples_Y = Y.shape[0]
    K = np.zeros((n_samples_X, n_samples_Y))
    
    for i in range(n_samples_X):
        for j in range(n_samples_Y):
            # Compute the AI distance between matrices
            distances[i, j] = AI_dist(X[i, :, :], Y[j, :, :])
    
    # Apply the generalized Gaussian kernel formula
    K = np.exp(-distances**q / (2 * sigma**2))
    
    return K
