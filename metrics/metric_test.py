##### Comparing the metrics from PyRiemann #####

import unittest

import numpy as np
import time
import pyriemann as riemann
import scipy


import BW_metric
import AI_metric
import LogE_metric


def is_positive_definite(A):
    """
    Check if a given matrix is positive definite.

    Parameters:
    A: A matrix

    Returns:
    is_pd: (bool): Whether the matrix is postive definite

    """
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.linalg.LinAlgError as err:
        return False

def is_symmetric(A, epsilon=1e-8):
    """
    Check if a given matrix is symmetric.

    Parameters: 
    A: A matrix
    epsilon: The max difference range

    Returns:
    is_symettric: (bool): Whether the matrix is symetric

    """
    return np.all(np.abs(A-A.T) < epsilon)

def gaussian(mat, sigma=1):
    return np.exp(-mat**2 / (2 * sigma**2))

def are_matrices_equal(mat1, mat2, epsilon=1e-9):
    """
    Checks whether two matrices are equal within a given epsilon value.
    
    Parameters:
    mat1 (np.ndarray): First matrix
    mat2 (np.ndarray): Second matrix.
    epsilon (float): Tolerance for equality.
    
    Returns:
    bool: Whether the matrices are equal within epsilon
    """
    
    if mat1.shape != mat2.shape:
        return False
    
    return np.all(np.abs(mat1 - mat2) <= epsilon)

def load_matrice(filepath):
    dataset = scipy.io.loadmat(filepath)
    keys = list(dataset.keys())[3:]

    dim = dataset[keys[0]].shape[0]
    
    stacked_mat = np.empty(shape=(dim, dim, 0))

    for _, k in enumerate(keys):
      mat = dataset[k]
      stacked_mat = np.concatenate((stacked_mat, mat), axis=2)


    stacked_mat = stacked_mat.transpose((2, 1 ,0))
    return stacked_mat
    
class CheckEqualMetric(unittest.TestCase):
    @unittest.skip("skip logeuclid")
    def test_logeuclid_metric(self):
    
        # TODO - Change filepath, check in implementation 
        ptsd_mat = load_matrice("../../fMRI/sfc_ptsd_dc_filt.mat")[0:5]


        start = time.time()
        kern_matrix1 = gaussian(
            riemann.utils.distance.pairwise_distance(ptsd_mat, metric="logeuclid")
        )
        end = time.time()
        
        print("Riemann Logeuclid: ",  end - start)

        start = time.time()
        kern_matrix2 = LogE_metric.loge_gaussian_kernel(ptsd_mat)
        end = time.time()
        print("Implemented Logeuclid: ", end - start)
        self.assertTrue(are_matrices_equal(kern_matrix1, kern_matrix2))

    # @unittest.skip("skip bw")
    def test_bw_metric(self):
        ptsd_mat = load_matrice("../../fMRI/sfc_ptsd_dc_filt.mat")[0:5]


        start = time.time()
       
        kern_matrix1 = gaussian(
            riemann.utils.distance.pairwise_distance(ptsd_mat, metric="wasserstein")
        )
        end = time.time()
        print("Riemann BW: ",  end - start)


        start = time.time()
        kern_matrix2 = BW_metric.bw_gaussian_kernel(ptsd_mat)
        end = time.time()
        print("Implemented BW: ", end - start)
        self.assertTrue(are_matrices_equal(kern_matrix1, kern_matrix2), True)

    # @unittest.skip("ai dist")
    def test_riemann_mean(self):
        ptsd_mat = load_matrice("../../fMRI/sfc_ptsd_dc_filt.mat")[0:5]


        start = time.time()
        kern_matrix1 = gaussian(
            riemann.utils.distance.pairwise_distance(ptsd_mat, metric="riemann")
        )
        end = time.time()
        print("Riemann Affine-Invariant: ",  end - start)


        start = time.time()
        kern_matrix2 = AI_metric.ai_gaussian_kernel(ptsd_mat)
        end = time.time()
        print("Implemented Riemann Mean: ", end - start)
        print(are_matrices_equal(kern_matrix1, kern_matrix2))
        self.assertTrue(are_matrices_equal(kern_matrix1, kern_matrix2))
  
if __name__ == '__main__':
    unittest.main()