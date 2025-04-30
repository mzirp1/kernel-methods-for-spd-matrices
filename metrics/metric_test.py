##### Comparing the metrics from PyRiemann #####

import unittest

import time
import pyriemann as riemann

import BW_metric
import AI_metric
import LogE_metric

import sys
sys.path.append("..")

from utils import *

class CheckEqualMetric(unittest.TestCase):
    @unittest.skip("skip logeuclid")
    def test_logeuclid_gaussian_kernel(self):
        print()
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
    def test_bw_gaussian_kernel(self):
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

    @unittest.skip("ai dist")
    def test_riemann_gaussian_kernel(self):
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

        print("Implemented Affine-Invarient: ", end - start)
        self.assertTrue(are_matrices_equal(kern_matrix1, kern_matrix2))
  
    def test_bw_linear_kernel(self):
        ptsd_mat = load_matrice("../../fMRI/sfc_ptsd_dc_filt.mat")[0:3]


        # start = time.time()
        # kern_matrix1 = gaussian(
        #     riemann.utils.distance.pairwise_distance(ptsd_mat, metric="riemann")
        # )
        # end = time.time()
        # print("Riemann Affine-Invariant: ",  end - start)


        start = time.time()
        kern_matrix = BW_metric.bw_linear_kernel(ptsd_mat)
        end = time.time()
        print(kern_matrix)

        print("BW Linear Kernel: ", end - start)
        # self.assertTrue(are_matrices_equal(kern_matrix1, kern_matrix2))
if __name__ == '__main__':
    unittest.main()