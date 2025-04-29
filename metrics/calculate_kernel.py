import unittest

import numpy as np
import time
import os
import pyriemann as riemann
import scipy

import AI_metric
import BW_metric
import LogE_metric

import sys
sys.path.append("..")

from utils import *

def get_kernal_func(metric):

    kernel_func = None

    if metric == "wasserstein":
        kernel_func = BW_metric.bw_gaussian_kernel
    elif metric == "riemann":
        kernel_func = AI_metric.ai_gaussian_kernel
    elif metric == "logeuclid":
        kernel_func = LogE_metric.loge_gaussian_kernel
    
    return kernel_func
    

def calculate_avg_kernel_time(dataset: Dataset, metric, write_path):


    kernel_write_path = os.path.join(write_path, dataset.name)
    try_make_dir(kernel_write_path)
    
            
    kernel_file_name = os.path.join(kernel_write_path, metric + ".npy")
    
    mat = dataset.stacked_mat

    kernel_result = None
    start = time.time()
    if dataset.is_pd:
        kernel_result = riemann.utils.distance.pairwise_distance(mat, metric=metric)

    else:
        kernel_result = get_kernal_func(metric)
    end = time.time()
    print(f"Kernel Computation Time ({metric}): {end - start}")
    save_matrix(kernel_result, kernel_file_name)



  



def main():

    PTSD_DATASET = Dataset("../../fMRI/sfc_ptsd_dc_filt.mat", PTSD, True)
    ALZHEIMERS_DATASET = Dataset("../../fMRI/sfc_adni_dc.mat", ALZHEIMERS, False)
    ADHD_DATASET = Dataset("../../fMRI/sfc_adhd_dc.mat", ADHD, False)
    ABIDE_DATASET = Dataset("../../fMRI/sfc_abide_dc.mat", ABIDE, False)

    datasets = [
        PTSD_DATASET, 
        # ALZHEIMERS_DATASET, 
        # ADHD_DATASET, 
        # ABIDE_DATASET
    ]
    metrics = ["wasserstein", "riemann", "logeuclid"]


    for metric in metrics:
        for ds in datasets:
            calculate_avg_kernel_time(ds, metric, "../../fMRI/kernels")
            
    
if __name__ == '__main__':
    main()