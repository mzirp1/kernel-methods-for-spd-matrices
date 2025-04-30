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

def get_kernel_func(metric, kernel_type):
    kernel_dictionary = {
        WASSERSTEIN: {
            LINEAR:  BW_metric.bw_linear_kernel,
            GAUSSIAN: BW_metric.bw_gaussian_kernel,
        },

        RIEMANN_AI: {
            LINEAR:  None,
            GAUSSIAN: AI_metric.ai_gaussian_kernel,

        },

        LOGEUCLID: {
            LINEAR:  LogE_metric.loge_linear_kernel,
            GAUSSIAN: LogE_metric.loge_gaussian_kernel,
        }
    }

    return kernel_dictionary[metric][kernel_type]

def calculate_kernel_time(dataset: Dataset, metric, kernel_type, write_path):
    

    kernel_dataset_write_path = os.path.join(write_path, dataset.name)
    try_make_dir(kernel_dataset_write_path)

    kernel_dataset_metric_write_path = os.path.join(kernel_dataset_write_path, kernel_type)
    try_make_dir(kernel_dataset_metric_write_path)
    
            
    kernel_file_name = os.path.join(kernel_dataset_metric_write_path, metric + ".npy")
    
    print(kernel_file_name)
    mat = dataset.stacked_mat

    kernel_result = None
    start = time.time()
    if dataset.is_pd and kernel_type == GAUSSIAN:
        print(".")
        kernel_result = riemann.utils.distance.pairwise_distance(mat, metric=metric)
    else:
        kernel_func = get_kernel_func(metric, kernel_type)
        if kernel_func == None:
            return
        else:
            kernel_result = kernel_func(mat)

    end = time.time()
    print(f"Kernel Computation Time ({metric}) ({kernel_type}): {end - start}")
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
    metrics = [WASSERSTEIN, LOGEUCLID, RIEMANN_AI]

    kernel_type = [GAUSSIAN, LINEAR]

    for ds in datasets:
        for metric in metrics:
            for kernel in kernel_type:
                calculate_kernel_time(ds, metric, kernel, "../../fMRI/kernels")
            
    
if __name__ == '__main__':
    main()