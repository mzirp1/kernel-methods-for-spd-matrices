import numpy as np
import time
import pyriemann as riemann
import scipy
from sklearn.metrics import accuracy_score, cohen_kappa_score

import csv
from utils import *
from sklearn.model_selection import ParameterGrid, RepeatedKFold
from sklearn.decomposition import KernelPCA
from sklearn.svm import SVC



def evaluate_kernel(kernel_mat, y, gamma, n_splits=5, n_repeats=3):
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)
    accuracies = []
    kappas = []
    times = []

    # Iterate over each fold and repetition
    for train_index, test_index in rkf.split(kernel_mat):
        y_train, y_test = y[train_index], y[test_index]
        trainK = kernel_mat[train_index, :]
        

        # Fit SVM classifier
        clf = SVC(kernel='rbf', C=1, gamma=gamma)
    
        start = time.time()
        model = clf.fit(trainK, y_train)
        
        # Predict on the test kernel
        y_pred = model.predict(kernel_mat[test_index, :])
        end = time.time()
        t = end - start
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)

        # Append metrics to lists
        accuracies.append(accuracy)
        kappas.append(kappa)
        times.append(t)


    # Calculate average accuracy
    avg_accuracy = np.mean(accuracies)
    avg_kappa = np.mean(kappas)
    avg_time = np.mean(times)



    return avg_accuracy, avg_kappa, avg_time


def create_fmri_results(pairwise_dist_path, parameters):
    """
    Perform Kernel Support Vector Machine (SVM) classification using a precomputed 
    kernel matrix with repeated k-fold cross-validation on the fMRI dataset.
    The function tunes the  regularization parameter C and evaluates model performance based on accuracy 
    and Cohen's kappa metric.

    Parameters:
    ----------
    kernel_path : string
        Path to where the the pairwise distance matrix for the dataset 
    parameters : list[dict]
        Parameter grid for kernels

    """

    with open('kernel_result.csv', 'w', newline='') as csvfile:
        fieldnames = [
            'Dataset', 
            'Classification Type', 
            'Metric', 
            'Kernel Type', 
            'Q',
            'Sigma', 
            'C',
            'D',
            'SVM Time', 
            'PCA + SVM Results Accuracy', 
            'PCA + SVM Kappa',
            'KPCA-Time', 
            'Gamma', 
            'Total PCA Components'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for params in parameters:
            gamma = params['gamma']
            classification = params['classification']
            distance = params['distance']
            dataset = params['dataset']
            ncomponents = params['n_components']
            kernel = params['kernel']
                            
            precomputed_pairwise_dist_file = os.path.join(pairwise_dist_path, dataset.name, distance + ".npy")

            # Skip over if the kernel has not been precomputed 
            if not os.path.exists(precomputed_pairwise_dist_file):
                continue

            y = dataset.ys if classification == MULTI else dataset.binary_ys
            kernel_mat = load_matrix_from_file(precomputed_pairwise_dist_file)

            c = None
            d = None
            q = None
            sigma = None
            if kernel == LINEAR:
                c = params['c']
                d = params['d']
                if c != 0 or d != 1:
                    kernel = POLYNOMIAL

                kernel_mat = (kernel_mat + c) ** d

            elif kernel == GAUSSIAN:
                sigma = params['sigma']
                q = params['q']
                if q != 2:
                    kernel = GAUSSIAN_GENERAL

                kernel_mat = gen_gaussian(kernel_mat, sigma, q)

            # Skip over if our kernel matrix is the identity matrix
            I = np.eye(kernel_mat.shape[0])
            if np.all(np.equal(kernel_mat, I)):
                continue

            # Skip over if our kernel matrix is not positive definite
            if not is_positive_definite(kernel_mat):
                continue

            kernel_pca = KernelPCA(
                n_components=ncomponents, kernel="precomputed",
            )

            pca_start = time.time()
            reduced_kernel_mat = kernel_pca.fit_transform(kernel_mat)
            pca_end = time.time()
            pca_time = np.round(pca_end - pca_start, 2)

            avg_accuracy, avg_kappa, avg_time = evaluate_kernel(reduced_kernel_mat, y, gamma)

            writer.writerow({
                'Dataset': dataset.name,
                'Classification Type': classification,
                'Metric': distance,
                'Kernel Type': kernel,
                'SVM Time': avg_time,
                'PCA + SVM Results Accuracy': avg_accuracy,
                'PCA + SVM Kappa': avg_kappa,
                'KPCA-Time': pca_time,
                'Sigma': sigma,
                'Q': q,
                'C': c,
                'D': d,
                'Gamma': gamma,
                'Total PCA Components': ncomponents,
            })



    



def main():
    PTSD_DATASET = Dataset("../fMRI/sfc_ptsd_dc_filt.mat", PTSD, True)
    ALZHEIMERS_DATASET = Dataset("../fMRI/sfc_adni_dc.mat", ALZHEIMERS, False)
    ADHD_DATASET = Dataset("../fMRI/sfc_adhd_dc.mat", ADHD, False)
    ABIDE_DATASET = Dataset("../fMRI/sfc_abide_dc.mat", ABIDE, False)


    # TODO: Figure out what to do for commented out datasets which are non-spd!
    datasets = [
        PTSD_DATASET, 
        # ALZHEIMERS_DATASET,
        # ADHD_DATASET, 
        # ABIDE_DATASET
    ]

    for data in datasets:
        print(f"{data.name} Matrix Shape: {data.stacked_mat.shape}")
        data.print_indexes()
        print()
        # print(data.ys)

    distance_choices = [LOGEUCLID, WASSERSTEIN, RIEMANN_AI]
    classification_choices = [MULTI, BINARY]

    # Utilized for gamma parameter in SVM
    gamma_choices = [0.005, 0.5, 1, 2, 4, 8, 10]

    # Utilized for the gaussian + generalized gaussian kernel
    # Gaussian of q = 2
    sigma_choices = [0.005, 0.5, 1, 2, 4, 8, 10]
    q_choices = [0.5, 2, 3]

    # Utilized for the polynomial and linear kernel
    # Linear kernel if c = 0, d = 1
    c_choices = [0.1, 0, 1, 5]
    d_choices = [0.5, 1, 2, 5]

    ncomponent_choices = np.arange(15, 120, 5)
    pairwise_dist_path = "../fMRI/kernels"


    grid = [
        {'kernel': [LINEAR], 'dataset': datasets, 'distance': distance_choices, 'classification': classification_choices, 'c': c_choices, 'd': d_choices, 'gamma': gamma_choices, 'n_components': ncomponent_choices}, 
        # {'kernel': [GAUSSIAN], 'dataset': datasets, 'distance': distance_choices, 'classification': classification_choices, 'sigma': sigma_choices, 'q': q_choices, 'gamma': gamma_choices, 'n_components': ncomponent_choices}
    ]
    
    parameters = ParameterGrid(grid)
    
    create_fmri_results(pairwise_dist_path, parameters)



if __name__ == '__main__':
    main()




