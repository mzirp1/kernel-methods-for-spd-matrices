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


def create_fmri_results(datasets: list[Dataset], pairwise_dist_path, distance_choices, classification_choices, sigma_choices, gamma_choices, ncomponent_choices):
    """
    Perform Kernel Support Vector Machine (SVM) classification using a precomputed 
    kernel matrix with repeated k-fold cross-validation on the fMRI dataset.
    The function tunes the  regularization parameter C and evaluates model performance based on accuracy 
    and Cohen's kappa metric.

    Parameters:
    ----------
    datasets : list[Dataset]
        An array of fMRI of datasets to evaluate

    kernel_path : string
        Path to where the the pairwise distance matrix for the dataset 

    distances_choices : array
        An array of distance choices to evaluate the fMRI dataset on (logeuclid, riemann, wasserstein)

    classification_choices : array
        An array of classifcations options for SVM (Binary or Multi)
    """

    with open('kernel_result.csv', 'w', newline='') as csvfile:

        # TODO: Include the parameters for each kernel, and get their SVM result, PCA results
        # Add results for SVM Results (Accuracy) for fMRI Dataset, PCA + SVM Results Accuracy, time on SVM?,  FDA (Fischer Discriminant Analysis)?, Time spent on PCA?, BOTH MULTI AND BINARY CLASSICATION, sigma numbers
        fieldnames = ['Dataset', 'Classification Type', 'Metric', 'Kernel Type', 'SVM Time', 'PCA + SVM Results Accuracy', 'PCA + SVM Kappa', 'KPCA-Time', 'Sigma', 'Gamma', 'Total PCA Components']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for params in ParameterGrid({'dataset': datasets, 'distance': distance_choices, 'classification': classification_choices, 'sigma': sigma_choices, 'gamma': gamma_choices, 'n_components': ncomponent_choices}):
            gamma = params['gamma']
            sigma = params['sigma']
            classification = params['classification']
            distance = params['distance']
            dataset = params['dataset']
            ncomponents = params['n_components']

            precomputed_pairwise_dist = os.path.join(pairwise_dist_path, dataset.name, distance + ".npy")
            kernel_mat = load_matrix_from_file(precomputed_pairwise_dist)
            kernel_mat = gaussian(kernel_mat, sigma)
            I = np.eye(kernel_mat.shape[0])

            # Skip over if our kernel matrix is the identity matrix
            if np.all(np.equal(kernel_mat, I)):
                continue



            y = dataset.ys if classification == MULTI else dataset.binary_ys

            # pca = KernelPCA(
            #     kernel="rbf", fit_inverse_transform=False, gamma=gamma
            # )

            kernel_pca = KernelPCA(
                n_components=ncomponents, kernel="precomputed", fit_inverse_transform=False, coef0=0.0001,
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
                'Kernel Type': 'Gaussian',
                'SVM Time': avg_time,
                'PCA + SVM Results Accuracy': avg_accuracy,
                'PCA + SVM Kappa': avg_kappa,
                'KPCA-Time': pca_time,
                'Sigma': sigma,
                'Gamma': gamma,
                'Total PCA Components': ncomponents,
            })



    



def main():
    PTSD_DATASET = Dataset("../fMRI/sfc_ptsd_dc_filt.mat", PTSD, True)
    ALZHEIMERS_DATASET = Dataset("../fMRI/sfc_adni_dc.mat", ALZHEIMERS, False)
    ADHD_DATASET = Dataset("../fMRI/sfc_adhd_dc.mat", ADHD, False)
    ABIDE_DATASET = Dataset("../fMRI/sfc_abide_dc.mat", ABIDE, False)

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

    # Utilized for the gaussian kernel
    sigmas_choices = [0.005, 0.5, 1, 2, 4, 8, 10]

    # Utilized for gamma parameter in SVM
    gamma_choices = [0.005, 0.5, 1, 2, 4, 8, 10]
    ncomponent_choices = np.arange(15, 120, 5)
    pairwise_dist_path = "../fMRI/kernels"

    create_fmri_results(datasets, pairwise_dist_path, distance_choices, classification_choices, sigmas_choices, gamma_choices, ncomponent_choices)


if __name__ == '__main__':
    main()




