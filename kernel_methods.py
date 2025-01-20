import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, cohen_kappa_score
import time

# Kernel SVM
def evaluate_kernel_svm(y, K, C_values=[1.0], n_splits=5, n_repeats=3):
    """
    Perform Kernel Support Vector Machine (SVM) classification using a precomputed 
    kernel matrix with repeated k-fold cross-validation. The function tunes the 
    regularization parameter C and evaluates model performance based on accuracy 
    and Cohen's kappa metric.

    Parameters:
    ----------
    y : array
        A 1D numpy array containing the labels for the dataset.
        Each element corresponds to the class label of a data point.

    K : array
        A precomputed kernel matrix (numpy array) where K[i, j] represents the 
        kernel value between data points i and j.

    C_values : list, optional (default=[1.0])
        A list of candidate values for the regularization parameter C used in 
        the SVM model.

    n_splits : int, optional (default=5)
        The number of splits for k-fold cross-validation.

    n_repeats : int, optional (default=3)
        The number of times the k-fold cross-validation is repeated.

    Returns:
    -------
        A dictionary containing:
        - 'best_accuracy': The highest average accuracy achieved during cross-validation.
        - 'best_kappa': The Cohen's kappa value corresponding to the best accuracy.
        - 'best_params': A dictionary containing the best value of C.
        - 'overall_time': The total time (in seconds) taken for cross-validation and parameter tuning.
    """
    best_accuracy = 0
    best_kappa = 0
    best_params = None
    
    # Start the overall timer
    start_overall = time.time()
    
    # Perform repeated cross-validation
    kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    for C_value in C_values:
        accuracies = []
        kappas = []
        
        # Perform repeated cross-validation
        for train_index, test_index in kf.split(y):
            y_train, y_test = y[train_index], y[test_index]
            trainK = K[train_index][:, train_index]  # Kernel for training set
            
            # Fit the SVM classifier using the precomputed kernel
            svm_model = SVC(kernel='precomputed', C=C_value)
            svm_model.fit(trainK, y_train)
            
            # Predict on the test kernel
            testK = K[test_index][:, train_index]  # Kernel for testing set
            y_pred = svm_model.predict(testK)
            
            # Calculate accuracy and kappa
            accuracy = accuracy_score(y_test, y_pred)
            kappa_val = cohen_kappa_score(y_test, y_pred)
            
            # Append metrics
            accuracies.append(accuracy)
            kappas.append(kappa_val)
        
        # Calculate average accuracy and kappa
        avg_accuracy = np.mean(accuracies)
        avg_kappa = np.mean(kappas)
        
        # Check if current parameters yield better accuracy
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_kappa = avg_kappa
            best_params = {'C': C_value}
    
    # End the overall timer
    end_overall = time.time()
    overall_time = end_overall - start_overall
    
    # Return the best accuracy, kappa, and parameters
    return {
        'best_accuracy': best_accuracy,
        'best_kappa': best_kappa,
        'best_params': best_params,
        'overall_time': overall_time
    }

# Kernel FDA
def evaluate_kernel_fda(y, K, lambdas, n_projections_list, n_splits=5, n_repeats=1):
    """
    Perform Kernel Fisher Discriminant Analysis (Kernel FDA) using a precomputed 
    kernel matrix with repeated k-fold cross-validation. The function tunes the 
    regularization parameter lambda and the number of projections, and evaluates 
    model performance based on accuracy and Cohen's kappa metric.

    Parameters:
    ----------
    y : array
        A 1D numpy array containing the labels for the dataset.
        Each element corresponds to the class label of a data point.

    K : array
        A precomputed kernel matrix (numpy array) where K[i, j] 
        represents the kernel value between data points i and j.

    lambdas : list, optional
        A list of candidate values for the regularization parameter lambda 
        used in the Kernel FDA model.

    n_projections_list : list, optional
        A list of candidate values for the number of projections to tune in 
        the Kernel FDA model.

    n_splits : int, optional (default=5)
        The number of splits for k-fold cross-validation.

    n_repeats : int, optional (default=3)
        The number of times the k-fold cross-validation is repeated.

    Returns:
    -------
        A dictionary containing:
        - 'best_accuracy': The highest average accuracy achieved during cross-validation.
        - 'best_kappa': The Cohen's kappa value corresponding to the best accuracy.
        - 'best_lambda': The value of lambda that resulted in the best performance.
        - 'best_projections': The number of projections that resulted in the best performance.
        - 'overall_time': The total time (in seconds) taken for cross-validation and parameter tuning.
"""

    best_accuracy = 0
    best_kappa = 0
    best_lambda = None
    best_n_projections = None

    # Start overall timer
    start_overall = time.time()

    # Perform repeated cross-validation
    kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    for lambda_value in lambdas:
        for n_projections in n_projections_list:
            accuracies = []
            kappas = []

            for train_index, test_index in kf.split(y):
                y_train, y_test = y[train_index], y[test_index]

                # Partition the precomputed kernel matrix
                K_train = K[train_index][:, train_index]
                K_test = K[test_index][:, train_index]

                # Center the kernel matrix
                N_train = len(train_index)
                one_n = np.ones((N_train, N_train)) / N_train
                K_centered = K_train - one_n @ K_train - K_train @ one_n + one_n @ K_train @ one_n

                # Between-class scatter matrix
                unique_classes = np.unique(y_train)
                S_B = np.zeros_like(K_centered)
                S_W = np.zeros_like(K_centered)

                for cls in unique_classes:
                    idx = np.where(y_train == cls)[0]
                    n_cls = len(idx)
                    K_cls = K_centered[:, idx]
                    mean_cls = K_cls.mean(axis=1, keepdims=True)
                    S_B += n_cls * (mean_cls @ mean_cls.T)
                    S_W += (K_cls @ K_cls.T) - n_cls * (mean_cls @ mean_cls.T)

                # Regularize S_W to ensure it is positive definite
                S_W += lambda_value * np.eye(S_W.shape[0])

                eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(S_W) @ S_B)

                # Sort eigenvectors by eigenvalues in descending order
                sorted_indices = np.argsort(eigvals)[::-1]
                eigvecs = eigvecs[:, sorted_indices]

                # Select top eigenvectors corresponding to the largest eigenvalues
                W = eigvecs[:, :n_projections]

                # Project data onto the discriminant
                projections = K_train @ W

                # Classify test samples based on projections
                projections_test = K_test @ W
                thresholds = [
                    projections[y_train == cls].mean(axis=0)
                    for cls in unique_classes
                ]
                y_pred = np.array([
                    unique_classes[np.argmin([np.linalg.norm(p - t) for t in thresholds])]
                    for p in projections_test
                ])

                # Compute metrics
                accuracy = accuracy_score(y_test, y_pred)
                kappa_val = cohen_kappa_score(y_test, y_pred)

                accuracies.append(accuracy)
                kappas.append(kappa_val)

            # Calculate average metrics for current parameters
            avg_accuracy = np.mean(accuracies)
            avg_kappa = np.mean(kappas)

            # Update best parameters if current combination is better
            if avg_accuracy > best_accuracy:
                best_accuracy = avg_accuracy
                best_kappa = avg_kappa
                best_lambda = lambda_value
                best_n_projections = n_projections

    # End overall timer
    end_overall = time.time()
    overall_time = end_overall - start_overall

    return {
        'best_accuracy': best_accuracy,
        'best_kappa': best_kappa,
        'best_lambda': best_lambda,
        'best_n_projections': best_n_projections,
        'overall_time': overall_time
    }
