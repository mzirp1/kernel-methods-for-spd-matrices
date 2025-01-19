import numpy as np
import os
import cv2

# Covariance Descriptors (5x5)
def compute_covariance_descriptor(image):
    """
    Compute the covariance descriptor for an image.

    Parameters:
        image (numpy array): Input image, either grayscale or RGB.

    Returns:
        numpy array: Covariance matrix representing spatial, intensity, and gradient relationships.
    """
    # Convert image to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Compute image gradients
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in x-direction
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in y-direction
    I = gray.astype(np.float64)  # Convert grayscale image to float64

    # Get image dimensions
    height, width = gray.shape

    # Initialize feature matrix
    features = np.zeros((height * width, 5))

    # Fill the feature matrix
    idx = 0
    for y in range(height):
        for x in range(width):
            features[idx, 0] = x  # Spatial x-coordinate
            features[idx, 1] = y  # Spatial y-coordinate
            features[idx, 2] = I[y, x]  # Intensity value
            features[idx, 3] = np.abs(Ix[y, x])  # Gradient magnitude in x-direction
            features[idx, 4] = np.abs(Iy[y, x])  # Gradient magnitude in y-direction
            idx += 1

    # Compute covariance matrix
    mean_vector = np.mean(features, axis=0)  # Calculate the mean of each feature
    centered_features = features - mean_vector  # Center the features
    covariance_matrix = np.cov(centered_features, rowvar=False)  # Compute the covariance matrix

    return covariance_matrix


def load_covariance_matrices(base_dir):
    """
    Load saved covariance matrices from the ETH-80 dataset.

    Parameters:
    ----------
    base_dir : str
        Base directory containing the dataset.

    Returns:
    -------
    cov_matrices : array, shape (n_samples, n_features, n_features)
        Array of covariance matrices.
    true_labels : array, shape (n_samples,)
        Corresponding labels for the matrices.
    """
    cov_matrices = []
    true_labels = []
    
    categories = ["apple", "car", "cow", "cup", "dog", "horse", "pear", "tomato"]
    
    for label, category in enumerate(categories):
        category_path = os.path.join(base_dir, category)
        print(f"Processing category: {category_path}")
        
        for subfolder in range(1, 11):  # Subfolders 1 to 10
            subfolder_path = os.path.join(category_path, str(subfolder))
            if not os.path.exists(subfolder_path):
                print(f"Subfolder does not exist: {subfolder_path}")
                continue
            
            # Look for .npy files in the subfolder
            for filename in os.listdir(subfolder_path):
                if filename.endswith(".npy"):
                    matrix_path = os.path.join(subfolder_path, filename)
                    print(f"Loading covariance matrix: {matrix_path}")
                    
                    # Load the covariance matrix
                    cov_matrix = np.load(matrix_path)
                    
                    # Ensure the matrix is 2D
                    if cov_matrix.ndim == 2:
                        cov_matrices.append(cov_matrix)
                        true_labels.append(label)
                    else:
                        print(f"Invalid matrix shape for file: {matrix_path}")

    if not cov_matrices:
        print("No covariance matrices were loaded. Please check the directory structure and files.")

    return np.array(cov_matrices), np.array(true_labels)


def filter_matrices(matrices, labels, classes):
    """
    Filter loaded covariance descriptors from ETH-80 dataset

    Parameters:
    ----------
    matrices : array, shape (n, p, p)
        Array of covariance descriptors
    labels : array, shape (n, )
        Corresponding labels for the descriptors
    classes : list
        List of desired classes from ETH-80
            # Specify the classes you want to filter
            #   0    1   2   3   4    5    6     7
            # apple car cow cup dog horse pear tomato
        
    Returns:
    -------
    filtered_matrices : array, shape (n, p, p)
        Array of filtered covariance descriptors
    filtered_labels : array, shape (n, )
        Corresponding labels for the filtered descriptors
    """
    filtered_matrices = []
    filtered_labels = []
    
    for i in range(len(labels)):
        if labels[i] in classes:
            filtered_matrices.append(matrices[i])
            filtered_labels.append(labels[i])
    
    return np.array(filtered_matrices), np.array(filtered_labels)

def subset_matrices(X, y, subset_size, num_classes=8, points_per_class=410):
    """
    Take smaller subset of matrices for quicker testing.

    Parameters:
    ----------
    X : array, shape (n, p, p)
        Array of covariance descriptors
    y : array, shape (n, )
        Corresponding labels for the descriptors
    subset_size : int
            number of images to consider per class
    num_classes : int
        number of classes used for evaluation
    points_per_class : int
        total number of images per class
    
    Returns:
    -------
    X : array, shape (subset_size*num_classes, p, p)
        Array of subsetted covariance descriptors
    y : array, shape (subset_size*num_classes, )
        Corresponding subsetted labels for the covariance descriptors
    """
    n_classes = 8
    points_per_class = 410
    indices = []
    for i in range(n_classes):
        class_indices = np.arange(i * points_per_class, (i + 1) * points_per_class)
        selected_indices = np.random.choice(class_indices, size=subset_size, replace=False)
        indices.extend(selected_indices)

    # Subset the data
    indices = np.array(indices)
    
    return X[indices], y[indices]
