# ETH-80 Covariance Descriptors

This folder contains utility functions for generating and loading covariance descriptors from the ETH-80 dataset. Covariance descriptors are \(5 $\times$ 5\) matrices that represent spatial, intensity, and gradient relationships in images.

---

## Functions Overview

### `compute_covariance_descriptor(image)`

Generates a \(5 $\times$ 5\) covariance matrix for an input image.

**What It Does**:
1. Converts the input image to grayscale (if not already).
2. Computes image gradients in the x and y directions using the Sobel operator.
3. Constructs a feature matrix where each pixel contributes:
   - \(x\)-coordinate (spatial position).
   - \(y\)-coordinate (spatial position).
   - Intensity value.
   - Gradient magnitude in x-direction.
   - Gradient magnitude in y-direction.
4. Computes the covariance matrix of the feature matrix to summarize the spatial, intensity, and gradient relationships.

**Parameters**:
- `image` (numpy array): Input image, either grayscale or RGB.

**Returns**:
- `covariance_matrix` (numpy array): \(5 $\times$ 5\) covariance descriptor of the input image.

---

### `load_covariance_matrices(base_dir)`

Loads precomputed covariance matrices from the ETH-80 dataset.

**What It Does**:
1. Iterates through object categories in the dataset.
2. Reads `.npy` files containing covariance matrices from subfolders within each category.
3. Assigns labels to each covariance matrix based on its category.

**Parameters**:
- `base_dir` (str): Base directory containing the ETH-80 dataset.

**Returns**:
- `cov_matrices` (numpy array): Array of covariance matrices (\(n $\times$ 5 $\times$ 5\)).
- `true_labels` (numpy array): Integer labels corresponding to categories.

**Expected Folder Structure**:
The dataset is organized as follows:

```plaintext
base_dir/
├── apple/
│   ├── 1/
│   │   ├── cov_matrix1.npy
│   │   ├── cov_matrix2.npy
│   │   └── ...
│   ├── 2/
│   │   ├── cov_matrix1.npy
│   │   ├── cov_matrix2.npy
│   │   └── ...
│   └── ...
├── car/
│   ├── 1/
│   │   ├── cov_matrix1.npy
│   │   ├── cov_matrix2.npy
│   │   └── ...
│   └── ...
├── cow/
│   └── ...
├── cup/
│   └── ...
├── dog/
│   └── ...
├── horse/
│   └── ...
├── pear/
│   └── ...
└── tomato/
    └── ...
```
