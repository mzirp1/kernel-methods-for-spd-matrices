import numpy as np
import scipy
import os

RIEMANN_AI = "riemann"
LOGEUCLID = "logeuclid"
WASSERSTEIN = "wasserstein"

PTSD = "PTSD"
ALZHEIMERS = "Alzheimers"
ADHD = "ADHD"
ABIDE = "Autism"

MULTI = "Multi"
BINARY = "Binary"

GAUSSIAN = "Gaussian"
GAUSSIAN_GENERAL = "GaussianGeneral"
LINEAR = "Linear"
POLYNOMIAL= "Polynomial"

# NOTE: This should probably be rewritten everything is a bit hacky...
class ClassIndex:
    """
    Given a class inside a dataset, the class index represents the starting and ending index in the array
    Parameters: 
    start (int): The starting index for the class
    end (int): The ending index for the class
    class_name (string): Name of the group
    """
    def __init__(self, start: int, end: int, class_name: str) -> None:
        self.start = start
        self.end = end
        self.class_name = class_name
    
    def __str__(self) -> str:
        return f'{self.class_name}: ({self.start}, {self.end})'


class Dataset:
    """
    Parameters: 
    dc_path (string): The relative path to the .mat file
    dc_name (string): The name of the database of the .mat file
    is_pd (bool): Whether 
    """
    def __init__(self, dc_path, dc_name, is_pd):

        # List of the ClassIndexes for all the classes given an fMRI dataset  
        self.indexes = []
        self.name = dc_name
        self.is_pd = is_pd

        dataset = scipy.io.loadmat(dc_path)


        # Get the keys (note that all the class keys are after index 3) and reorder so that the control is first
        # NOTE: This is hardcoded...
        keys = list(dataset.keys())[3:]
        keys = self.reorder_control(keys)

        dim = dataset[keys[0]].shape[0]

        self.stacked_mat = np.empty(shape=(dim, dim, 0))
        self.ys = np.empty(shape=(0), dtype=int)

        
        self.binary_ys = np.empty(shape=(0), dtype=int)
        
        cummulative = 0
        
        for i, k in enumerate(keys):
            
            mat = dataset[k]
            mat_n_elements = mat.shape[2]
            
            self.ys = np.concatenate((self.ys, [i] * mat_n_elements))
            
            # Since the controls are always first, the rest of the labels are 1s for the binary case
            if i == 0:
                self.binary_ys = np.concatenate((self.binary_ys, [0] * mat_n_elements))
            else: 
                self.binary_ys = np.concatenate((self.binary_ys, [1] * mat_n_elements))
            
            # The last word in the key in the class name
            keywords = k.split("_")
            class_name = keywords[-1]
            
            self.stacked_mat = np.concatenate((self.stacked_mat, mat), axis=2)
            
            class_index = ClassIndex(cummulative, cummulative + mat_n_elements, class_name)
            cummulative += mat_n_elements
            self.indexes.append(class_index)
        self.stacked_mat = self.stacked_mat.transpose((2, 1 ,0))

    # Function that ensures that the control is the first element in the list
    def reorder_control(self, keys):
        key_index = -1
        for i, k in enumerate(keys):
            if "control" in k:
                key_index = i
                break
        # Swap the control ksey so that it's the first element
        if key_index > 0:
            keys[0], keys[key_index] = keys[key_index], keys[0]
        return keys
    
    def print_indexes(self):
        for i in self.indexes:
            print(i)

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


def make_lower_symmetric(original_array):
    n, m, p = original_array.shape  # Get dimensions of the original array
    
    # Initialize result array with zeros
    result_array = np.zeros((n, m, p))
    
    # Apply the make_lower_symmetric operation to each 2D slice
    for i in range(p):
        slice_ = original_array[:, :, i]
        np.fill_diagonal(slice_, 1)  # Set diagonal elements to 1
        slice_[np.triu_indices(n, k=1)] = slice_.T[np.triu_indices(n, k=1)]  # Fill upper triangle with corresponding lower triangle values
        result_array[:, :, i] = slice_
    
    return result_array



def nearest_pd(A):
    """Find the nearest positive-definite matrix to input
    
    Code Retrieved From: https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite
    

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if is_positive_definite(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not is_positive_definite(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def gaussian(mat, sigma=1):
    return np.exp(-mat**2 / (2 * sigma**2))

def gen_gaussian(mat, sigma=1, q=2):
    return np.exp(-mat**q / (2 * sigma**2))

def try_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def save_matrix(matrix, path):
    with open(path, 'wb') as f:
        np.save(f, matrix)

def load_matrix_from_file(path):
    with open(path, 'rb') as f:
        matrix = np.load(f)
        return matrix