import numpy as np
import time
import pyriemann as riemann
import scipy

import csv
from sklearn.model_selection import ParameterGrid


RIEMANN = "riemann"
LOGEUCLID = "logeuclid"
WASSERSTEIN = "wasserstein"

PTSD = "PTSD"
ALZHEIMERS = "Alzheimers"
ADHD = "ADHD"
ABIDE = "Autism"

MULTI = "Multi"
BINARY = "Binary"




# NOTE: This should probably be rewritten everything is a bit hacky...

class ClassIndex:

    """
    Given a class inside a dataset, the class index represents the starting and ending index in the array
    Parameters: 
    dc_path: 
    dc_name: The name of the database of the .mat file
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
    dc_path: The relative path to the .mat file
    dc_name: The name of the database of the .mat file
    """
    def __init__(self, dc_path, dc_name):

        # List of the ClassIndexes for all the classes given an fMRI dataset  
        self.indexes = []
        self.name = dc_name

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
            keywords = k.split("_")
            
            mat = dataset[k]
            mat_n_elements = mat.shape[2]
            
            self.ys = np.concatenate((self.ys, [i] * mat_n_elements))
            
            self.binary_ys = np.concatenate((self.binary_ys, [min(i, 1)] * mat_n_elements))
            
            # The last word in the key in the class name
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
        if key_index > 0:
            keys[0], keys[key_index] = keys[key_index], keys[0]
        return keys
    
    def print_indexes(self):
        for i in self.indexes:
            print(i)
        


if __name__ == '__main__':
    PTSD_DATASET = Dataset("../fMRI/sfc_ptsd_dc_filt.mat", PTSD)

    ALZHEIMERS_DATASET = Dataset("../fMRI/sfc_adni_dc.mat", ALZHEIMERS)
    ADHD_DATASET = Dataset("../fMRI/sfc_adhd_dc.mat", ADHD)

    ABIDE_DATASET = Dataset("../fMRI/sfc_abide_dc.mat", ABIDE)

    datasets = [PTSD_DATASET]

    for data in datasets:
        print(f"{data.name} Matrix Shape: {data.stacked_mat.shape}")
        data.print_indexes()
        print()

    distance_choices = [LOGEUCLID]
    classification_choices = [MULTI]

    # gamma_choices = [0.003]
    # sigma_choices = [0.03]



    with open('kernel_result.csv', 'w', newline='') as csvfile:
        fieldnames = ['Dataset', 'Kernel Type', 'Comp Time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()


        for dataset in datasets:
            mat = dataset.stacked_mat
            start = time.time()
            # TODO: Take the gaussian
            kern_matrix = riemann.utils.distance.pairwise_distance(mat, mat, metric="wasserstein")
            end = time.time()
            print(end-start)
            writer.writerow({'Dataset': dataset.name, 'Kernel Type': 'BW', 'Comp Time': np.round(end-start, 2)})




