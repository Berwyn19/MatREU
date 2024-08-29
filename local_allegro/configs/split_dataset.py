import numpy as np
from ase.io import read, write


def split_dataset(data_file):
    

    # Specifying the proportion of each type of dataset
    data_size = len(atoms)
    training_size = int(0.8 * data_size)
    validation_size = int(0.1 * data_size)
    test_size = int(0.1 * data_size)
    
    random_indices = np.random.permutation(data_size)


    training_dataset = [atoms[j] for j in random_indices[0:training_size]]
    validation_dataset = [atoms[j] for j in random_indices[training_size:(training_size + validation_size)]]
    test_dataset = [atoms[j] for j in random_indices[training_size + validation_size:]]

    write('inter_training.xyz', training_dataset)
    write('inter_validation.xyz', validation_dataset)
    write('inter_dataset.xyz', test_dataset)

split_dataset('inter_mos2_lammps.xyz')