from ase.io import read, write
import numpy as np
import json

def remove_zeros(file_name):
    # Create the template vector of zeros
    atoms = read(file_name, index=":", format="extxyz")
    template_vector = [[-0.00000000, -0.00000000, -0.00000000] for i in range(12)]

    # Remove the conf with zeros
    updated_atoms = []
    indices = []
    for index, atom in enumerate(atoms):
        if not np.array_equal(atom.arrays["MACE_forces"], template_vector):
            updated_atoms.append(atom)
            indices.append(index)


    with open('indices.json', 'w') as file:
        json.dump(indices, file)

    write(file_name, updated_atoms, format="extxyz")

def remove_zeros_from_indices(file_name, indices):
    atoms = read(file_name, index=":", format="extxyz")
    updated_atoms = []
    for index, atom in enumerate(atoms):
        if index in indices:
            updated_atoms.append(atom)

    write(file_name, updated_atoms, format="extxyz")


if __name__ == "__main__":
    file_name = "inter-test.xyz"
    with open('indices.json', 'r') as file:
        indices = json.load(file)
    remove_zeros_from_indices(file_name, indices)
