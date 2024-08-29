import numpy as np
from ase.io import read, write

def sort_data(data_file):
    atoms = read(data_file, index=":")
    new_cfg = sorted(atoms, key=lambda atom: atom.get_potential_energy())
    return new_cfg


def split_inter(data_file):
    data = sort_data(data_file)

sort_data('inter_mos2_lammps.xyz')