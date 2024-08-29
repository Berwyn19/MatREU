"""
Author: Berwyn
File Name: xyz_to_cfg.py
Description: This file converts an xyz file into a cfg file needed for MLIP
"""

from ase.io import read


def read_xyz(file_name, idx):
    atoms = read(file_name, idx, format="extxyz")
    return atoms


def write_cfg(file_name, num_cfgs):
    # Write to the cfg file
    with open('inter-test.cfg', 'w') as file:
        for i in range(num_cfgs):
            # Get all the properties of a configuration
            cfg = read_xyz(file_name, i)
            forces = cfg.get_forces()
            energy = cfg.get_potential_energy()
            chemical_symbols = cfg.get_chemical_symbols()
            cells = list(cfg.get_cell())
            size = len(cfg)
            coordinates = cfg.get_positions()

            file.write("BEGIN_CFG\n")
            file.write(f" Size\n  {size}\n")
            file.write(f" Supercell\n  {cells[0][0]}    {cells[0][1]}    {cells[0][2]}\n")
            file.write(f"  {cells[1][0]}    {cells[1][1]}    {cells[1][2]}\n")
            file.write(f"  {cells[2][0]}    {cells[2][1]}    {cells[2][2]}\n")
            file.write(f" AtomData:   id    type cartes_x   cartes_y   cartes_z          fx           fy          fz\n")

            for i in range(size):
                if chemical_symbols[i] == "Mo":
                    atom_type = 0
                elif chemical_symbols[i] == "S":
                    atom_type = 1
                elif chemical_symbols[i] == "W":
                    atom_type = 2
                elif chemical_symbols[i] == "Se":
                    atom_type = 3
                file.write(f"              {i+1}     {atom_type}  {coordinates[i][0]}  {coordinates[i][1]}  "
                               f"{coordinates[i][2]}    {forces[i][0]}   {forces[i][1]}   {forces[i][2]}\n")

            file.write(f" Energy\n  {energy}\n")
            file.write("END_CFG\n\n")


# a = read_xyz("intra_mos2_lammps_recalculate.xyz", 0)
if __name__ == "__main__":
    write_cfg('inter-test.xyz', 3240)




