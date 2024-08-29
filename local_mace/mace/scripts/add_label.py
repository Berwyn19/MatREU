from ase.io import read, write


def relabel_xyz(file_name):
    atoms = read(file_name, index=":", format="extxyz")
    for atom in atoms:
        atom.info["interlayer_ml"] = True
    write(file_name, atoms, format="extxyz")


if __name__ == "__main__":
    data_filename = "inter-test.xyz"
    ats = read(data_filename, ":", format="extxyz")
    for at in ats:
        at.set_atomic_numbers([42, 16, 16, 42, 16, 16, 42, 16, 16, 42, 16, 16])
    write("inter-test.xyz", ats, format="extxyz")