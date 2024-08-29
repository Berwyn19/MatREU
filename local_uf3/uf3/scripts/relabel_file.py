from ase.io import read, write
import numpy as np
import json

def relabel_file(file_name):
    ats = read(file_name, ":", format="extxyz")
    for at in ats:
        at.set_atomic_numbers([42, 16, 16, 42, 16, 16, 74, 34, 34, 74, 34, 34])
    write(file_name, ats, format="extxyz")

if __name__ == "__main__":
    relabel_file("<file name>")