import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
from uf3.data import io
from uf3.data import geometry
from uf3.data import composition
from uf3.representation import bspline
from uf3.representation import distances
from uf3.representation import process
from uf3.regression import least_squares
from uf3.forcefield import calculator
from uf3.forcefield import lammps
from uf3.util import parallel
from uf3.util import plotting
import pandas as pd
import json

# Defining the chemical symbols and degree of the cubic spline
element_list = ['Mo', 'S', 'W', 'Se']
degree = 2

# # Build ChemicalSystem of this form:
# """
# ChemicalSystem:
#     Elements: ('S', 'Mo')
#     Degree: 2
#     Pairs: [('S', 'S'), ('S', 'Mo'), ('Mo', 'Mo')]
# """
chemical_system = composition.ChemicalSystem(element_list=element_list,
                                             degree=degree)
print(chemical_system)
# Initialize several parameters
r_min_map = {('S', 'S'): 0.0,
             ('S', 'Se'): 1.5,
             ('S', 'Mo'): 0.0,
             ('S', 'W'): 1.5,
             ('Se', 'Se'): 0.0,
             ('Se', 'Mo'): 1.5,
             ('Se', 'W'): 0.0,
             ('Mo', 'Mo'): 0.0,
             ('Mo', 'W'): 1.5,
             ('W', 'W'): 0.0
            }

r_max_map = {('S', 'S'): 0.0,
             ('S', 'Se'): 8.0,
             ('S', 'Mo'): 0.0,
             ('S', 'W'): 8.0,
             ('Se', 'Se'): 0.0,
             ('Se', 'Mo'): 8.0,
             ('Se', 'W'): 0.0,
             ('Mo', 'Mo'): 0.0,
             ('Mo', 'W'): 8.0,
             ('W', 'W'): 0.0
            }


resolution_map = {('S', 'S'): 0,
             ('S', 'Se'): 40,
             ('S', 'Mo'): 0,
             ('S', 'W'): 40,
             ('Se', 'Se'): 0,
             ('Se', 'Mo'): 40,
             ('Se', 'W'): 0,
             ('Mo', 'Mo'): 0,
             ('Mo', 'W'): 40,
             ('W', 'W'): 0
            }

trailing_trim = 3
leading_trim = 0

# Number of workers
n_cores = 4

# Initialize the cubic spline basis
bspline_config = bspline.BSplineBasis(chemical_system,
                                      r_min_map=r_min_map,
                                      r_max_map=r_max_map,
                                      resolution_map=resolution_map,
                                      leading_trim=leading_trim,
                                      trailing_trim=trailing_trim)

# Load the dataset
example_directory = os.getcwd()
data_filename = os.path.join(example_directory, "inter-training-refined.xyz")
with open(os.path.join(example_directory, "training-idx-refined.txt"), "r") as f:
    training_799 = [int(idx) for idx in f.read().splitlines()]

data_coordinator = io.DataCoordinator()
data_coordinator.dataframe_from_trajectory(data_filename,
                                           prefix='dft')

df_data = data_coordinator.consolidate()

# Compute features
representation = process.BasisFeaturizer(bspline_config)
client = ProcessPoolExecutor(max_workers=n_cores)
n_batches = n_cores * 16  # added granularity for more progress bar updates
df_features = representation.evaluate_parallel(df_data,
                                               client,
                                               energy_key=data_coordinator.energy_key,
                                               n_jobs=n_batches)

# Fitting the model
training_keys = df_data.index[training_799]
df_slice = df_features.loc[training_keys]
n_elements = len(chemical_system.element_list)

x_e, y_e, x_f, y_f = least_squares.dataframe_to_tuples(df_slice,
                                                       n_elements=n_elements,
                                                       energy_key="energy")

regularizer = bspline_config.get_regularization_matrix(ridge_1b=1e-16,
                                                       ridge_2b=1e-20,
                                                       curvature_2b=1e-6)

model = least_squares.WeightedLinearModel(bspline_config,
                                          regularizer=regularizer)

model.fit(x_e, y_e, x_f, y_f, weight=0.8)

# Load the test data
test_directory = os.getcwd()
test_filename = os.path.join(test_directory, "inter-test-refined.xyz")
with open(os.path.join(test_directory, "test-idx-refined.txt"), "r") as f:
    training_101 = [int(idx) for idx in f.read().splitlines()]
    print(training_101)

data_coordinator = io.DataCoordinator()
data_coordinator.dataframe_from_trajectory(test_filename,
                                           prefix='dft')

test_data = data_coordinator.consolidate()
df_features = representation.evaluate_parallel(test_data,
                                               client,
                                               energy_key=data_coordinator.energy_key,
                                               n_jobs=n_batches)
holdout_keys = test_data.index[training_101]
df_holdout = df_features.loc[holdout_keys]


# Initialize the model
x_e, y_e, x_f, y_f = least_squares.dataframe_to_tuples(df_holdout,
                                                       n_elements=n_elements,
                                                       energy_key="energy")

# Predict the test data with the model
p_e = model.predict(x_e)
p_f = model.predict(x_f)

# print(p_e)
print(len(p_f))
print(len(y_f))

with open('forces.json', 'w') as file:
    json.dump(list(p_f), file)

with open('actual.json', 'w') as file:
    json.dump(list(y_f), file)
# Calculate energy and force rmse
rmse_e = np.sqrt(np.mean(np.subtract(y_e, p_e)**2))

rmse_f = np.sqrt(np.mean(np.subtract(y_f, p_f)**2))
print(f"Energy root-mean-square error: {rmse_e} eV/atom")
print(f"Force component root-mean-square error: {rmse_f} eV/angstrom")
 

