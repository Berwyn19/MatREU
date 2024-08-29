import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
from uf3.data import io
from uf3.data import geometry
from uf3.data import composition
from uf3.representation import bspline
from uf3.representation import process
from uf3.regression import least_squares
from uf3.forcefield import calculator
from uf3.forcefield import lammps
from uf3.util import parallel
from uf3.util import plotting
from uf3.util import plotting3d
import json

element_list = ['Mo', 'S', 'S']
degree = 3
chemical_system = composition.ChemicalSystem(element_list=element_list,
                                             degree=degree)

r_max_map = {("S", "S"): 7.0,
             ("S", "Mo"): 7.0,
             ("Mo", "Mo"): 7.0,
             ('S', 'S', 'S'): [3.2, 3.2, 6.4],
             ('S', 'S', 'Mo'): [3.2, 3.2, 6.4],
             ('S', 'Mo', 'Mo'): [3.2, 3.2, 6.4],
             ('Mo', 'S', 'S'): [3.2, 3.2, 6.4],
             ('Mo', 'S', 'Mo'): [3.2, 3.2, 6.4],
             ('Mo', 'Mo', 'Mo'): [3.2, 3.2, 6.4]
            }

resolution_map = {("S", "S"): 15,
                  ("S", "Mo"): 15,
                  ("Mo", "Mo"): 15,
                  ('S', 'S', 'S'): [5, 10, 10],
                  ('S', 'S', 'Mo'): [5, 10, 10],
                  ('S', 'Mo', 'Mo'): [5, 10, 10],
                  ('Mo', 'S', 'S'): [5, 10, 10],
                  ('Mo', 'S', 'Mo'): [5, 10, 10],
                  ('Mo', 'Mo', 'Mo'): [5, 10, 10]
                }

trailing_trim = 3
leading_trim = 0

n_cores = 4
example_directory = os.getcwd()
data_filename = os.path.join(example_directory, "intra-training.xyz")
with open(os.path.join(example_directory, "training_idx.txt"), "r") as f:
    training_799 = [int(idx) for idx in f.read().splitlines()]

bspline_config = bspline.BSplineBasis(chemical_system,
                                      r_max_map=r_max_map,
                                      resolution_map=resolution_map,
                                      leading_trim=leading_trim,
                                      trailing_trim=trailing_trim)

data_coordinator = io.DataCoordinator()
data_coordinator.dataframe_from_trajectory(data_filename,
                                           prefix='dft')
df_data = data_coordinator.consolidate()

representation = process.BasisFeaturizer(bspline_config)
client = ProcessPoolExecutor(max_workers=n_cores)
filename = "df_feature.h5"
table_template = "features_{}"
representation.batched_to_hdf(filename,
                              df_data,
                              client,
                              n_jobs = n_cores,
                              batch_size=50,
                              progress="bar",
                              table_template=table_template)

# Fit the model
regularizer = bspline_config.get_regularization_matrix(ridge_1b=0.0,
                                                       ridge_2b=0.0,
                                                       ridge_3b=1e-8,
                                                       curvature_2b=1e-8,
                                                       curvature_3b=0.0)

model = least_squares.WeightedLinearModel(bspline_config,
                                          regularizer=regularizer)


model.fit_from_file(filename, 
                    df_data.index[training_799],
                    weight=0.8, 
                    batch_size=1000,
                    energy_key="energy", 
                    progress="bar")

# Load the test data
test_directory = os.getcwd()
test_filename = os.path.join(test_directory, "intra-test.xyz")
with open(os.path.join(test_directory, "test_idx.txt"), "r") as f:
    training_101 = [int(idx) for idx in f.read().splitlines()]

data_coordinator = io.DataCoordinator()
data_coordinator.dataframe_from_trajectory(test_filename,
                                           prefix='dft')

test_data = data_coordinator.consolidate()
holdout_keys = test_data.index[training_101]

y_e, p_e, y_f, p_f, rmse_e, rmse_f = model.batched_predict(filename, 
                                                           keys=holdout_keys)

with open('forces-intra-predicted.json', 'w') as file:
    json.dump(list(p_f), file)

with open('force-intra-actual.json', 'w') as file:
    json.dump(list(y_f), file)

print(rmse_e)
print(rmse_f)