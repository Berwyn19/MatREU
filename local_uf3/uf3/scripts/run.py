import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from uf3.data import io
from uf3.data import geometry
from uf3.data import composition
from uf3.representation import bspline
from uf3.representation import process
from uf3.regression import least_squares
from uf3.regression.optimize import (get_bspline_config, 
                               get_lower_cutoffs, 
                               get_columns_to_drop_2b, 
                               get_columns_to_drop_3b)

# Defining the system
element_list = ['Mo', 'S', 'S']
degree = 3
chemical_system = composition.ChemicalSystem(element_list=element_list, degree=degree)
print(chemical_system)


# Defining hyperparameters (cutoff radius)
rmin_2b = 0 #Min of radial cutoff for 2-body
rmin_3b = 1.6 #Min of radial cutoff for 2-body

knot_spacing_2b = 0.5 #spacing between consecutive knots for 2-body
knot_spacing_3b = 0.8 #spacing between consecutive knots for 3-body

rmax_2b = 8 #Cutoff 2-body

rmax_3b = 5.6 #Cutoff 3-body. If A-B-C are the interacting bodies with A as the central atom. Then rmax_3b is
                #maximum possible distance between A-B (or A-C).
                #rmax_3b = Max dist(A,B) = Max dist(A,C) = Max dist(B,C)/2.

bspline_config = get_bspline_config(chemical_system=chemical_system,
                                    rmin_2b=rmin_2b,
                                    rmin_3b=rmin_3b,
                                    rmax_2b=rmax_2b,
                                    rmax_3b=rmax_3b,
                                    knot_spacing_2b=knot_spacing_2b,
                                    knot_spacing_3b=knot_spacing_3b,
                                    leading_trim=0,trailing_trim=3)

# Load the Data
example_directory = os.getcwd()
data_filename = os.path.join(example_directory, "intra-training.xyz")
data_coordinator = io.DataCoordinator()

data_coordinator.dataframe_from_trajectory(data_filename,
                                           prefix='dft')
df_data = data_coordinator.consolidate()

# Compute forces and energies
with open(os.path.join(example_directory, "tune_idx.txt"), "r") as f:
    training_100 = [int(idx) for idx in f.read().splitlines()]

filename = "df_features.h5"
table_template = "features_{}"

n_cores = 30
representation = process.BasisFeaturizer(bspline_config)
client = ProcessPoolExecutor(max_workers=n_cores)

representation.batched_to_hdf(filename,
                              df_data.iloc[training_100],
                              client,
                              n_jobs = n_cores,
                              batch_size=5,
                              progress='bar',
                              table_template=table_template)

lower_cutoffs = get_lower_cutoffs(original_bspline_config=bspline_config)
lower_rmax_2b, lower_rmax_3b = lower_cutoffs["lower_rmax_2b"], lower_cutoffs["lower_rmax_3b"]

lower_rmax_3b, lower_rmax_2b = np.meshgrid(lower_rmax_3b, lower_rmax_2b)
lower_rmax_3b, lower_rmax_2b = lower_rmax_3b.flatten(), lower_rmax_2b.flatten()

outer_hp = []
                                        
for i in range(lower_rmax_2b.shape[0]):
    if lower_rmax_2b[i]>=2*lower_rmax_3b[i]:
        outer_hp.append([lower_rmax_2b[i],lower_rmax_3b[i]])

# Split data into 3 equal parts for three-fold cross validation
df_folds = [i for i in np.array_split(df_data.iloc[training_100],3)]

# Perform three fold cross validation
results = []
cv_fold = 3

for i in outer_hp:
    rmax_2b = i[0]
    rmax_3b = i[1]
    data = [rmax_2b,rmax_3b]
    bspline_config_lower_cutoff = get_bspline_config(chemical_system=chemical_system,
                                                     rmin_2b=rmin_2b,
                                                     rmin_3b=rmin_3b,
                                                     rmax_2b=rmax_2b,
                                                     rmax_3b=rmax_3b,
                                                     knot_spacing_2b=knot_spacing_2b,
                                                     knot_spacing_3b=knot_spacing_3b,
                                                     leading_trim=0,trailing_trim=3)
    
    columns_to_drop_2b = get_columns_to_drop_2b(original_bspline_config=bspline_config,
                                                modify_2b_cutoff=rmax_2b,
                                                knot_spacing_2b=knot_spacing_2b)

    columns_to_drop_3b = get_columns_to_drop_3b(original_bspline_config=bspline_config,
                                                modify_3b_cutoff=rmax_3b,
                                                knot_spacing_3b=knot_spacing_3b)
    
    columns_to_drop = columns_to_drop_2b
    columns_to_drop.extend(columns_to_drop_3b)
    
    for hold_out_fold in range(0,cv_fold):
        df_train_data = [df_folds[j].copy() for j in range(len(df_folds)) if j!=hold_out_fold]
        df_valid_data = df_folds[hold_out_fold]
        
        df_train_data = pd.concat(df_train_data)
        
        training_keys = list(df_train_data.index)
        validation_keys = list(df_valid_data.index)
        
        regularizer = bspline_config_lower_cutoff.get_regularization_matrix(ridge_1b=0,
                                                                    ridge_2b=0.0,
                                                                    ridge_3b=0)

        model = least_squares.WeightedLinearModel(bspline_config_lower_cutoff, regularizer=regularizer)
        
        model.fit_from_file(filename="df_features.h5",
                            subset=training_keys,
                            weight=1,
                            batch_size=100,
                            energy_key="energy", 
                            progress=None,
                            drop_columns=columns_to_drop)
        
        yt_e, pt_e, yt_f, pt_f, rmset_e, rmset_f = model.batched_predict(filename="df_features.h5",
                                                                         keys=training_keys,
                                                                         drop_columns=columns_to_drop)
        
        yv_e, pv_e, yv_f, pv_f, rmsev_e, rmsev_f = model.batched_predict(filename="df_features.h5",
                                                                         keys=validation_keys,
                                                                         drop_columns=columns_to_drop)
        if hold_out_fold ==0:                                                   
            errors = np.array([rmset_e,rmset_f,rmsev_e,rmsev_f])     
        else:                                                                   
            errors = errors + np.array([rmset_e,rmset_f,rmsev_e,rmsev_f])
            
    errors = errors/cv_fold
    data.extend(list(errors))
    results.append(data)

df_result = pd.DataFrame(results,columns=["rmax_2b","rmax_3b","training_error_energy","traininig_error_force",
                                         "validation_error_energy","validation_error_force"])
print(df_result)



