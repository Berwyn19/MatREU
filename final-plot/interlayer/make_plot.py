import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from ase.io import read, write
import json
import pandas as pd


def make_plot():
    # Sample data
    plt.style.use('matplotlib.rc')
    atoms_allegro = read('allegro-predicted.xyz', index=":", format="extxyz")

    # Special treatment for mace because need to remove configuration with zero forces
    atoms_mace = read('mace-inter-final.xyz', index=":", format="extxyz")
    actual_mace = read('inter-test.xyz', index=":", format="extxyz")

    atoms = read('actual.xyz', index=":", format="extxyz")

    # All predicted values
    predicted_allegro = np.array([atom.get_forces() for atom in atoms_allegro]).flatten()
    predicted_mace = np.array([atom.arrays["MACE_forces"] for atom in atoms_mace]).flatten()
    
    with open('mtp-predicted.json', 'r') as file:
        predicted_mtp = np.array(json.load(file)).flatten()
    
    with open('predicted-uf23.json', 'r') as file:
        predicted_uf3 = np.array(json.load(file))

    # Actual values, differentiate between uf23 and the others because the orders are different
    actual_values = np.array([atom.get_forces() for atom in atoms]).flatten()
    with open('actual-uf23.json', 'r') as file:
        actual_uf3 = np.array(json.load(file))

    forces_mace = np.array([atom.get_forces() for atom in actual_mace]).flatten()

    # Calculate residuals
    residuals_allegro = np.log(np.abs(predicted_allegro - actual_values))
    residuals_mace = np.log(np.abs(predicted_mace - forces_mace))
    residuals_mtp = np.log(np.abs(predicted_mtp - actual_values))
    residuals_uf3 = np.log(np.abs(predicted_uf3 - actual_uf3))

    residuals_data = {
        'Log Residual of Force Components': np.concatenate([residuals_uf3, residuals_mtp, residuals_allegro, residuals_mace]),
        'ML Model': ['UF23'] * len(residuals_uf3) +
                   ['MTP'] * len(residuals_mtp) +
                   ['Allegro'] * len(residuals_allegro) + 
                   ['MACE'] * len(residuals_mace) 
                }
    df = pd.DataFrame(residuals_data)
    plt.rcParams.update({
        'font.size': 25,       # General font size
        'axes.titlesize': 24,  # Title font size
        'axes.labelsize': 24,  # Axis labels font size
        'xtick.labelsize': 24, # X-tick labels font size
        'ytick.labelsize': 24, # Y-tick labels font size
        'legend.fontsize': 16  # Legend font size
    })
    # Create a box plot of residuals
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='ML Model', y='Log Residual of Force Components', data=df, palette='Set2')
    plt.xlabel('ML Model')
    plt.ylabel('Log Residual of Force Components')
    plt.title('Box Plot of Residuals for Multiple Models on Interlayer Dataset', pad=20)
    plt.grid(True)
    plt.savefig('boxplot.png')


if __name__ == "__main__":
    make_plot()
