import matplotlib.pyplot as plt
import numpy as np
from ase.io import read, write

# Example data: actual values and predicted values
plt.style.use('matplotlib.rc')
atoms_actual = read("actual-data.xyz", index=":", format="extxyz")
allegro_predicted = read("allegro-predicted.xyz", index=":", format="extxyz")

actual_allegro = np.array([atom.get_forces() for atom in atoms_actual]).flatten()
predicted_allegro = np.array([atom.get_forces() for atom in allegro_predicted]).flatten()

# Create the parity plot
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(actual_allegro, predicted_allegro, color='dodgerblue', alpha=1.0, label='Predicted vs Actual', edgecolor='black')

# Add the y = x line
min_val = min(min(actual_allegro), min(predicted_allegro))
max_val = max(max(actual_allegro), max(predicted_allegro))
ax.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='-', linewidth=2)

# Set font sizes directly
ax.set_xlabel('Actual Force (eV/Å)', fontsize=24)
ax.set_ylabel('Predicted Force (eV/Å)', fontsize=24)
ax.set_title('Parity Plot for ALLEGRO Forces on Intralayer Dataset', fontsize=28, pad=20)
ax.tick_params(axis='both', which='major', labelsize=20)

# Add grid and adjust axis limits
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)

# Add legend with font size
ax.legend(fontsize=16)
plt.rcParams['text.usetex'] = True
plt.rcParams.update({
    'font.size': 24,       # General font size
    'axes.titlesize': 24,  # Title font size
    'axes.labelsize': 24,  # Axis labels font size
    'xtick.labelsize': 24, # X-tick labels font size
    'ytick.labelsize': 24, # Y-tick labels font size
    'legend.fontsize': 16  # Legend font size
})
# Save the plot
plt.savefig('boxplot.png')
