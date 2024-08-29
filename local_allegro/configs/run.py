from nequip.ase import NequIPCalculator

from nequip.scripts.deploy import load_deployed_model
from ase.io import read, write
import torch
from ase.calculators.calculator import Calculator, all_changes
import numpy as np
from nequip.scripts.deploy import load_deployed_model

from nequip.data import AtomicData, AtomicDataDict
import json

from ase.calculators.emt import EMT

DEVICE = "cuda:0"
def get_results_from_model_out(model_out):
    results = {}
    if AtomicDataDict.TOTAL_ENERGY_KEY in model_out:
        results["energy"] = (
            model_out[AtomicDataDict.TOTAL_ENERGY_KEY]
            .detach()
            .cpu()
            .numpy()
            .reshape(tuple())
        )
        results["free_energy"] = results["energy"]
    if AtomicDataDict.PER_ATOM_ENERGY_KEY in model_out:
        results["energies"] = (
            model_out[AtomicDataDict.PER_ATOM_ENERGY_KEY]
            .detach()
            .squeeze(-1)
            .cpu()
            .numpy()
        )
    if AtomicDataDict.FORCE_KEY in model_out:
        results["forces"] = model_out[AtomicDataDict.FORCE_KEY].detach().cpu().numpy()
    return results




class AllegroCalculator(Calculator):
    # Define the properties that the calculator can handle
    implemented_properties = ["energy", "energies", "forces", "free_energy"]

    def __init__(
        self,
        atoms,
        layer_symbols: list[str],
        model_file: str,
        device=DEVICE,
        **kwargs,
    ):
        """
        Initializes the AllegroCalculator with a given set of atoms, layer symbols, model file, and device.

        :param atoms: ASE atoms object.
        :param layer_symbols: List of symbols representing different layers in the structure.
        :param model_file: Path to the file containing the trained model.
        :param device: Device to run the calculations on, default is 'cpu'.
        :param kwargs: Additional keyword arguments for the base class.
        """
        self.atoms = atoms  # ASE atoms object
        self.atom_types = atoms.arrays[
            "atom_types"
        ]  # Extract atom types from atoms object
        self.device = device  # Device for computations

        # Flatten the layer symbols list
        self.layer_symbols = [
            symbol
            for sublist in layer_symbols
            for symbol in (sublist if isinstance(sublist, list) else [sublist])
        ]

        # Load the trained model and metadata
        self.model, self.metadata_dict = load_deployed_model(
            model_path=model_file, device=DEVICE, freeze=True
        )

        print(self.metadata_dict)

        

        # print(list(self.metadata_dict.keys()))
        # if int(self.metadata_dict["n_species"]) != len(self.layer_symbols):
        #    raise ValueError(
        #        "Mismatch between the number of atom types in model and provided layer symbols.",
        #        "Are you using an intralayer or interlayer model?",
        #    )

        # Determine unique atom types and their indices
        unique_types, inverse = np.unique(self.atom_types, return_inverse=True)

        # Map atom types to their relative positions in the unique_types array
        self.relative_layer_types = inverse

        # Ensure the number of unique atom types matches the number of layer symbols provided
        # if len(unique_types) != len(self.layer_symbols):
        #     raise ValueError(
        #         "Mismatch between the number of atom types and provided layer symbols."
        #     )

        # Initialize the base Calculator class with any additional keyword arguments
        Calculator.__init__(self, **kwargs)

    def calculate(self, atoms, properties=None, system_changes=all_changes):
        """
        Performs the calculation for the given atoms and properties.

        :param atoms: ASE atoms object to calculate properties for.
        :param properties: List of properties to calculate. If None, uses implemented_properties.
        :param system_changes: List of changes that have been made to the system since last calculation.
        """
        # Default to implemented properties if none are specified

        # Determine unique atom types and their indices
        self.atom_types = atoms.arrays["atom_types"]
        unique_types, inverse = np.unique(self.atom_types, return_inverse=True)

        # Map atom types to their relative positions in the unique_types array
        self.relative_layer_types = inverse

        if properties is None:
            properties = self.implemented_properties

        # Create a temporary copy of the atoms object
        tmp_atoms = atoms.copy()[:]
        tmp_atoms.calc = None  # Remove any attached calculator

        r_max = self.metadata_dict["r_max"]  # Maximum radius for calculations

        # Backup original atomic numbers and set new atomic numbers based on relative layer types
        original_atom_numbers = tmp_atoms.numbers.copy()
        tmp_atoms.set_atomic_numbers(self.relative_layer_types + 1)
        tmp_atoms.arrays["atom_types"] = self.relative_layer_types

        # Prepare atomic data for the model
        data = AtomicData.from_ase(
            atoms=tmp_atoms, r_max=r_max, include_keys=[AtomicDataDict.ATOM_TYPE_KEY]
        )

        # Remove energy keys from the data if present
        for k in AtomicDataDict.ALL_ENERGY_KEYS:
            if k in data:
                del data[k]

        # Move data to the specified device and convert to AtomicDataDict format
        data = data.to(self.device)
        data = AtomicData.to_AtomicDataDict(data)

        # Pass data through the model to get the output
        out = self.model(data)

        # Restore the original atomic numbers and types
        tmp_atoms.set_atomic_numbers(original_atom_numbers)
        tmp_atoms.arrays["atom_types"] = self.atom_types

        # Process the model output to get the desired results
        self.results = get_results_from_model_out(out)

def run_model(data_file, path_to_model):
    prediction_array = []
    for i in range(4320):
        atoms = read(data_file, index=i)
        calc = AllegroCalculator(
            atoms=atoms,
            layer_symbols=["Mo", "S", "S"],
            model_file=path_to_model,
            device=DEVICE
        )

        atoms.set_calculator(calc=calc)
        calc.calculate(atoms)
        prediction_array.append(calc.results['energy'])
    return prediction_array

def extract_all_energies(file_name):
    energies = []
    
    with open(file_name, 'r') as file:
        for line in file:
            if 'energy=' in line:
                parts = line.split()
                for part in parts:
                    if part.startswith('energy='):
                        energy = float(part.split('=')[1])
                        energies.append(energy)
    
    return energies

predicted = extract_all_energies('intra_dataset.xyz')

with open('mace1.json', 'w') as file:
    json.dump(predicted, file)

# with open('observed_intra.json', 'w') as file:
#     json.dump(observed, file)


# run_model('inter_mos2_lammps.xyz', 'inter-deployed.pth')

# NUMPY
ats_predicted = read("fix-plot.xyz", ":", format="extxyz")
forces_predicted = np.array([at.get_forces() for at in ats_predicted]).flatten() # Size (Nstrucs, nat_per_struc,3)
# print(forces_predicted.shape)
ats_observed = read("inter_dataset.xyz", ":", format="extxyz")
forces_observed = np.array([at.get_potential_energy() for at in ats_observed]).flatten()


# force_rmse = np.sqrt(np.mean((forces_predicted-forces_observed)**2))
# print(force_rmse)
with open('mace-final-intra.json', 'w') as file:
    json.dump(list(forces_observed), file)
# print(force_rmse)

# ats = read("inter_mos2_lammps.xyz", ":", format="extxyz")
# energies = np.array([at.get_potential_energy() for at in ats])
# print(energies)
# average = np.mean(energies)
# std = np.std(energies)
# print(f"Energy: {average}, Standard Deviation: {std}")



    



