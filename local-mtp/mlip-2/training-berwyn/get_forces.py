import re
import json

def parse_configurations(filename):
    with open(filename, 'r') as file:
        data = file.read()
    
    # Split the data by configurations
    configs = re.split(r'(?=BEGIN_CFG)', data)
    
    result = []

    for config in configs:
        if 'BEGIN_CFG' in config and 'END_CFG' in config:
            # Extract the relevant part between BEGIN_CFG and END_CFG
            config_body = config.split('END_CFG')[0]
            
            # Find the lines that contain the atom data
            atom_data_lines = re.findall(r'^\s*\d+\s*\d+\s*-?\d+\.\d+\s*-?\d+\.\d+\s*-?\d+\.\d+\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)', config_body, re.MULTILINE)
            
            if len(atom_data_lines) == 12:
                # Convert lines to float and organize into sublists
                forces = [[float(fx), float(fy), float(fz)] for fx, fy, fz in atom_data_lines]
                result.append(forces)

    return result

# Example usage
filename = 'kontol.cfg'  # Replace with the path to your file
all_forces = parse_configurations(filename)
with open('kontol.json', 'w') as file:
    json.dump(list(all_forces), file)