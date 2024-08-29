import yaml
import json
import subprocess


def convert_yaml_to_json(file_name):
    # Read the yaml file
    with open(file_name, 'r') as file:
        configuration = yaml.safe_load(file)

    # Dump the data from the yaml to a json file
    with open('config.json', 'w') as json_file:
        json.dump(configuration, json_file)


def convert_json_to_yaml(file_name):
    # Read the json file and save it as configuration
    with open(file_name, 'r') as file:
        configuration = json.load(file)

    # Convert the json file into a yaml file and save it as config.yaml
    with open('config.yaml', 'w') as yaml_file:
        yaml.dump(configuration, yaml_file)


def update_value(file_name, key, value):
    with open(file_name, 'r') as file:
        data = json.load(file)

    # Update the value in the json file
    if key in data:
        data[key] = value
    else:
        print("Key" + " " + key + " " +  "not found in the JSON data")

    # Write back the dictionary to the json file
    with open('config.json', 'w') as file:
        json.dump(data, file, indent=4)


def run_training():
    command = ["nequip-train", "config.yaml"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(result.stdout)
        print(result.stderr)
    except subprocess.CalledProcessError as e:
        print("Error: " + str(e.stderr))


def ask_input():
    key_list = []
    values_list = []
    num_training = int(input("Input the number of training to perform: "))
    convert_yaml_to_json('config.yaml')

    run_name = input("Input the run name: ")

    with open('config.json', 'r') as f:
        default_params = json.load(f)

    while True:
        user_input = input("Input the parameter to change from the YAML file and the values for each training "
                           "separated by spaces: ")

        # Parsing the user inputs
        parsed_input = user_input.split(" ")

        if len(parsed_input[1:]) == num_training:
            parameter = parsed_input[0]
            values = [int(i) for i in parsed_input[1:]]
            key_list.append(parameter)
            values_list.append(values)

            cont = ""
            while True:
                cont = input("Any other parameter to change from the default YAML file (y/n): ")
                if cont != 'y' and cont != 'n':
                    continue
                break

            if cont == 'n':
                break

        else:
            print("Number of values does not match the intended number of training")

    return key_list, values_list, run_name


def run_multiple_trainings(key_list, values_list, run_name):
    num_parameters = len(key_list)
    num_training = len(values_list[0])

    for i in range(num_training):
        with open('config.json', 'r') as file:
            data = json.load(file)

        data["run_name"] = run_name + str(i)

        with open('config.json', 'w') as file:
            json.dump(data, file, indent=4)

        for j in range(num_parameters):
            parameter = key_list[j]
            update_value('config.json', parameter, values_list[j][i])

        convert_json_to_yaml('config.json')
        # Run the training on the updated YAML file
        run_training()


def run_program():
    key_list, values_list, run_name = ask_input()
    run_multiple_trainings(key_list, values_list, run_name)


run_program()




