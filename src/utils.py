from math import exp
from os.path import exists, dirname
from os import makedirs
from json import dump, load

def sigmoid(x):
    return 1 / (1 + exp(-x))

def sigmoid_derivative(y):
    return y * (1 - y)

def save_example_to_json(file_path, inputs, label):
    data = {"input": inputs, "label": label}
    if not exists(file_path):
        makedirs(dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write("[]")
        all_data = []
    else:
        with open(file_path, "r") as f:
            all_data = load(f)
    all_data.append(data)
    with open(file_path, "w") as f:
        dump(all_data, f, indent=2)

def load_dataset_from_json(file_path):
    if not exists(file_path):
        return []
    with open(file_path, "r") as f:
        data = load(f)
    return [(item["input"], item["label"]) for item in data]