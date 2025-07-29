from math import exp
from os.path import exists
from json import dump, load

def sigmoid(x):
    return 1 / (1 + exp(-x))

def sigmoid_derivative(y):
    return y * (1 - y)

def save_example_to_json(file_path, inputs, label):
    data = {"input": inputs, "label": label}
    if exists(file_path):
        with open(file_path, "r") as f:
            all_data = load(f)
    else:
        all_data = []
    all_data.append(data)
    with open(file_path, "w") as f:
        dump(all_data, f, indent=2)

def load_dataset_from_json(file_path):
    if not exists(file_path):
        return []
    with open(file_path, "r") as f:
        data = load(f)
    return [(item["input"], item["label"]) for item in data]