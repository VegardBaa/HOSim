import json5
import numpy as np

def lists_to_numpy(obj):
    for key, value in obj.items():
        if isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
            obj[key] = np.array(value, dtype=np.float64)
    return obj

def load_json(input_path):
    with open(input_path, "r") as file:
        config = json5.load(file, object_hook=lists_to_numpy)

    for sim in config.values():
        sim["depth"][sim["depth"] <= 0] = np.inf
        
        if sim["Ta"] < 0:
            sim["Ta"] = 0

    return config