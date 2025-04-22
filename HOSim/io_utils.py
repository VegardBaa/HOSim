import json5
import numpy as np
import h5py

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

def save_results(params, results):
    filename = f"output/{params["id"]}.h5"
    with h5py.File(filename, "w") as f:
        for key, val in params.items():
            if isinstance(val, (int, float, str)):
                f.attrs[key] = val

        f.create_dataset("eta_hat", data=results[:, :params["modes"]+1])
        f.create_dataset("phi_hat", data=results[:, params["modes"]+1:])