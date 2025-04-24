import json5
import numpy as np
import h5py
import glob
import os
import re
from collections import Counter

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

def join():
    files = glob.glob(os.path.join("output", "*.h5"))
    stripped = set(re.sub(r'^.*?([^\\]+)_\d+\.h5$', r'\1', f) for f in files)

    for sim in stripped:
        eta_hat = []
        phi_hat = []

        Hs = []
        Tp = []
        gamma = []
        depth = []

        for file in files:
            if re.fullmatch(rf"output\\{re.escape(sim)}_\d+\.h5", file):
                with h5py.File(file, "r") as f:
                    eta_hat.append(f["eta_hat"][:])
                    phi_hat.append(f["phi_hat"][:])

                    Hs.append(f.attrs["Hs"][()])
                    Tp.append(f.attrs['Tp'][()])
                    gamma.append(f.attrs['gamma'][()])
                    depth.append(f.attrs['Hs'][()])

        with h5py.File(f"output/{sim}.h5", "w") as f_out:
            f_out.create_dataset("eta_hat", data=np.array(eta_hat))
            f_out.create_dataset("phi_hat", data=np.array(phi_hat))

            f_out.create_dataset("Hs", data=np.array(Hs))
            f_out.create_dataset("Tp", data=np.array(Tp))
            f_out.create_dataset("gamma", data=np.array(gamma))
            f_out.create_dataset("depth", data=np.array(depth))

            with h5py.File(f"output/{sim}_0.h5", "r") as f:
                f_out.attrs["length"] = f.attrs["length"]
                f_out.attrs["gravity"] = f.attrs["gravity"]
                f_out.attrs["modes"] = f.attrs["modes"]
                f_out.attrs["mHOS"] = f.attrs["mHOS"]
                f_out.attrs["tMax"] = f.attrs["time"]
                f_out.attrs["step_size"] = f.attrs["step_size"]
                f_out.attrs["output_interval"] = f.attrs["output_interval"]
                f_out.attrs["Ta"] = f.attrs["Ta"]

                f_out.create_dataset("time", data=np.linspace(0, f.attrs["time"], len(eta_hat[0])))
                f_out.create_dataset("x", data=np.linspace(0, f.attrs["length"], 2*f.attrs["modes"]))

    for file in files:
        os.remove(file)