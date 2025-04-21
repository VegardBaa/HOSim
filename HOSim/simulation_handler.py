import numpy as np
import jax
from multiprocessing import Pool
import matplotlib.pyplot as plt
import os

from . import solver

def get_all_simulation_params(config):
    initial_params = []
    for simulation_name, simulation_params in config.items():
        for index in range(len(simulation_params["Hs"])):
            params = {
                "id" : simulation_name + "_" + str(index),
                "Hs" : simulation_params["Hs"][index],
                "Tp" : simulation_params["Tp"][index],
                "gamma" : simulation_params["gamma"][index],
                "depth" : simulation_params["depth"][index],
                "length" : simulation_params["length"],
                "modes" : simulation_params["modes"],
                "mHOS" : simulation_params["mHOS"],
                "gravity" : simulation_params["gravity"],
                "time" : simulation_params["time"],
                "step_size" : simulation_params["step_size"],
                "output_interval" : simulation_params["output_interval"],
                "Ta" : simulation_params["Ta"]
            }

            initial_params.append(params)
    return initial_params

def run(config):
    all_initial_params = get_all_simulation_params(config)

    os.makedirs("output", exist_ok=True)

    with Pool() as pool:
        pool.map(solver.run_simulation, all_initial_params)
    
    return 0
