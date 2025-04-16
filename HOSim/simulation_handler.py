import numpy as np
import jax

from . import solver

def get_all_simulation_params(config):
    initial_params = []
    for simulation_name, simulation_params in config.items():
        for index in range(len(simulation_params["Hs"])):
            f = jax.jit(solver.f, static_argnums=(2, 3, 4, 5, 6))
            rk4_step = jax.jit(solver.rk4_step, static_argnums=(2, 3, 4, 5, 6))

            params = {
                "name" : simulation_name,
                "Hs" : simulation_params["Hs"][index],
                "Tp" : simulation_params["Tp"][index],
                "depth" : simulation_params["depth"][index],
                "length" : simulation_params["length"],
                "modes" : simulation_params["modes"],
                "gravity" : simulation_params["gravity"],
                "time" : simulation_params["time"],
                "step_size" : simulation_params["step_size"],
                "output_interval" : simulation_params["output_interval"],
                "f_jit" : f,
                "rk4_step_jit" : rk4_step
            }

            initial_params.append(params)
    return initial_params

def run(config):
    all_initial_params = get_all_simulation_params(config)

    
    
    return 0
