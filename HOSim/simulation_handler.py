import numpy as np
import jax
from multiprocessing import Pool, Value, Process
import matplotlib.pyplot as plt
import os
import time

from tqdm import tqdm

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

_counter = None

def _init_worker(counter):
    global _counter
    _counter = counter

def _worker(params):
    solver.run_simulation(params, _counter)

def _monitor_progress(counter, total_updates):
    with tqdm(total=total_updates) as pbar:
        last = 0
        while last < total_updates:
            time.sleep(0.5)
            with counter.get_lock():
                current = counter.value
            delta = current - last
            if delta:
                pbar.update(delta)
                last = current

def run(config):
    all_initial_params = get_all_simulation_params(config)
    os.makedirs("output", exist_ok=True)

    total_updates = len(all_initial_params) * 10
    counter = Value('i', 0)

    monitor = Process(target=_monitor_progress, args=(counter, total_updates))
    monitor.start()

    with Pool(initializer=_init_worker, initargs=(counter,)) as pool:
        pool.map(_worker, all_initial_params)

    monitor.join()
    return 0