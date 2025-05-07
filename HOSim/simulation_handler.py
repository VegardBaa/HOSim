import numpy as np
import jax
from multiprocessing import Pool, Value, Process, cpu_count
import matplotlib.pyplot as plt
import os
import time

from tqdm import tqdm

from . import solver, io_utils

def get_all_simulation_params(config):
    initial_params = []
    workers = 1
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
                "Ta" : simulation_params["Ta"],
            }

            if "2d" in simulation_params:
                if simulation_params["2d"]:
                    params["2d"] = True
                    params["beta"] = simulation_params["beta"][index]
                else:
                    params["2d"] = False
            else:
                params["2d"] = False

            if "workers" in simulation_params:
                workers = simulation_params["workers"]

            initial_params.append(params)
    return initial_params, workers

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
    all_initial_params, workers = get_all_simulation_params(config)
    os.makedirs("output", exist_ok=True)
    for filename in os.listdir("output"):
        file_path = os.path.join("output", filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    total_updates = len(all_initial_params) * 10
    counter = Value('i', 0)

    monitor = Process(target=_monitor_progress, args=(counter, total_updates))
    monitor.start()

    num_workers = min(workers, cpu_count())
    with Pool(processes=num_workers, initializer=_init_worker, initargs=(counter,)) as pool:
        pool.map(_worker, all_initial_params)

    monitor.join()

    io_utils.join()
    
    return 0