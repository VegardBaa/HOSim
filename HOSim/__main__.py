import sys
import os
import time

from . import io_utils, simulation_handler

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m HOSim <input_file.json>")
        sys.exit(1)

    input_path = sys.argv[1] + ".json"
    if not os.path.isfile(input_path):
        print(f"Input file '{input_path}' does not exist.")
        sys.exit(1)

    start_time = time.time()
    config = io_utils.load_json(input_path)
    result = simulation_handler.run(config)
    end_time = time.time()

    print(f"Simulation complete. Elapsed time: {end_time - start_time:.2f}s")

if __name__ == "__main__":
    main()