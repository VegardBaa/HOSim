import sys
import os

from . import input_utils, simulation_handler

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m HOSim <input_file.json>")
        sys.exit(1)

    input_path = sys.argv[1] + ".json"
    if not os.path.isfile(input_path):
        print(f"Input file '{input_path}' does not exist.")
        sys.exit(1)

    config = input_utils.load_json(input_path)
    result = simulation_handler.run(config)

    print("Simulation complete.")

if __name__ == "__main__":
    main()