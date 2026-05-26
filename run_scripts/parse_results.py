import argparse
import os

import yaml
from aimnet.models.base import load_model
from ase.io import read
from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit

from tstools_nnp.utils import interfaces


def get_args():
    """
    Parse command-line arguments.

    Returns:
    - argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--defaults-file", action="store", type=str, default="run_scripts/default_arguments.yaml")
    parser.add_argument("--target-dir", action="store", type=str, default="work_dir")
    parser.add_argument("--model-path", action="store", type=str, required=True)
    parser.add_argument("--save-results", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    # Get arguments
    args = get_args()
    with open(args.defaults_file, "r") as f:
        default_args = yaml.safe_load(f)

    # Load the model
    if default_args["model_type"] == "OMol":
        predictor = load_predict_unit(args.model_path, device="cpu")
        calc = FAIRChemCalculator(predictor, task_name="omol")
    elif default_args["model_type"] == "AIMNET":
        model, _ = load_model(args.model_path, device="cpu")
        calc = interfaces.AIMNet2ASECalculator(model)
    else:
        raise NotImplementedError(f"No model type {default_args['model_type']}!")

    unfinished_reactions = []
    if not os.path.isdir(args.target_dir):
        print(f"No results folder named {args.target_dir}!")
    else:
        os.chdir(args.target_dir)
        results = []
        reverse_results = []
        for dir in os.listdir("."):
            if os.path.isdir(dir):
                if default_args["model_type"] == "AIMNET":
                    calc.do_reset()
                min_barrier = 9999999999
                min_reverse_barrier = 9999999999
                ts_folder = f"{dir}/final_ts_guess/"
                rp_folder = f"{dir}/rp_geometries/"
                for i in range(20):
                    ts_file = f"{ts_folder}/ts_guess_{i}.xyz"
                    reactant_file = f"{rp_folder}/reactant_{i}.xyz"
                    product_file = f"{rp_folder}/product_{i}.xyz"
                    if os.path.exists(ts_file) and os.path.exists(reactant_file):
                        ts_atoms = read(ts_file)
                        reactant_atoms = read(reactant_file)
                        product_atoms = read(product_file)
                        ts_atoms.info["charge"] = 0
                        reactant_atoms.info["charge"] = 0
                        product_atoms.info["charge"] = 0
                        ts_atoms.calc = calc
                        reactant_atoms.calc = calc
                        product_atoms.calc = calc
                        ts_energy = ts_atoms.get_potential_energy()
                        barrier = (ts_energy - reactant_atoms.get_potential_energy()) * 23.06
                        reverse_barrier = (ts_energy - product_atoms.get_potential_energy()) * 23.06
                        if barrier < min_barrier:
                            min_barrier = barrier
                        if reverse_barrier < min_reverse_barrier:
                            min_reverse_barrier = reverse_barrier
                if min_barrier != 9999999999:
                    results.append((dir, min_barrier))
                if min_reverse_barrier != 9999999999:
                    reverse_results.append((dir, min_reverse_barrier))
                else:
                    unfinished_reactions.append(dir)
        if unfinished_reactions:
            print("The following reactions were not finished and have no barrier calculated:")
            print(unfinished_reactions)
        if args.save_results:
            with open("reverse_barriers.txt", "w") as f:
                # If possible, sort results by index
                try:
                    reverse_results.sort(key=lambda x: int(x[0].split("R")[-1]))
                except Exception as e:
                    print(f"Could not sort reverse results by index: {e}")
                    pass
                for result in reverse_results:
                    f.write(f"{result[0]} {result[1]}\n")
            with open("barriers.txt", "w") as f:
                # If possible, sort results by index
                try:
                    results.sort(key=lambda x: int(x[0].split("R")[-1]))
                except Exception as e:
                    print(f"Could not sort results by index: {e}")
                    pass
                for result in results:
                    f.write(f"{result[0]} {result[1]}\n")
        else:
            for result in results:
                print(result)
            for result in reverse_results:
                print(result)
