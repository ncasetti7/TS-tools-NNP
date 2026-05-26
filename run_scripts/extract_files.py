import argparse
import os

import torch
import yaml
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
        model = torch.jit.load(args.model_path, map_location="cpu")
        calc = interfaces.AIMNet2ASECalculator(model)
    else:
        raise NotImplementedError(f"No model type {default_args['model_type']}!")

    if not os.path.isdir(args.target_dir):
        print(f"No results folder named {args.target_dir}!")
    else:
        os.chdir(args.target_dir)
        results = []
        for dir in os.listdir("."):
            if os.path.isdir(dir):
                if default_args["model_type"] == "AIMNET":
                    calc.do_reset()
                min_barrier = 9999999999
                min_ts_file = None
                min_reactant_file = None
                ts_folder = f"{dir}/final_ts_guess/"
                rp_folder = f"{dir}/rp_geometries/"
                for i in range(20):
                    ts_file = f"{ts_folder}/ts_guess_{i}.xyz"
                    reactant_file = f"{rp_folder}/reactant_{i}.xyz"
                    if os.path.exists(ts_file) and os.path.exists(reactant_file):
                        ts_atoms = read(ts_file)
                        reactant_atoms = read(reactant_file)
                        ts_atoms.calc = calc
                        reactant_atoms.calc = calc
                        barrier = (ts_atoms.get_potential_energy() - reactant_atoms.get_potential_energy()) * 23.06
                        if barrier < min_barrier:
                            min_barrier = barrier
                            min_ts_file = ts_file
                            min_reactant_file = reactant_file
                if min_barrier != 9999999999:
                    results.append((dir, min_ts_file, min_reactant_file))

        os.makedirs("final_ts_reactants", exist_ok=True)
        for result in results:
            dir = result[0]
            ts_file = result[1]
            reactant_file = result[2]
            os.system(f"scp {ts_file} final_ts_reactants/{dir}_ts.xyz")
            os.system(f"scp {reactant_file} final_ts_reactants/{dir}_reac.xyz")
