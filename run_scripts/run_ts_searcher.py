import os
import time
import argparse
import yaml
import torch
import logging
from fairchem.core.units.mlip_unit import load_predict_unit
from fairchem.core import FAIRChemCalculator
from tstools_nnp.path.path_generator import PathGenerator
from tstools_nnp.ts.ts_optimizer import TSOptimizer
from tstools_nnp.ts.ts_optimizer_batch import TSOptimizerBatch
from tstools_nnp.utils import interfaces, multiproc

def run_class_func(cls, func_name):
    func = getattr(cls, func_name)
    return func()

def get_args():
    """
    Parse command-line arguments.

    Returns:
    - argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--defaults-file', action='store', type=str, default='run_scripts/default_arguments.yaml')
    parser.add_argument('--input-file', action='store', type=str, default='data/reactions_am.txt')
    parser.add_argument('--target-dir', action='store', type=str, default='work_dir')
    parser.add_argument('--model-path', action='store', type=str, required=True)

    return parser.parse_args()

def obtain_transition_states(target_dir, reaction_list,
                             reactive_complex_factor_list_intermolecular,
                             reactive_complex_factor_list_intramolecular,
                             model_path, model_type, batch, batch_size,
                             num_workers, use_gpu, attempts, save_paths,
                             calc_hess):
    """
    Obtain transition states for a list of reactions.

    Parameters:
    - target_dir (str): Target directory.
    - reaction_list (list): List of reactions.
    - reactive_complex_factor_list_intermolecular (list): List of reactive complex factors for intermolecular reactions.
    - reactive_complex_factor_list_intramolecular (list): List of reactive complex factors for intramolecular reactions.
    - model_path (str): Path to the NNP model
    - model_type (str): Type of the NNP model (e.g., 'AIMNET', 'OMol')
    - batch (bool): Whether to process reactions in batches.
    - batch_size (int): Size of each batch.
    - num_workers (int): Number of CPU workers to use for calculations (only used if batch is False and there's no available GPU, setting to 0 will allocate all visible CPUs)
    - use_gpu (bool): Whether to use GPU acceleration if available
    - attempts (int): Number of attempts per reactive complex factor
    - save_paths (bool): Whether to save the reactive pathways
    - calc_hess (bool): Whether to calculate Hessians for TS optimizations and IRC calculations (not available for OMol)

    Returns:
    - list: List of successful reactions.
    """
    home_dir = os.getcwd()
    os.chdir(target_dir)
    results_directory = os.getcwd()

    # Load ase and batch calculators
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not batch:
        print("Warning: GPU acceleration is only available with batch processing. Setting batch to True.")
        batch = True
    if model_type == 'OMol':
        predictor = load_predict_unit(model_path, device=device)
        calc = FAIRChemCalculator(predictor, task_name="omol")
        batch_calc = interfaces.Fairchem(model_path)
        if calc_hess:
            print("Warning: Hessian calculations are not available for OMol models. Setting calc_hess to False.")
            calc_hess = False
    elif model_type == 'AIMNET':
        model = torch.jit.load(model_path, map_location=device)
        model.to(device)
        model.share_memory()
        calc = interfaces.AIMNet2ASECalculator(model)
        batch_calc = interfaces.AIMNET(model, device)
    
    ts_optimizers = []
    for rxn_id, reaction in reaction_list:
        if "." in reaction:
            reactive_complex_factors = reactive_complex_factor_list_intermolecular
        else:
            reactive_complex_factors = reactive_complex_factor_list_intramolecular
        path_generator = PathGenerator(rxn_id, reaction, calc)
        ts_optimizers.append(TSOptimizer(path_generator, reactive_complex_factors, attempts, save_paths, results_directory, calc_hess))

    results = []
    if batch:
        batches = [ts_optimizers[i:i + batch_size] for i in range(0, len(ts_optimizers), batch_size)]
        for batch in batches:
            ts_optimizer_batch = TSOptimizerBatch(batch, batch_calc, device)
            results.extend(ts_optimizer_batch.generate_ts_batch())
    else:
        inputs = [{"cls": ts_optimizer, "func_name": "generate_ts"} for ts_optimizer in ts_optimizers]
        if num_workers == 0:
            num_workers = min(os.cpu_count(), len(ts_optimizers))
        else:
            num_workers = min(num_workers, len(ts_optimizers))
        results = multiproc.parallel_run_proc(run_class_func, inputs, num_workers=num_workers)

    successful_reactions = [r for r in results if r is not None]
    os.chdir(home_dir)

    return successful_reactions


if __name__ == "__main__":
    # Suppress logging from autode and jax for cleaner output
    jax_logger = logging.getLogger("jax")
    autode_logger = logging.getLogger("autode")
    jax_logger.setLevel(logging.ERROR)
    autode_logger.setLevel(logging.ERROR)
    
    # Get arguments
    args = get_args()
    with open(args.defaults_file, 'r') as f:
        default_args = yaml.safe_load(f)
    
    # Make a directory for the results
    os.makedirs(args.target_dir, exist_ok=True)

    # Read in the reactions
    with open(args.input_file, 'r') as f:
        reaction_list = [line.strip().split() for line in f if line.strip()]

    start = time.time()
    # Obtain transition states
    successful_reactions = obtain_transition_states(
        target_dir=args.target_dir,
        reaction_list=reaction_list,
        reactive_complex_factor_list_intermolecular=default_args['reactive_complex_factor_list_intermolecular'],
        reactive_complex_factor_list_intramolecular=default_args['reactive_complex_factor_list_intramolecular'],
        model_path=args.model_path,
        model_type=default_args['model_type'],
        batch=default_args['batch'],
        batch_size=default_args['batch_size'],
        num_workers=default_args['num_workers'],
        use_gpu=default_args['use_gpu'],
        attempts=default_args['attempts'],
        save_paths=default_args['save_paths'],
        calc_hess=default_args['calc_hess']
    )
    print(f"Successful reactions: {successful_reactions}")
    end = time.time()
    print(f"Time taken: {end - start} seconds")