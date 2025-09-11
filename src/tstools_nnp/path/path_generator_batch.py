import torch
import numpy as np
from tstools_nnp.path.path_generator import PathGenerator
from tstools_nnp.utils.batch_calculations import GeometryCalculation, ConstrainedCalculator, FIRE, HookeanConstraint
from tstools_nnp.utils.interfaces import AIMNET

class PathGeneratorBatch:

    CONVERSION_FACTOR = 97.1738

    def __init__(self, path_generators, nnp_calculator, device):
        """
        Initialize a PathGeneratorBatch object.

        Parameters:
        - path_generators (list): List of PathGenerator objects.

        Returns:
        None
        """
        self.path_generators = path_generators
        self.states = {}
        for path_generator in self.path_generators:
            self.states[path_generator.rxn_id] = PathGenerator.OPT_STATES['initial_optimization']
        self.nnp_calculator = nnp_calculator
        self.device = device
        self.pad = False
        if type(self.nnp_calculator) == AIMNET:
            self.pad = True

    def get_paths(self, max_iterations=100):
        """
        Run the path generator for each reaction in the batch.

        Returns:
        energies (dict): Dictionary of energies for each reaction.
        potentials (dict): Dictionary of potentials for each reaction.
        paths (dict): Dictionary of paths for each reaction.
        """
        energies = {}
        potentials = {}
        paths = {}

        while not all([(state == PathGenerator.OPT_STATES["completed"] or state == PathGenerator.OPT_STATES["failed"]) for state in self.states.values()]):
            if max_iterations == 0:
                break
            opt_dict = self.initialize_optimizations()
            trajectories = self.run_optimizations(opt_dict)
            self.update_states(trajectories)
            max_iterations -= 1

        for path_generator in self.path_generators:
            if self.states[path_generator.rxn_id] == PathGenerator.OPT_STATES["completed"]:
                energies[path_generator.rxn_id] = path_generator.energies
                potentials[path_generator.rxn_id] = path_generator.potentials
                paths[path_generator.rxn_id] = path_generator.paths
        
        return energies, potentials, paths
 
    def initialize_optimizations(self):
        """
        Initialize optimizations for each reaction in the batch.

        Returns:
        opt_dict (dict): Dictionary of optimization parameters.
        """
        opt_dict = {}
        for path_generator in self.path_generators:
            if path_generator.opt_state == PathGenerator.OPT_STATES["failed"] or path_generator.opt_state == PathGenerator.OPT_STATES["completed"]:
                continue
            opt_dict[path_generator.rxn_id] = path_generator.initialize_optimization()
        return opt_dict
    
    def run_optimizations(self, opt_dict):
        """
        Run a batched optimizaiton

        Parameters:
        - opt_dict (dict): Dictionary of optimization parameters.

        Returns:
        None
        """
        cons = []
        coords = []
        numbers = []
        charge = torch.zeros(len(opt_dict.values()), device=self.device)
        for b, opt_info in enumerate(opt_dict.values()):
            coords.append(opt_info['coords'])
            charge[b] = opt_info['charge']
            numbers.append(opt_info['numbers'])
            if opt_info['constraints'] is not None:
                for key, val in opt_info['constraints'].items():
                    k = opt_info['fc'] * self.CONVERSION_FACTOR
                    cons.append(HookeanConstraint(k=k, r0=val, b=b, a1=key[0], a2=key[1]))
        coords = self.batch_and_pad_coords(coords)
        numbers = self.batch_and_pad_numbers(numbers)
        calc = self.nnp_calculator
        cons_calc = ConstrainedCalculator(calc, cons)
        calculation = GeometryCalculation(cons_calc, max_cycles=300, conv_thresh="gau")
        opt = FIRE(calculation)
        final_coords= opt.run(coords, numbers, charge, dump_traj=True)
        trajectories = self.process_trajectories(final_coords)
        return trajectories
    
    def get_energies_for_path(self, path, path_generator):
        """
        Get the energies of a path.

        Parameters:
        - path (list): List of conformers.
        - path_generator (PathGenerator): Path generator object.

        Returns:
        - list: List of energies.
        """
        # Format the path for the calculator
        coords = torch.zeros((len(path), len(path[0]), 3), device=self.device)
        numbers = torch.zeros((len(path), len(path[0])), device=self.device)
        charge = torch.zeros(len(path), device=self.device)
        for i, conf in enumerate(path):
            coords[i, :len(conf)] = torch.tensor(conf)
            numbers[i, :len(conf)] = path_generator.numbers
            charge[i] = path_generator.charge
        calc = self.nnp_calculator
        calculation = GeometryCalculation(calc, max_cycles=500, conv_thresh="gau")
        energies = calculation.calc_energies(coord=coords, numbers=numbers, charges=charge)
        if calc.device == 'cuda':
            energies = [energy.cpu().detach().numpy() for energy in energies]
        return energies
    
    def batch_and_pad_coords(self, coords):
        '''
        Batch and pad coordinates for a batch of conformers

        Parameters:
        - coords (list): List of conformer coordinates.

        Returns:
        - torch.Tensor: Padded tensor of coordinates.
        '''
        max_atoms = max([len(coord) for coord in coords])
        padded_coords = torch.zeros((len(coords), max_atoms, 3), device=self.device)
        for i, coord in enumerate(coords):
            padded_coords[i, :len(coord)] = torch.tensor(coord)
        return padded_coords
    
    def batch_and_pad_numbers(self, numbers):
        '''
        Batch and pad atomic numbers for a batch of conformers

        Parameters:
        - numbers (list): List of atomic numbers.

        Returns:
        - torch.Tensor: Padded tensor of atomic numbers.
        '''
        max_atoms = max([len(num) for num in numbers])
        padded_numbers = torch.zeros((len(numbers), max_atoms), device=self.device, dtype=torch.long)
        for i, num in enumerate(numbers):
            padded_numbers[i, :len(num)] = torch.tensor(num)
        return padded_numbers
    
    def process_trajectories(self, opt_results):
        """
        Take the optimization results and make them a list of trajectories

        Parameters:
        - opt_results (dict): Dictionary of optimization results.

        Returns:
        - list: List of trajectories.
        """
        # The opt results are a list of batched geometries from each cycle, separate them into trajectories
        # Need to check if the trajectory is no longer changing by comparing the last two geometries
        # Also unpad the geometries
        if self.device == 'cuda':
            opt_results = [result.cpu() for result in opt_results]
        trajectories = []
        for i in range(len(opt_results[0])):
            trajectory = []
            for j in range(len(opt_results)):
                if j == 0:
                    trajectory.append(self.unpad_coords(opt_results[j][i].detach().numpy()))
                else:
                    if not np.allclose(opt_results[j][i].detach().numpy(), opt_results[j-1][i].detach().numpy(), atol=1e-8):
                        trajectory.append(self.unpad_coords(opt_results[j][i].detach().numpy()))
            trajectories.append(trajectory)
        return trajectories

    def unpad_coords(self, coords):
        """
        Unpad coordinates.

        Parameters:
        - coords (np.array): Padded array of coordinates.

        Returns:
        - np.array: Unpadded array of coordinates.
        """
        return coords[~np.all(coords == 0, axis=1)]

    def update_states(self, trajectories):
        """
        Update the states and values of the reactions in the batch.

        Parameters:
        - opt_results (dict): Dictionary of optimization results.

        Returns:
        None
        """
        # Iterate through the results and calculate the potentials for each path
        path_generators = [path_generator for path_generator in self.path_generators if (path_generator.opt_state != PathGenerator.OPT_STATES["completed"] and path_generator.opt_state != PathGenerator.OPT_STATES["failed"])]
        for path_generator, trajectory in zip(path_generators, trajectories):
            path_generator.update_state(trajectory)
            self.states[path_generator.rxn_id] = path_generator.opt_state