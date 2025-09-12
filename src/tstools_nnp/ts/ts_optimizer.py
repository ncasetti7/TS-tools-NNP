import os
import ase
from ase.io import read, write
from tstools_nnp.utils import calculations, cheminformatics

class TSOptimizer():
    def __init__(self, path_generator, reactive_complex_factors, attempts, save_paths, results_directory, calc_hess):
        self.path_generator = path_generator
        self.reactive_complex_factors = reactive_complex_factors
        self.attempts = attempts
        self.save_paths = save_paths
        self.results_directory = results_directory
        self.calc_hess = calc_hess

    def check_ts_guesses(self, energies, paths, atomic_symbols, index):
        ts_guesses = self.determine_and_filter_local_maxima(energies, paths)
        for ts_guess in ts_guesses:
            ts_guess_atoms = ase.Atoms(symbols=atomic_symbols, positions=ts_guess)
            ts_guess_atoms.info['charge'] = self.path_generator.charge
            ts_guess_atoms.info['spin'] = self.path_generator.multiplicity
            print(f"Reaction {self.path_generator.rxn_id}: TS Optimization")
            try:
                ts_atoms = calculations.ts_optimize_geometry(ts_guess_atoms, self.path_generator.calc, calc_hessian=self.calc_hess)
            except Exception as e:
                print(f"TS optimization failed: {e}")
                ts_atoms = None
            if ts_atoms is None:
                continue
            write(f"temp_ts.xyz", ts_atoms, format='xyz')
            print(f"Reaction {self.path_generator.rxn_id}: IRC")
            try:
                first, last = calculations.calc_irc(ts_atoms, self.path_generator.calc)
            except Exception as e:
                print(f"IRC calculation failed: {e}")
                continue
            if first is None:
                continue
            if cheminformatics.check_identity_both(first, last, self.path_generator.reactant_rdkit_mol, self.path_generator.product_rdkit_mol, self.path_generator.charge, self.path_generator.multiplicity):
                print(f"Reaction {self.path_generator.rxn_id}: Found valid TS!")
                optimized_reactant = calculations.constrained_optimization(first, [], self.path_generator.calc, fmax=0.01)
                optimized_product = calculations.constrained_optimization(last, [], self.path_generator.calc, fmax=0.01)
                write(f"rp_geometries/reactant_{index}.xyz", optimized_reactant, format='xyz')
                write(f"rp_geometries/product_{index}.xyz", optimized_product, format='xyz')
                os.system(f"mv temp_ts.xyz final_ts_guess/ts_guess_{index}.xyz")
                return True
            elif cheminformatics.check_identity_both(last, first, self.path_generator.reactant_rdkit_mol, self.path_generator.product_rdkit_mol, self.path_generator.charge, self.path_generator.multiplicity):
                print(f"Reaction {self.path_generator.rxn_id}: Found valid TS!")
                optimized_reactant = calculations.constrained_optimization(last, [], self.path_generator.calc, fmax=0.01)
                optimized_product = calculations.constrained_optimization(first, [], self.path_generator.calc, fmax=0.01)
                write(f"rp_geometries/reactant_{index}.xyz", optimized_reactant, format='xyz')
                write(f"rp_geometries/product_{index}.xyz", optimized_product, format='xyz')
                os.system(f"mv temp_ts.xyz final_ts_guess/ts_guess_{index}.xyz")
                return True
            else:
                print("TS guess did not connect the correct reactant and product.")
        return False

    def generate_ts(self):
        # Create directory for this reaction
        os.chdir(self.results_directory)
        os.makedirs(self.path_generator.rxn_id, exist_ok=True)
        os.chdir(self.path_generator.rxn_id)
        # Make directories to store final results
        os.makedirs("rp_geometries", exist_ok=True)
        os.makedirs("final_ts_guess", exist_ok=True)
        if self.save_paths:
            os.makedirs("path_dir", exist_ok=True)

        index = 0
        for reactive_complex_factor in self.reactive_complex_factors:
            for _ in range(self.attempts):
                self.path_generator.set_reactive_complex_factor(reactive_complex_factor)
                self.path_generator.reset_opt_state()
                energies, _, paths = self.path_generator.get_path()
                if energies is not None:
                    break
            if energies is not None:
                if self.save_paths:
                    cheminformatics.path_to_xyz_file(paths, self.path_generator.atomic_symbols, f"path_dir/path_{reactive_complex_factor}_{index}.xyz")
                if self.check_ts_guesses(energies, paths, self.path_generator.atomic_symbols, index):
                    return self.path_generator.rxn_id
                index += 1
        return None

    def determine_and_filter_local_maxima(self, true_energies, trajectory):
        """
        Determine and filter local maxima in the path.

        Parameters:
        - true_energies (list): List of true energy values.
        - path_xyz_files (list): List of path XYZ files.
        - charge: Charge information.

        Returns:
        - list: List of ranked transition state guess files based on energy.
        """
        # Find local maxima in path
        indices_local_maxima = self.find_local_max_indices(list(true_energies))

        # Validate the local maxima and store their energy values
        ts_guesses = []
        energies = []
        for index in indices_local_maxima:
            #ts_guess_file, _ = validate_ts_guess(path_xyz_files[index], self.reaction_dir, self.freq_cut_off, charge)
            ts_guess = trajectory[index]
            energies.append(true_energies[index])
            ts_guesses.append(ts_guess)

        # Sort guesses based on energy
        sorted_guess_dict = sorted(zip(ts_guesses, energies), key=lambda x: x[1], reverse=True)
        ranked_guess_files = [item[0] for item in sorted_guess_dict]

        return ranked_guess_files
    
    def find_local_max_indices(self, numbers):
        """
        Find indices of local maxima in a list of numbers.

        Parameters:
        - numbers (list): List of numbers.

        Returns:
        - list: List of indices corresponding to local maxima.
        """
        local_max_indices = []
        for i in range(len(numbers) - 2, 0, -1):
            if numbers[i] > numbers[i - 1] and numbers[i] > numbers[i + 1]:
                local_max_indices.append(i)
        return local_max_indices
    