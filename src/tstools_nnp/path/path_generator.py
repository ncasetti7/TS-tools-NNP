import random
from itertools import product

import ase
from ase.constraints import Hookean
from pysisyphus.constants import ANG2BOHR
from rdkit import Chem
from scipy.spatial import distance_matrix

from tstools_nnp.utils import calculations, cheminformatics
from tstools_nnp.utils.interfaces import AIMNet2ASECalculator

metal_list = [
    "Al",
    "Sb",
    "Ag",
    "As",
    "Ba",
    "Be",
    "Bi",
    "Cd",
    "Ca",
    "Cr",
    "Co",
    "Cu",
    "Au",
    "Fe",
    "Pb",
    "Mg",
    "Mn",
    "Hg",
    "Mo",
    "Ni",
    "Pd",
    "Pt",
    "K",
    "Rh",
    "Rb",
    "Ru",
    "Sc",
    "Ag",
    "Na",
    "Sr",
    "Ta",
    "Tl",
    "Th",
    "Ti",
    "U",
    "V",
    "Y",
    "Zn",
    "Zr",
]


class PathGenerator:
    # Constants
    FC_CRUDE_LOWER_BOUND = 0.1
    FC_CRUDE_UPPER_BOUND = 4.0
    FC_CRUDE_INCREMENT = 0.1
    FC_CRUDE_ATTEMTPS = 1

    FC_REFINED_LOWER_BOUND = 0.09
    FC_REFINED_UPPER_BOUND = 0.03
    FC_REFINED_INCREMENT = 0.01
    FC_REFINED_ATTEMPTS = 1

    MIN_FC_LOWER_BOUND = 0.009
    MIN_FC_UPPER_BOUND = 0.005
    FC_INCREMENT = 0.001
    MAX_FORCE_CONSTANT = 0.1
    POTENTIAL_THRESHOLD = 0.005

    STRETCH_FACTOR_LOWER_BOUND = 1.0
    STRETCH_FACTOR_UPPER_BOUND = 1.3

    OPT_STATES = {
        "initial_optimization": "Initial Optimization",
        "stretched_optimization": "Stretched Optimization",
        "reactive_optimization": "Reactive Optimization",
        "reaction_path": "Reaction Path Optimization",
        "completed": "Completed",
        "failed": "Failed",
    }

    CONVERSION_FACTOR = 97.1738

    def __init__(self, rxn_id, reaction_smiles, calc, reactive_complex_factor=2.0):
        self.rxn_id = rxn_id
        self.reaction_smiles = reaction_smiles
        self.calc = calc
        self.reactive_complex_factor = reactive_complex_factor
        self.opt_state = self.OPT_STATES["initial_optimization"]

        self.reactant_rdkit_mol = cheminformatics.make_mol(self.reaction_smiles.split(">>")[0])
        self.product_rdkit_mol = cheminformatics.make_mol(self.reaction_smiles.split(">>")[1])
        self.charge = Chem.GetFormalCharge(self.reactant_rdkit_mol)
        num_radicals = sum([atom.GetNumRadicalElectrons() for atom in self.reactant_rdkit_mol.GetAtoms()])
        self.multiplicity = num_radicals + 1

        self.reactant_conformer = None
        self.product_conformer = None
        self.atomic_symbols = None
        self.numbers = None

        self.formed_bonds, self.broken_bonds = self.get_active_bonds_from_mols()

        self.atom_map_dict = {atom.GetAtomMapNum(): atom.GetIdx() for atom in self.reactant_rdkit_mol.GetAtoms()}
        self.atom_idx_dict = {atom.GetIdx(): atom.GetAtomMapNum() for atom in self.reactant_rdkit_mol.GetAtoms()}

        self.owning_dict_rsmiles = cheminformatics.get_owning_mol_dict(reaction_smiles.split(">>")[0])
        self.owning_dict_psmiles = cheminformatics.get_owning_mol_dict(reaction_smiles.split(">>")[1])

        self.reaction_is_organometallic = self.check_if_reaction_organometallic()
        self.cycle = 0
        self.reactant_conformer = None
        self.crude_complete = False
        self.refined_complete = False

    def get_path(self, max_iterations=100):
        while not (self.opt_state == self.OPT_STATES["completed"] or self.opt_state == self.OPT_STATES["failed"]):
            opt_dict = self.initialize_optimization()
            trajectory = self.run_optimization(opt_dict)
            self.update_state(trajectory)
            max_iterations -= 1
            if max_iterations == 0:
                print("Max iterations reached")
                break
        if self.opt_state == self.OPT_STATES["completed"]:
            print(f"Path optimization for reaction {self.rxn_id} completed successfully.")
            return self.energies, self.potentials, self.paths
        else:
            print(f"Path optimization for reaction {self.rxn_id} failed.")
            return [], [], []

    def initialize_optimization(self):
        opt_dict = {}
        if self.opt_state == self.OPT_STATES["failed"] or self.opt_state == self.OPT_STATES["completed"]:
            return opt_dict
        if self.opt_state == self.OPT_STATES["initial_optimization"]:
            opt_dict["coords"] = self.set_conformer(prod=True)
            opt_dict["constraints"] = None
            opt_dict["fc"] = None
        elif self.opt_state == self.OPT_STATES["stretched_optimization"]:
            opt_dict["coords"] = self.set_conformer()
            opt_dict["constraints"] = self.get_formation_constraints_stretched()
            opt_dict["fc"] = min(self.fc, self.MAX_FORCE_CONSTANT)
        elif self.opt_state in (self.OPT_STATES["reactive_optimization"], self.OPT_STATES["reaction_path"]):
            opt_dict["coords"] = self.reactive_conformer
            opt_dict["constraints"] = self.formation_constraints
            opt_dict["fc"] = self.fc
        opt_dict["charge"] = self.charge
        opt_dict["numbers"] = self.numbers
        return opt_dict

    def run_optimization(self, opt_dict):
        atoms = ase.Atoms(symbols=self.atomic_symbols, positions=opt_dict["coords"])
        atoms.info["charge"] = self.charge
        atoms.info["spin"] = self.multiplicity
        atoms.calc = self.calc
        constraints = []
        if opt_dict["constraints"] is not None:
            for key, val in opt_dict["constraints"].items():
                atom1, atom2 = key
                distance = val
                constraint = Hookean(a1=atom1, a2=atom2, rt=distance, k=opt_dict["fc"] * self.CONVERSION_FACTOR)
                constraints.append(constraint)
        opt_results = calculations.constrained_optimization(atoms, constraints, self.calc)
        return self.process_trajectory(opt_results)

    def process_trajectory(self, opt_results):
        """
        Process the trajectory from the optimization results.

        Args:
            opt_results: The results from the optimization.

        Returns:
            trajectory: A list of coordinates representing the trajectory.
        """
        trajectory = []
        for atoms in opt_results:
            trajectory.append(atoms.get_positions())
        return trajectory

    def set_conformer(self, prod=False):
        """
        Generate a conformer for the reactant if not already present or time for a reset.

        Returns:
        None
        """
        if prod:
            atoms = cheminformatics.autode_conf_gen(self.product_rdkit_mol, charge=self.charge, spin=self.multiplicity)
            self.product_conformer = atoms.get_positions()
            self.atomic_symbols = atoms.get_chemical_symbols()
            self.numbers = atoms.get_atomic_numbers()
            return self.product_conformer
        if self.reactant_conformer is None or self.cycle > 1:
            atoms = cheminformatics.autode_conf_gen(self.reactant_rdkit_mol, charge=self.charge, spin=self.multiplicity)
            self.reactant_conformer = atoms.get_positions()
            self.atomic_symbols = atoms.get_chemical_symbols()
            self.numbers = atoms.get_atomic_numbers()
        return self.reactant_conformer

    def get_formation_constraints_stretched(self):
        """
        Get stretched formation constraints for bonds that are to be stretched.

        Returns:
        dict: Dictionary of stretched formation constraints.
        """
        formation_constraints_to_stretch = self.get_bonds_to_stretch()
        formation_constraints_stretched = {}
        if self.reactive_complex_factor < 0.01:
            return formation_constraints_stretched

        for bond, original_distance in self.formation_constraints.items():
            if bond in formation_constraints_to_stretch:
                stretch_factor = random.uniform(
                    PathGenerator.STRETCH_FACTOR_LOWER_BOUND * self.reactive_complex_factor,
                    PathGenerator.STRETCH_FACTOR_UPPER_BOUND * self.reactive_complex_factor,
                )
                formation_constraints_stretched[bond] = stretch_factor * original_distance

        return formation_constraints_stretched

    def get_bonds_to_stretch(self):
        """
        Get the set of bonds to be stretched based on formation constraints.

        Returns:
        set: Set of bonds to be stretched.
        """
        bonds_to_stretch = set()

        for bond in self.formation_constraints.keys():
            atom_1, atom_2 = bond
            owner_1 = self.owning_dict_rsmiles[self.atom_idx_dict[atom_1]]
            owner_2 = self.owning_dict_rsmiles[self.atom_idx_dict[atom_2]]

            if owner_1 != owner_2:
                bonds_to_stretch.add(bond)

        # If no intermolecular bonds found, consider all bonds
        if not bonds_to_stretch:
            bonds_to_stretch = set(self.formation_constraints.keys())

        return bonds_to_stretch

    def get_active_bonds_from_mols(self):
        """
        Identify formed and broken bonds between reactant and product molecules.

        Returns:
        set: Formed bonds in the product.
        set: Broken bonds in the reactant.
        """
        reactant_bonds = cheminformatics.get_bonds(self.reactant_rdkit_mol)
        product_bonds = cheminformatics.get_bonds(self.product_rdkit_mol)

        formed_bonds = product_bonds - reactant_bonds
        broken_bonds = reactant_bonds - product_bonds

        return formed_bonds, broken_bonds

    def check_if_reaction_organometallic(self):
        """
        Check if the reactant molecule contains any organometallic atoms.

        Returns:
        - bool: True if organometallic atoms are present, False otherwise.

        Notes:
        - The function examines the atomic symbols of atoms in the reactant molecule.
        - It checks if any of the symbols match those in the 'metal_list'.
        """
        symbol_list = [atom.GetSymbol() for atom in self.reactant_rdkit_mol.GetAtoms()]

        for symbol in symbol_list:
            if symbol in metal_list:
                return True
            else:
                continue

        return False

    def determine_potential(self, all_coords, constraints, force_constant):
        """
        Determine the potential energy for a set of coordinates based on distance constraints and a force constant.

        Args:
            all_coords (list): A list of coordinate arrays.
            constraints (dict): A dictionary specifying the atom index pairs and their corresponding distances.
            force_constant (float): The force constant to apply to the constraints.

        Returns:
            list: A list of potential energy values.
        """
        potentials = []
        for coords in all_coords:
            potential = 0
            dist_matrix = distance_matrix(coords, coords)
            for key, val in constraints.items():
                actual_distance = dist_matrix[key[0], key[1]] - val
                potential += force_constant * ANG2BOHR * actual_distance**2
            potentials.append(potential)
        return potentials

    def get_energies_for_path(self, path_coords):
        """
        Get the energies of a path.

        Parameters:
        - path_coords (list): List of coordinates for each step in the path.

        Returns:
        - list: List of energies corresponding to each step in the path.
        """
        energies = []
        if isinstance(self.calc, AIMNet2ASECalculator):
            self.calc.do_reset()
        for coords in path_coords:
            atoms = ase.Atoms(symbols=self.atomic_symbols, positions=coords)
            atoms.info["charge"] = self.charge
            atoms.info["spin"] = self.multiplicity
            atoms.calc = self.calc
            energy = atoms.get_potential_energy() * 23.0609  # Convert eV to kcal/mol
            energies.append(energy)
        return energies

    def get_optimal_distances(self, coords):
        """
        Calculate optimal distances for formed bonds in the product
        (add additional distance constraints if organometallic system).

        Returns:
        dict: Dictionary of optimal distances for formed bonds.
        """
        optimal_distances = {}
        # product_smiles = [smi for smi in self.product_smiles.split('.')]
        # product_molecules = [Chem.MolFromSmiles(smi, ps) for smi in self.product_smiles.split('.')]
        formed_bonds = self.formed_bonds

        atoms_involved_in_formed_bonds = []

        mol_dict = {atom.GetAtomMapNum(): atom.GetIdx() for atom in self.product_rdkit_mol.GetAtoms()}

        for bond in formed_bonds:
            atom_i = int(bond[0])
            atom_j = int(bond[1])

            idx1, idx2 = self.atom_map_dict[atom_i], self.atom_map_dict[atom_j]

            # mol, mol_dict, smiles = self.get_mol_and_mol_dict(atom_i, atom_j, product_molecules, product_smiles)
            current_bond_length = self.obtain_current_distance(coords, mol_dict[atom_i], mol_dict[atom_j])

            optimal_distances[idx1, idx2] = current_bond_length
            # for metal-containing bonds, add the atoms that involve main group elements
            # to the atoms_involved_in_formed_bonds list
            # if mol.GetAtomWithIdx(mol_dict[atom_i]).GetSymbol() not in metal_list and \
            #      mol.GetAtomWithIdx(mol_dict[atom_j]).GetSymbol() in metal_list:
            #    atoms_involved_in_formed_bonds.append(atom_i)
            # if mol.GetAtomWithIdx(mol_dict[atom_i]).GetSymbol() in metal_list and \
            #      mol.GetAtomWithIdx(mol_dict[atom_j]).GetSymbol() not in metal_list:
            #    atoms_involved_in_formed_bonds.append(atom_j)

        if self.reaction_is_organometallic:
            for atom_i, atom_j in list(product(atoms_involved_in_formed_bonds, repeat=2)):
                if (min(atom_i, atom_j), max(atom_i, atom_j)) in self.broken_bonds:
                    idx1, idx2 = self.atom_map_dict[atom_i], self.atom_map_dict[atom_j]
                    current_distance = self.obtain_current_distance(coords, mol_dict[atom_i], mol_dict[atom_j])
                    optimal_distances[min(idx1, idx2), max(idx1, idx2)] = current_distance
                    break
                else:
                    continue

        return optimal_distances

    def obtain_current_distance(self, coords, atom_i, atom_j):
        """
        Calculate the current distance between two atoms in the molecule.

        Parameters:
        - coords (numpy.ndarray): The coordinates of the atoms in the molecule.
        - atom_i (int): Index of the first atom.
        - atom_j (int): Index of the second atom.

        Returns:
        - current_distance (float): The distance between the specified atoms in the molecule.

        Notes:
        - The function internally uses the obtain_dist_matrix method to compute the distance matrix.
        """
        dist_matrix = distance_matrix(coords, coords)
        current_distance = dist_matrix[atom_i, atom_j]

        return current_distance

    def update_state(self, trajectory):
        # Start by initially optimizing the product to get the formation constraints
        if self.opt_state == self.OPT_STATES["initial_optimization"]:
            self.fc = PathGenerator.FC_CRUDE_LOWER_BOUND
            self.formation_constraints = self.get_optimal_distances(trajectory[-1])
            self.opt_state = self.OPT_STATES["stretched_optimization"]
        # Before any path optimization get a reactive complex, when going from screening to pathing, log the minimal_fc
        elif self.opt_state == self.OPT_STATES["stretched_optimization"]:
            # Set the reactive conformer and determine whether we are screening fc's or checking the reaction path
            self.reactive_conformer = trajectory[-1]
            # print(self.formation_constraints)
            # Set the new force constant depending on the result of the optimization
            if not self.refined_complete:
                self.opt_state = self.OPT_STATES["reactive_optimization"]
            else:
                self.opt_state = self.OPT_STATES["reaction_path"]

        # When screening fc's, use a path optimization to calcualte the potential and increment fc accordingly
        elif self.opt_state == self.OPT_STATES["reactive_optimization"]:
            potentials = self.determine_potential(trajectory, self.formation_constraints, self.fc)
            # print(self.formation_constraints)
            # print(len(potentials), potentials[-1])
            # Set the new force constant depending on the result of the optimization
            if potentials[-1] < PathGenerator.POTENTIAL_THRESHOLD:
                # Move to path optimization in the case of just completing the fc screen
                if self.crude_complete:
                    self.refined_complete = True
                    self.opt_state = self.OPT_STATES["reaction_path"]
                    self.minimal_fc = self.fc
                    self.fc = self.fc - PathGenerator.MIN_FC_LOWER_BOUND
                else:
                    self.crude_complete = True
                    self.fc = self.fc - PathGenerator.FC_REFINED_LOWER_BOUND
            else:
                if not self.crude_complete:
                    self.fc = self.fc + PathGenerator.FC_CRUDE_INCREMENT
                else:
                    self.fc = self.fc + PathGenerator.FC_REFINED_INCREMENT
            # Set the state back to stretched optimization to prepare for next path optimization
            self.opt_state = self.OPT_STATES["stretched_optimization"]

        # When testing the reaction path, check the potential and optimization results
        # to determine whether to increment fc or finish
        elif self.opt_state == self.OPT_STATES["reaction_path"]:
            potentials = self.determine_potential(trajectory, self.formation_constraints, self.fc)
            # print(len(potentials), potentials[-1])
            if potentials[-1] > PathGenerator.POTENTIAL_THRESHOLD:
                self.opt_state = self.OPT_STATES["stretched_optimization"]
                self.fc = self.fc + PathGenerator.FC_INCREMENT
                if self.fc > self.minimal_fc + PathGenerator.MIN_FC_UPPER_BOUND:
                    self.opt_state = self.OPT_STATES["failed"]
            else:
                reactant_atoms = ase.Atoms(symbols=self.atomic_symbols, positions=trajectory[0])
                product_atoms = ase.Atoms(symbols=self.atomic_symbols, positions=trajectory[-1])
                if not cheminformatics.check_identity_both(
                    reactant_atoms,
                    product_atoms,
                    self.reactant_rdkit_mol,
                    self.product_rdkit_mol,
                    self.charge,
                    self.multiplicity,
                ):
                    if self.cycle > 2:
                        self.opt_state = self.OPT_STATES["failed"]
                    self.cycle += 1
                    self.opt_state = self.OPT_STATES["stretched_optimization"]
                    self.fc = self.fc + PathGenerator.FC_INCREMENT
                    if self.fc > self.minimal_fc + PathGenerator.MIN_FC_UPPER_BOUND:
                        self.opt_state = self.OPT_STATES["failed"]
                else:
                    self.opt_state = self.OPT_STATES["completed"]
                    self.potentials = potentials
                    self.paths = trajectory
                    self.energies = self.get_energies_for_path(trajectory)
        # Print the current state for debugging purposes
        print(f"Reaction {self.rxn_id}: {self.opt_state}")

    def set_reactive_complex_factor(self, factor):
        self.reactive_complex_factor = factor

    def reset_opt_state(self):
        self.opt_state = self.OPT_STATES["initial_optimization"]
        self.cycle = 0
        self.reactant_conformer = None
        self.crude_complete = False
        self.refined_complete = False
