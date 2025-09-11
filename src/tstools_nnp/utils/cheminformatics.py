"""Module for converting between various molecule representations"""
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import autode as ade
from autode.conformers import conf_gen
import ase
from ase.io import write

def ade_mol_to_ase_atoms(ade_mol, charge=0, spin=1):
    '''
    Convert an autode Molecule to an ASE Atoms object

    Args:
        ade_mol (ade.Molecule): autode Molecule object

    Returns:
        ase.Atoms: ASE Atoms object
    '''
    symbols = [atom.atomic_symbol for atom in ade_mol.atoms]
    positions = [atom.coord for atom in ade_mol.atoms]
    atoms = ase.Atoms(symbols=symbols, positions=positions)
    atoms.info['charge'] = charge
    atoms.info['spin'] = spin
    return atoms

def make_mol(smi):
    '''
    Initialize a rdkit molecule from a SMILES string while preserving atom mapping

    Args:
        smi: str, SMILES string

    Returns:
        mol: rdkit.Chem.Mol
    '''
    ps = Chem.SmilesParserParams()
    ps.removeHs = False
    og = Chem.MolFromSmiles(smi, ps)
    fake_map = []
    for atom in og.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            fake_map.append(og.GetNumAtoms() - 1)
        else:
            fake_map.append(atom.GetAtomMapNum() - 1)
    indices_order = sorted(range(len(fake_map)), key=lambda x: fake_map[x])
    mol = Chem.RenumberAtoms(og, indices_order)
    return mol

def check_mol(rwmol):
    mol = rwmol.GetMol()
    Chem.SanitizeMol(mol)
    a = AllChem.EmbedMolecule(mol, maxAttempts=10000000)
    assert a == 0
    return mol

def get_metal_atoms(mol):
    """
    Get the indices of metal atoms in the reactant molecule.

    Returns:
    list: List of indices of metal atoms.
    """
    metal_list = ['Al', 'Sb', 'Ag', 'As', 'Ba', 'Be', 'Bi', 'Cd', 'Ca', 'Cr', 'Co', 'Cu', 'Au', 'Fe', 
              'Pb', 'Mg', 'Mn', 'Hg', 'Mo', 'Ni', 'Pd', 'Pt', 'K', 'Rh', 'Rb', 'Ru', 'Sc', 'Ag', 
              'Na', 'Sr', 'Ta', 'Tl', 'Th', 'Ti', 'U', 'V', 'Y', 'Zn', 'Zr']
    metal_atoms = []

    for atom in mol.GetAtoms():
        if atom.GetSymbol() in metal_list:
            metal_atoms.append(atom.GetIdx())
    
    return metal_atoms

def get_bonds(mol):
    """
    Get the bond strings of a molecule.

    Args:
        mol (Chem.Mol): Molecule.

    Returns:
        set: Set of bond strings.
    """
    bonds = set()
    for bond in mol.GetBonds():
        atom_1 = mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomMapNum()
        atom_2 = mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomMapNum()

        if atom_1 < atom_2:
            bonds.add((atom_1, atom_2))
        else:
            bonds.add((atom_2, atom_1))

    return bonds

def check_identity_both(reac_atoms, prod_atoms, reac_mol, prod_mol, charge, multiplicity, required_dist_change=0.03, metal_bond_length=3.5):
    '''
    Check if both the reactant and product conformers correspond to the input molecules

    Args:
        reac_atoms (ase.Atoms): ASE Atoms object of the reactant conformer
        prod_atoms (ase.Atoms): ASE Atoms object of the product conformer
        reac_mol (rdkit.Chem.rdchem.Mol): Reactant molecule
        prod_mol (rdkit.Chem.rdchem.Mol): Product molecule
        required_dist_change (float): Required change in distance for bond formation/breaking
        metal_bond_length (float): Maximum bond length for metal bonds
    
    Returns:
        bool: True if the conformers correspond to the input molecules, False otherwise
    '''
    halides = ['F', 'Cl', 'Br', 'I']
    reactant_bonds = list(get_bonds(reac_mol))
    product_bonds = list(get_bonds(prod_mol))
    write("reactant_check.xyz", reac_atoms, format='xyz')
    write("product_check.xyz", prod_atoms, format='xyz')
    ade_mol_r = ade.Molecule("reactant_check.xyz", name="reactant", charge=charge, mult=multiplicity)
    ade_mol_p = ade.Molecule("product_check.xyz", name="product", charge=charge, mult=multiplicity)
    metal_atoms = get_metal_atoms(reac_mol)
    metal_atoms = [atom + 1 for atom in metal_atoms]
    halides = [atom.GetIdx() + 1 for atom in reac_mol.GetAtoms() if atom.GetSymbol() in halides]
    # Ignore bonds that involve metals (and halide-halide bonds) by removing from the sets of bonds
    rdmol_reactant_bonds = []
    ade_reactant_bonds = []
    rdmol_product_bonds = []
    ade_product_bonds = []
    reactant_metal_bonds = []
    product_metal_bonds = []
    for bond in reactant_bonds:
        if bond[0] in metal_atoms or bond[1] in metal_atoms:
            reactant_metal_bonds.append(bond)
        if bond[0] not in metal_atoms and bond[1] not in metal_atoms:
            if bond[0] not in halides or bond[1] not in halides:
                rdmol_reactant_bonds.append(bond)

    for bond in ade_mol_r.graph.edges:
        bond = (bond[0] + 1, bond[1] + 1)
        if bond[0] not in metal_atoms and bond[1] not in metal_atoms:
            if bond[0] not in halides or bond[1] not in halides:
                ade_reactant_bonds.append(bond) 

    for bond in product_bonds:
        if bond[0] in metal_atoms or bond[1] in metal_atoms:
            product_metal_bonds.append(bond)
        if bond[0] not in metal_atoms and bond[1] not in metal_atoms:
            if bond[0] not in halides or bond[1] not in halides:
                rdmol_product_bonds.append(bond)

    for bond in ade_mol_p.graph.edges:
        bond = (bond[0] + 1, bond[1] + 1)
        if bond[0] not in metal_atoms and bond[1] not in metal_atoms:
            if bond[0] not in halides or bond[1] not in halides:
                ade_product_bonds.append(bond)

    # Check whether the atoms bonded to the metal are within the metal bond length
    for bond in reactant_metal_bonds:
        atom_1 = ade_mol_r.atoms[bond[0] - 1]
        atom_2 = ade_mol_r.atoms[bond[1] - 1]
        dist = np.linalg.norm(atom_1.coord - atom_2.coord)
        if dist > metal_bond_length:
            return False

    for bond in product_metal_bonds:
        atom_1 = ade_mol_p.atoms[bond[0] - 1]
        atom_2 = ade_mol_p.atoms[bond[1] - 1]
        dist = np.linalg.norm(atom_1.coord - atom_2.coord)
        if dist > metal_bond_length:
            return False
    # Find the metal bonds that have changed
    reactant_metal_bonds = set(reactant_metal_bonds)
    product_metal_bonds = set(product_metal_bonds)
    reactant_metal_bonds_diff = reactant_metal_bonds.difference(product_metal_bonds)
    product_metal_bonds_diff = product_metal_bonds.difference(reactant_metal_bonds)

    # In reactant_metal_bonds_diff, check the difference in distance between bonds for reactant and product
    # If the change in bond distance is less than the required_dist_change, return False
    for bond in reactant_metal_bonds_diff:
        atom_1 = ade_mol_r.atoms[bond[0] - 1]
        atom_2 = ade_mol_r.atoms[bond[1] - 1]
        reac_bond_length = np.linalg.norm(atom_1.coord - atom_2.coord)
        atom_1 = ade_mol_p.atoms[bond[0] - 1]
        atom_2 = ade_mol_p.atoms[bond[1] - 1]
        prod_bond_length = np.linalg.norm(atom_1.coord - atom_2.coord)
        if reac_bond_length - prod_bond_length > -required_dist_change:
            return False
    
    # In product_metal_bonds_diff, check the difference in distance between bonds for reactant and product
    # If the change in bond distance is less than the negative of required_dist_change, return False
    for bond in product_metal_bonds_diff:
        atom_1 = ade_mol_r.atoms[bond[0] - 1]
        atom_2 = ade_mol_r.atoms[bond[1] - 1]
        reac_bond_length = np.linalg.norm(atom_1.coord - atom_2.coord)
        atom_1 = ade_mol_p.atoms[bond[0] - 1]
        atom_2 = ade_mol_p.atoms[bond[1] - 1]
        prod_bond_length = np.linalg.norm(atom_1.coord - atom_2.coord)
        if reac_bond_length - prod_bond_length < required_dist_change:
            return False
    
    return set(rdmol_reactant_bonds) == set(ade_reactant_bonds) and set(rdmol_product_bonds) == set(ade_product_bonds)

def autode_conf_gen(mol, charge=0, spin=1):
    '''
    Generate conformers with autode
    '''
    AllChem.EmbedMolecule(mol, maxAttempts=1000000)
    Chem.MolToXYZFile(mol, 'tmp.xyz')
    ade_mol = ade.Molecule('tmp.xyz', name='tmp', charge=charge)
    confs = [conf_gen.get_simanl_conformer(ade_mol)]
    ade_mol.confs = confs
    return ade_mol_to_ase_atoms(ade_mol, charge=charge, spin=spin)

def path_to_xyz_file(path, atomic_symbols, file_name):
    '''
    Write a reaction path to an XYZ file

    Args:
        path (list): List of conformers, each conformer is a list of coordinates
        atomic_symbols (list): List of atomic symbols
        file_name (str): Name of the output XYZ file
    '''
    with open(file_name, 'w') as f:
        for conf in path:
            f.write(f"{len(conf)}\n\n")
            for i, coord in enumerate(conf):
                f.write(f"{atomic_symbols[i]} {coord[0]} {coord[1]} {coord[2]}\n")

def get_owning_mol_dict(smiles):
    """
    Create a dictionary mapping atom map numbers to the index of the molecule to which they belong.

    Parameters:
    reaction_smiles (str): Reaction SMILES string.

    Returns:
    dict: A dictionary where keys are atom map numbers and values are the corresponding molecule indices.
    """
    ps = Chem.SmilesParserParams()
    ps.removeHs = False
    molecules = [Chem.MolFromSmiles(smi, ps) for smi in smiles.split('.')]
    owning_mol_dict = {}

    for mol_index, mol in enumerate(molecules):
        for atom in mol.GetAtoms():
            owning_mol_dict[atom.GetAtomMapNum()] = mol_index

    return owning_mol_dict

def get_bonds(mol):
    """
    Get the bond strings of a molecule.

    Args:
        mol (Chem.Mol): Molecule.

    Returns:
        set: Set of bond strings.
    """
    bonds = set()
    for bond in mol.GetBonds():
        atom_1 = mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomMapNum()
        atom_2 = mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomMapNum()

        if atom_1 < atom_2:
            bonds.add((atom_1, atom_2))
        else:
            bonds.add((atom_2, atom_1))

    return bonds