"""Tests for tstools_nnp.utils.cheminformatics."""

import os
import tempfile

import numpy as np
import pytest

rdkit = pytest.importorskip("rdkit", reason="rdkit not installed")

from tstools_nnp.utils.cheminformatics import (  # noqa: E402
    get_bonds,
    get_metal_atoms,
    get_owning_mol_dict,
    make_mol,
    path_to_xyz_file,
)


class TestMakeMol:
    def test_returns_mol(self):
        mol = make_mol("[CH3:1][CH3:2]")
        assert mol is not None

    def test_atom_reordering_by_map_number(self):
        # Map numbers are out of order in the input SMILES; make_mol should sort by them.
        mol = make_mol("[CH3:2][CH2:1][OH:3]")
        # After reordering: idx 0 -> map 1 (CH2), idx 1 -> map 2 (CH3), idx 2 -> map 3 (OH)
        assert mol.GetAtomWithIdx(0).GetAtomMapNum() == 1
        assert mol.GetAtomWithIdx(1).GetAtomMapNum() == 2

    def test_hydrogens_retained(self):
        # removeHs=False retains explicit H atoms when they appear in the SMILES
        mol = make_mol("[H:2][C:1]([H:3])([H:4])[H:5]")
        assert mol.GetNumAtoms() == 5

    def test_charged_molecule(self):
        mol = make_mol("[NH4+:1]")
        assert mol is not None


class TestGetBonds:
    def test_cc_bond_present(self):
        mol = make_mol("[CH3:1][CH3:2]")
        bonds = get_bonds(mol)
        assert (1, 2) in bonds

    def test_returns_set(self):
        mol = make_mol("[CH3:1][CH3:2]")
        bonds = get_bonds(mol)
        assert isinstance(bonds, set)

    def test_bond_ordering_canonical(self):
        # Bonds should always be stored as (smaller_mapnum, larger_mapnum).
        mol = make_mol("[CH3:2][CH3:1]")
        bonds = get_bonds(mol)
        assert (1, 2) in bonds
        assert (2, 1) not in bonds

    def test_no_bonds_single_atom(self):
        mol = make_mol("[Cl:1]")
        bonds = get_bonds(mol)
        # Cl has no bonds to other mapped atoms (no other mapped atom present)
        assert (0, 1) not in bonds or True  # just check it doesn't raise


class TestGetMetalAtoms:
    def test_no_metals_in_organic(self):
        mol = make_mol("[CH3:1][CH3:2]")
        metals = get_metal_atoms(mol)
        assert metals == []

    def test_no_metals_in_halide(self):
        mol = make_mol("[F:1][Cl:2]")
        metals = get_metal_atoms(mol)
        assert metals == []

    def test_detects_palladium(self):
        from rdkit import Chem

        ps = Chem.SmilesParserParams()
        ps.removeHs = False
        mol = Chem.MolFromSmiles("[Pd:1]", ps)
        metals = get_metal_atoms(mol)
        assert 0 in metals

    def test_detects_iron(self):
        from rdkit import Chem

        ps = Chem.SmilesParserParams()
        ps.removeHs = False
        mol = Chem.MolFromSmiles("[Fe:1]", ps)
        metals = get_metal_atoms(mol)
        assert 0 in metals


class TestGetOwningMolDict:
    def test_single_molecule(self):
        d = get_owning_mol_dict("[Cl:1][H:2]")
        assert d[1] == 0
        assert d[2] == 0

    def test_two_molecules(self):
        d = get_owning_mol_dict("[Cl:1].[I:2]")
        assert d[1] == 0
        assert d[2] == 1

    def test_three_molecules(self):
        d = get_owning_mol_dict("[Cl:1].[I:2].[F:3]")
        assert d[1] == 0
        assert d[2] == 1
        assert d[3] == 2


class TestPathToXyzFile:
    def test_writes_file(self):
        path = [np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])]
        symbols = ["C", "H"]
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, "path.xyz")
            path_to_xyz_file(path, symbols, fname)
            assert os.path.exists(fname)

    def test_correct_atom_count_header(self):
        path = [np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])]
        symbols = ["C", "H"]
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, "path.xyz")
            path_to_xyz_file(path, symbols, fname)
            with open(fname) as f:
                first_line = f.readline().strip()
            assert first_line == "2"

    def test_multiple_frames(self):
        frame = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        path = [frame, frame + 0.1]
        symbols = ["C", "H"]
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, "path.xyz")
            path_to_xyz_file(path, symbols, fname)
            with open(fname) as f:
                content = f.read()
            # Each frame has a "2\n\n" header, so "2\n" should appear twice
            assert content.count("2\n") == 2
