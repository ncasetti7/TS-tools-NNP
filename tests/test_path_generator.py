"""Tests for tstools_nnp.path.path_generator.PathGenerator."""

from unittest.mock import MagicMock

import numpy as np
import pytest

rdkit = pytest.importorskip("rdkit", reason="rdkit not installed")

from tstools_nnp.path.path_generator import PathGenerator  # noqa: E402

# Simple bond-breaking reaction: HF dissociation
HF_DISSOCIATION = "[H:1][F:2]>>[H:1].[F:2]"

# Simple bond-forming reaction: H2 association (reverse of dissociation)
H2_ASSOCIATION = "[H:1].[H:2]>>[H:1][H:2]"

# Organic substitution (intramolecular-style: single reactant fragment)
INTRAMOLECULAR = "[CH3:1][CH2:2][F:3]>>[CH2:2]=[CH3:1].[F-:3]"

# Bimolecular reaction (two reactant fragments separated by ".")
BIMOLECULAR = "[CH3:1][Cl:2].[F-:3]>>[CH3:1][F:3].[Cl-:2]"


@pytest.fixture
def mock_calc():
    return MagicMock()


class TestPathGeneratorInit:
    def test_creates_instance(self, mock_calc):
        pg = PathGenerator("r0", HF_DISSOCIATION, mock_calc)
        assert pg is not None

    def test_rxn_id_stored(self, mock_calc):
        pg = PathGenerator("rxn42", HF_DISSOCIATION, mock_calc)
        assert pg.rxn_id == "rxn42"

    def test_reaction_smiles_stored(self, mock_calc):
        pg = PathGenerator("r0", HF_DISSOCIATION, mock_calc)
        assert pg.reaction_smiles == HF_DISSOCIATION

    def test_calc_stored(self, mock_calc):
        pg = PathGenerator("r0", HF_DISSOCIATION, mock_calc)
        assert pg.calc is mock_calc


class TestGetActiveBonds:
    def test_hf_dissociation_broken_bond(self, mock_calc):
        pg = PathGenerator("r0", HF_DISSOCIATION, mock_calc)
        # The H-F bond (map nums 1 and 2) should be broken
        assert len(pg.broken_bonds) == 1
        assert len(pg.formed_bonds) == 0

    def test_h2_association_formed_bond(self, mock_calc):
        pg = PathGenerator("r0", H2_ASSOCIATION, mock_calc)
        assert len(pg.formed_bonds) == 1
        assert len(pg.broken_bonds) == 0

    def test_bimolecular_bonds(self, mock_calc):
        pg = PathGenerator("r0", BIMOLECULAR, mock_calc)
        assert len(pg.formed_bonds) >= 1
        assert len(pg.broken_bonds) >= 1

    def test_returns_sets(self, mock_calc):
        pg = PathGenerator("r0", HF_DISSOCIATION, mock_calc)
        assert isinstance(pg.formed_bonds, set)
        assert isinstance(pg.broken_bonds, set)


class TestOrganometallicCheck:
    def test_organic_not_organometallic(self, mock_calc):
        pg = PathGenerator("r0", BIMOLECULAR, mock_calc)
        assert pg.reaction_is_organometallic is False

    def test_hf_not_organometallic(self, mock_calc):
        pg = PathGenerator("r0", HF_DISSOCIATION, mock_calc)
        assert pg.reaction_is_organometallic is False


class TestChargeMultiplicity:
    def test_neutral_charge(self, mock_calc):
        pg = PathGenerator("r0", HF_DISSOCIATION, mock_calc)
        assert pg.charge == 0

    def test_singlet_multiplicity(self, mock_calc):
        pg = PathGenerator("r0", HF_DISSOCIATION, mock_calc)
        assert pg.multiplicity == 1


class TestResetOptState:
    def test_reset_clears_cycle(self, mock_calc):
        pg = PathGenerator("r0", HF_DISSOCIATION, mock_calc)
        pg.cycle = 5
        pg.reset_opt_state()
        assert pg.cycle == 0

    def test_reset_clears_conformer(self, mock_calc):
        pg = PathGenerator("r0", HF_DISSOCIATION, mock_calc)
        pg.reactant_conformer = np.zeros((2, 3))
        pg.reset_opt_state()
        assert pg.reactant_conformer is None

    def test_reset_clears_flags(self, mock_calc):
        pg = PathGenerator("r0", HF_DISSOCIATION, mock_calc)
        pg.crude_complete = True
        pg.refined_complete = True
        pg.reset_opt_state()
        assert pg.crude_complete is False
        assert pg.refined_complete is False


class TestDeterminePotential:
    def test_zero_potential_at_equilibrium(self, mock_calc):
        """When atom distances match constraints exactly, potential should be ~0."""
        pg = PathGenerator("r0", HF_DISSOCIATION, mock_calc)
        coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        constraints = {(0, 1): 1.5}
        potentials = pg.determine_potential([coords], constraints, force_constant=1.0)
        assert abs(potentials[0]) < 1e-10

    def test_nonzero_potential_off_equilibrium(self, mock_calc):
        pg = PathGenerator("r0", HF_DISSOCIATION, mock_calc)
        coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        constraints = {(0, 1): 1.5}
        potentials = pg.determine_potential([coords], constraints, force_constant=1.0)
        assert potentials[0] > 0
