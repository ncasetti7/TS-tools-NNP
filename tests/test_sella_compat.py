"""Tests for tstools_nnp.utils.sella_compat."""

import numpy as np
import pytest

pytest.importorskip("sella", reason="sella not installed")

from ase import Atoms  # noqa: E402
from ase.calculators.lj import LennardJones  # noqa: E402

from tstools_nnp.utils.sella_compat import IRC  # noqa: E402

# A first-order saddle of the 6-atom Lennard-Jones cluster (one imaginary mode).
LJ6_TS_POSITIONS = [
    [-0.819098, -0.456198, -0.221436],
    [0.233188, -0.375765, -0.567537],
    [-0.259168, 0.501054, -0.063452],
    [-0.504031, -0.163298, 0.804587],
    [0.800850, 0.577070, -0.410659],
    [0.548259, -0.082862, 0.458498],
]


def make_ts():
    atoms = Atoms("Ar6", positions=LJ6_TS_POSITIONS)
    atoms.calc = LennardJones(sigma=1.0, epsilon=1.0, rc=3.0)
    return atoms


def test_irc_leaves_a_converged_ts():
    """The IRC must displace along the imaginary mode even when |F| < fmax already holds.

    Regression for zadorlab/sella#82: ase >= 3.28 checks gradient_converged()
    rather than converged(), so sella's IRC reported convergence at step 0 and
    returned the input TS for both directions.
    """
    x0 = make_ts().get_positions()
    endpoints = {}
    for direction in ("forward", "reverse"):
        atoms = make_ts()
        irc = IRC(atoms, dx=0.1, keep_going=True, logfile=None)
        irc.run(fmax=0.05, steps=100, direction=direction)
        assert irc.nsteps > 0, f"{direction} IRC took no steps"
        assert np.abs(atoms.get_positions() - x0).max() > 1e-3, f"{direction} IRC did not leave the TS"
        endpoints[direction] = atoms.get_positions()

    # Forward and reverse have to descend to opposite sides of the saddle
    assert np.abs(endpoints["forward"] - endpoints["reverse"]).max() > 1e-2
