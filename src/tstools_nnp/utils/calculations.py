"""This module is for running compuational chemistry calculations with AIMNet2ASECalculator"""

import contextlib
import os

from ase.io import read
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS, FIRE
from rdkit import Chem
from sella import Sella

from tstools_nnp.utils.interfaces import AIMNet2ASECalculator
from tstools_nnp.utils.sella_compat import IRC

EV_TO_KCAL = 23.0605


def optimize_geometry(atoms, calc, max_cycles=1000, convergence=0.01, verbose=False):
    """
    Optimize a geometry with pysisyphus

    Args:
        atoms (ase.Atoms): geometry to optimize
        calc (ase.calculator): ASE calculator to use for optimization
        max_cycles (int): maximum number of optimization cycles
        convergence (float): convergence criteria for forces
        verbose (bool): whether to print optimization output

    Returns:
        atoms (ase.Atoms): optimized geometry
    """
    if isinstance(calc, AIMNet2ASECalculator):
        calc.do_reset()
    atoms.calc = calc
    # Create an optimizer object
    opt = FIRE(atoms)
    # Run the optimization
    if verbose:
        opt.run(fmax=convergence, steps=max_cycles)
    else:
        with open(os.devnull, "w", encoding="utf-8") as f, contextlib.redirect_stdout(f):
            opt.run(fmax=convergence, steps=max_cycles)

    return atoms


def ts_optimize_geometry(atoms, calc, max_cycles=1000, convergence=0.01, calc_hessian=False, verbose=False):
    """
    Optimize a geometry with pysisyphus

    Args:
        atoms (ase.Atoms): geometry to optimize
        calc (ase.calculator): ASE calculator to use for optimization
        max_cycles (int): maximum number of optimization cycles
        rmse_convergence (float): convergence criteria for forces
        calc_hessian (bool): whether to calculate the hessian
        verbose (bool): whether to print optimization output

    Returns:
        atoms (ase.Atoms): optimized geometry
    """
    if isinstance(calc, AIMNet2ASECalculator):
        calc.do_reset()
    atoms.calc = calc
    # Create a Sella object
    if verbose:
        logfile = "-"
    else:
        logfile = None
    if calc_hessian:
        opt = Sella(atoms, logfile=logfile, hessian_function=calc.get_hessian)
    else:
        opt = Sella(atoms, logfile=logfile)
    # Run the optimization
    opt.run(fmax=convergence, steps=max_cycles)
    if opt.converged is False:
        print("Sella optimization did not converge")
        return None

    return atoms


def calc_irc(atoms, calc, max_cycles=1000, convergence=0.01, verbose=False):
    """
    Run an IRC calculation with pysisyphus

    Args:
        atoms (ase.Atoms): ts geometry to start from
        calc (ase.calculator): ASE calculator to use for optimization
        max_cycles (int): maximum number of irc steps
        convergence (float): convergence criteria for forces
        verbose (bool): whether to print optimization output

    Returns:
        first (ase.Atoms): first geometry along the IRC
        last (ase.Atoms): last geometry along the IRC
    """
    if isinstance(calc, AIMNet2ASECalculator):
        calc.do_reset()
    atoms.calc = calc
    # Create an IRC object
    if verbose:
        irc = IRC(atoms, trajectory="temp.traj", logfile="-")
    else:
        irc = IRC(atoms, trajectory="temp.traj", logfile=None)

    # Run the IRC calculation
    irc.run(direction="forward", steps=max_cycles, fmax=convergence)
    irc.run(direction="reverse", steps=max_cycles, fmax=convergence)

    if irc.converged is False:
        print("IRC calculation did not converge")
        return None, None

    # Convert the trajectory to XYZ format
    reader = Trajectory("temp.traj")
    switch_index = None
    first_image = reader[0]
    first_image.calc = calc
    prev_energy = first_image.get_potential_energy()
    for i in range(len(reader)):
        image = reader[i]
        image.calc = calc
        energy = image.get_potential_energy()
        if switch_index is None and energy > prev_energy + 0.01:
            switch_index = i - 1
            break
        prev_energy = energy
    if switch_index is None:
        return None, None
    os.remove("temp.traj")
    return reader[switch_index], reader[-1]


def get_energy(mol, cid, calc):
    """
    Calculate the energy of a molecule using ASE.

    Args:
        mol (rdkit.Chem.Mol): RDKit molecule object.
        cid (int): Conformer ID.
        calc (ase.calculators): ASE calculator to use for energy calculation.

    Returns:
        energy (float): Energy of the molecule.
    """
    Chem.MolToXYZFile(mol, "temp.xyz", cid)
    ase_atoms = read("temp.xyz")
    # Set up the calculator
    ase_atoms.calc = calc
    energy = ase_atoms.get_potential_energy()
    energy = energy * EV_TO_KCAL
    os.remove("temp.xyz")
    return energy


def constrained_optimization(atoms, constraints, calc, fmax=0.05, max_cycles=1000):
    """
    Perform constrained optimization using ASE.

    Args:
        atoms (ase.Atoms): Molecule to optimize.
        constraints (list): List of constraints to apply.
        calc (ase.calculators): ASE calculator to use for energy and forces.

    Returns:
        atoms (ase.Atoms): Optimized ASE atoms object.
    """
    if isinstance(calc, AIMNet2ASECalculator):
        calc.do_reset()
    atoms.calc = calc
    if len(constraints) > 0:
        atoms.set_constraint(constraints)
    # Perform optimization
    with open(os.devnull, "w", encoding="utf-8") as f, contextlib.redirect_stdout(f):
        optimizer = BFGS(atoms, trajectory="opt.traj")
        optimizer.run(fmax=fmax, steps=max_cycles)

    return Trajectory("opt.traj")


def constrained_optimization_geom(atoms, constraints, calc, fmax=0.05, max_cycles=1000):
    """
    Perform constrained optimization using ASE.

    Args:
        atoms (ase.Atoms): Molecule to optimize.
        constraints (list): List of constraints to apply.
        calc (ase.calculators): ASE calculator to use for energy and forces.
        fmax (float): Maximum force convergence criteria.
        max_cycles (int): Maximum number of optimization cycles.

    Returns:
        atoms (ase.Atoms): Optimized ASE atoms object.
    """
    atoms.calc = calc
    atoms.set_constraint(constraints)

    # Perform optimization
    with open(os.devnull, "w", encoding="utf-8") as f, contextlib.redirect_stdout(f):
        optimizer = BFGS(atoms)
        optimizer.run(fmax=fmax, steps=max_cycles)

    return atoms
