"""Compatibility shims for sella.

Every released sella (<= 2.5.0, 2026-06-23) predates the upstream fix for its
IRC taking zero steps under ase >= 3.28 (zadorlab/sella#82, fixed by #83, merged
2026-07-13). Drop this module and import IRC from sella directly once a release
containing that fix exists.
"""

from sella import IRC as _SellaIRC


class IRC(_SellaIRC):
    """sella's IRC with its first-step guard restored under ase >= 3.28.

    ase >= 3.28's ``Optimizer.irun`` checks ``gradient_converged()`` instead of
    ``converged()``, which bypasses ``IRC.converged()`` and the guards that live
    there (the first-step guard and the "lowest Hessian eigenvalue is positive"
    check). An IRC started from a converged TS, where ``|F| < fmax`` already
    holds, therefore reports convergence before taking its first step: the
    displacement along the imaginary mode is never applied, and forward and
    reverse both return the input TS unchanged. Routing ``gradient_converged()``
    through ``converged()`` restores the guard, exactly as upstream does.
    """

    def gradient_converged(self, gradient=None):
        return self.converged()
