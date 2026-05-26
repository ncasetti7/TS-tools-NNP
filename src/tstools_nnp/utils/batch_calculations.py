import torch
from pysisyphus.constants import ANG2BOHR, AU2EV

from tstools_nnp.utils.interfaces import BatchCalculator

EV2AU = 1 / AU2EV
BOHR2ANG = 1 / ANG2BOHR

CONV_THRESHS = {
    #              max_force, rms_force, max_step, rms_step
    "gau_loose": (2.5e-3, 1.7e-3, 1.0e-2, 6.7e-3),
    "gau": (4.5e-4, 3.0e-4, 1.8e-3, 1.2e-3),
    "gau_tight": (1.5e-5, 1.0e-5, 6.0e-5, 4.0e-5),
    "gau_vtight": (2.0e-6, 1.0e-6, 6.0e-6, 4.0e-6),
    "baker": (3.0e-4, 2.0e-4, 3.0e-4, 2.0e-4),
    "neb": (0.015, 0.01, 20, 20),
    # Dummy thresholds
    "never": (2.0e-6, 1.0e-6, 6.0e-6, 4.0e-6),
    # IRC thresholds delta_energy, rms_force, unused, unused
    "irc": (1.0e-6, 1.0e-3, 1.0e-3, 1.0e-3),
}

CONV_THRESHS_FORCES = {
    #              max_force, rms_force
    "gau_loose": (2.5e-3, 1.7e-3),
    "gau": (4.5e-4, 3.0e-4),
    "gau_tight": (1.5e-5, 1.0e-5),
    "gau_vtight": (2.0e-6, 1.0e-6),
    "baker": (3.0e-4, 2.0e-4),
    "neb": (0.015, 0.01),
    # Dummy thresholds
    "never": (2.0e-6, 1.0e-6),
    # IRC thresholds delta_energy, rms_force, unused, unused
    "irc": (1.0e-6, 1.0e-3),
}


class GeometryCalculation:
    def __init__(
        self, calc: BatchCalculator, conv_thresh: str = "gau_vtight", max_cycles: int = 300, hess_update: str = "bofill"
    ) -> None:
        self.calc = calc
        self.conv_thresh = conv_thresh
        self.max_cycles = max_cycles
        self.hess_update_func = hess_update

    def calc_energies(self, coord, numbers, charges) -> torch.Tensor:
        """
        Calculate energies for a batch of geometries

        Args:
        coord (torch.Tensor): coordinates of the geometries (B, N, 3)
        numbers (torch.Tensor): atomic numbers of the geometries (B, N)
        charges (torch.Tensor): charges of the geometries (B)

        Returns:
        torch.Tensor: energies
        """
        e = self.calc(coord, numbers, charges)
        return (e.detach() * EV2AU).to(torch.double)

    def calc_energies_forces(self, coord, numbers, charges) -> torch.Tensor:
        """
        Calculate energies and forces for a batch of geometries

        Args:
        coord (torch.Tensor): coordinates of the geometry (B, N, 3)
        numbers (torch.Tensor): atomic numbers of the geometry (B, N)
        charges (torch.Tensor): charges of the geometry (B)

        Returns:
        torch.Tensor: energies
        torch.Tensor: forces
        """
        e, f = self.calc.get_energies_forces(coord, numbers, charges)
        energies = (e.detach() * EV2AU).to(torch.double)
        forces = (f.detach().flatten(-2, -1) * (EV2AU / ANG2BOHR)).to(torch.double)
        return energies, forces

    def calc_energies_forces_hessians(self, coord, numbers, charges) -> torch.Tensor:
        """
        Calculate energies, forces and hessians for a batch of geometries

        Args:
        coord (torch.Tensor): coordinates of the geometry (B, N, 3)
        numbers (torch.Tensor): atomic numbers of the geometry (B, N)
        charges (torch.Tensor): charges of the geometry (B)

        Returns:
        torch.Tensor: energies
        torch.Tensor: forces
        torch.Tensor: hessians
        """
        e, f, h = self.calc.get_energies_forces_hessians(coord, numbers, charges)
        energies = (e.detach() * EV2AU).to(torch.double)
        forces = (f.detach().flatten(-2, -1) * (EV2AU / ANG2BOHR)).to(torch.double)
        hessians = (h.detach().flatten(-2, -1) * (EV2AU / ANG2BOHR / ANG2BOHR)).to(torch.double)

        return energies, forces, hessians

    def check_convergence(self, step, forces):
        """
        Check if the optimization has converged

        Args:
        step (torch.Tensor): step
        forces (torch.Tensor): forces

        Returns:
        torch.Tensor: mask of the geometries that have converged
        """
        rms_force = torch.sqrt((forces**2).mean((1)))
        max_force = forces.abs().max((1)).values
        conv_thresh = torch.tensor(CONV_THRESHS_FORCES[self.conv_thresh], device=forces.device)
        print(max_force, rms_force)
        conv = torch.stack([max_force, rms_force], dim=-1) < conv_thresh
        return conv.all(-1)


class ConstrainedCalculator(BatchCalculator):
    def __init__(self, calc, constraints):
        self.calc = calc
        self.constraints = constraints

    def get_energies(self, coord, numbers, charges):
        e = self.calc.get_energies(coord, numbers, charges)
        for constraint in self.constraints:
            e += constraint.get_energy(coord, numbers, charges)
        return e

    def get_energies_forces(self, coord, numbers, charges):
        e, f = self.calc.get_energies_forces(coord, numbers, charges)
        for constraint in self.constraints:
            e += constraint.get_energy(coord, numbers, charges)
            f += constraint.get_force(coord, numbers, charges)
        return e, f

    def get_energies_forces_hessians(self, coord, numbers, charges):
        e, f, h = self.calc.get_energies_forces_hessians(coord, numbers, charges)
        for constraint in self.constraints:
            e += constraint.get_energy(coord, numbers, charges)
            f += constraint.get_force(coord, numbers, charges)
        return e, f, h


class Constraint:
    def __init__(self):
        pass

    def get_energy(self, coord, numbers, charges):
        pass

    def get_force(self, coord, numbers, charges):
        pass


class HookeanConstraint(Constraint):
    def __init__(self, k, r0, b, a1, a2):
        super().__init__()
        self.k = k
        self.r0 = r0
        self.b = b
        self.a1 = a1
        self.a2 = a2

    def get_energy(self, coord, numbers, charges):
        d = coord[self.b, self.a1] - coord[self.b, self.a2]
        # Zero out d if it's less than r0
        d = torch.where(d.norm(dim=-1) < self.r0, torch.zeros_like(d), d)
        return 0.5 * self.k * (d * d).sum(-1).sum(-1)

    def get_force(self, coord, numbers, charges):
        f = torch.zeros_like(coord)
        d = coord[self.b, self.a1] - coord[self.b, self.a2]
        # Zero out d if it's less than r0
        d = torch.where(d.norm(dim=-1) < self.r0, torch.zeros_like(d), d)
        f[self.b, self.a1] = -self.k * d
        f[self.b, self.a2] = self.k * d
        return f


class FIRE:
    def __init__(
        self,
        calculation: GeometryCalculation,
        dt: torch.double = 0.1,
        dt_max: torch.double = 1,
        N_acc: torch.int = 2,
        f_inc: torch.double = 1.1,
        f_acc: torch.double = 0.99,
        f_dec: torch.double = 0.5,
        n_reset: torch.int = 0,
        a_start: torch.double = 0.1,
        max_step: torch.double = 0.4,
    ) -> None:
        self.calculation = calculation
        self.dt = dt
        self.dt_max = dt_max
        self.N_acc = N_acc
        self.f_inc = f_inc
        self.f_acc = f_acc
        self.f_dec = f_dec
        self.n_reset = n_reset
        self.a_start = a_start
        self.max_step = max_step
        self.energies = []
        self.forces = []

    def run(self, coord, numbers, charges, frags=None, collision_energy=None, dump_traj=False):
        """
        Run FIRE optimization

        Args:
        coord (torch.Tensor): coordinates of the geometry (B, N, 3)
        numbers (torch.Tensor): atomic numbers of the geometry (B, N)
        charges (torch.Tensor): charges of the geometry (B)
        dump_traj (bool): whether to return the optimization trajectory

        Returns:
        torch.Tensor: coordinates of the optimized geometry (B, N, 3)
        """
        # Set values for optimization
        self.batch_size = coord.size(0)
        self.n_reset = torch.zeros(coord.shape[0], dtype=torch.long, device=coord.device)
        self.dt = torch.full(coord.shape[:1], self.dt, device=coord.device, dtype=torch.double)
        self.a = torch.full(coord.shape[:1], self.a_start, dtype=torch.double, device=coord.device)
        self.v = torch.zeros_like(coord, device=coord.device, dtype=torch.double)
        traj = [coord]

        convergence_mask = torch.zeros(self.batch_size, dtype=torch.bool, device=coord.device)
        for cycle in range(self.calculation.max_cycles):
            e, f = self.calculation.calc_energies_forces(coord, numbers, charges)

            self.energies.append(e)
            self.forces.append(f)

            # Calculate the step
            step = self.calc_step(f)
            # Check for convergence
            new_convergence_mask = self.calculation.check_convergence(
                step.flatten(-2, -1), f.reshape(self.batch_size, -1)
            )
            # Merge the convergence masks
            convergence_mask = convergence_mask + new_convergence_mask
            if torch.all(convergence_mask):
                break

            # Update the coordinates
            coord = coord.detach().reshape(self.batch_size, -1) * ANG2BOHR
            coord += step.flatten(-2, -1) * ~convergence_mask[:, None]
            coord = coord.reshape(self.batch_size, -1, 3) * BOHR2ANG
            if dump_traj:
                traj.append(coord)
        if dump_traj:
            return traj
        return coord

    def calc_step(self, forces):
        """
        Run a single step of FIRE optimization

        Args:
        coord (torch.Tensor): coordinates of the geometry (B, N, 3)
        forces (torch.Tensor): forces on the geometry (B, N, 3)

        Returns:
        torch.Tensor: coordinates of the step (B, N, 3)
        """
        forces = forces.reshape(self.batch_size, -1, 3)
        vf = (forces * self.v).flatten(-2, -1).sum(-1)

        w_vf = vf > 0.0

        if w_vf.all():
            a = self.a.unsqueeze(-1).unsqueeze(-1)
            v = self.v
            f = forces
            self.v = (1.0 - a) * v + a * v.flatten(-2, -1).norm(p=2, dim=-1).unsqueeze(-1).unsqueeze(
                -1
            ) * f / f.flatten(-2, -1).norm(p=2, dim=-1).unsqueeze(-1).unsqueeze(-1)
            self.v = torch.squeeze(self.v, dim=(1, 2))
            w_N = self.n_reset > self.N_acc
            self.dt[w_N] = (self.dt[w_N] * self.f_inc).clamp(max=self.dt_max)
            self.n_reset += 1
        elif w_vf.any():
            a = self.a[w_vf].unsqueeze(-1).unsqueeze(-1)
            v = self.v[w_vf]
            f = forces[w_vf]
            self.v[w_vf] = (1.0 - a) * v + a * v.flatten(-2, -1).norm(p=2, dim=-1).unsqueeze(-1).unsqueeze(
                -1
            ) * f / f.flatten(-2, -1).norm(p=2, dim=-1).unsqueeze(-1).unsqueeze(-1)
            w_N = self.n_reset > self.N_acc
            w_vfN = w_vf & w_N
            self.dt[w_vfN] = (self.dt[w_vfN] * self.f_inc).clamp(max=self.dt_max)
            self.a[w_vfN] *= self.f_acc
            self.n_reset[w_vfN] += 1
        w_vf = ~w_vf

        if w_vf.all():
            self.v[:] = 0.0
            self.a[:] = torch.tensor(self.a_start, device=self.a.device, dtype=torch.double)
            self.dt[:] *= self.f_acc
            self.n_reset[:] = 0
        elif w_vf.any():
            self.v[w_vf] = torch.tensor(0.0, device=self.v.device, dtype=torch.double)
            self.a[w_vf] = torch.tensor(self.a_start, device=self.a.device, dtype=torch.double)
            self.dt[w_vf] *= self.f_acc
            self.n_reset[w_vf] = torch.tensor(0, device=self.v.device)

        dt = self.dt.unsqueeze(-1).unsqueeze(-1)
        self.v += dt * forces
        step = dt * self.v
        normdr = step.flatten(-2, -1).norm(p=2, dim=-1).unsqueeze(-1).unsqueeze(-1)
        step *= (self.max_step / normdr).clamp(max=1.0)
        return step
