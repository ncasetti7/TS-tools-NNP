import torch
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from fairchem.core.units.mlip_unit import load_predict_unit
import ase.calculators.calculator

class AIMNet2ASECalculator(ase.calculators.calculator.Calculator):
    """ ASE calculator for AIMNet2 model
    Arguments:
        model (:class:`torch.nn.Module`): AIMNet2 model
        charge (int or float): molecular charge.  Default: 0
    """

    implemented_properties = ['energy', 'forces', 'free_energy', 'charges']

    def __init__(self, model, charge=0):
        super().__init__()
        self.model = model
        self.charge = charge
        self.device = next(model.parameters()).device
        cutoff = max(v.item() for k, v in model.state_dict().items() if k.endswith('aev.rc_s'))
        self.cutoff = float(cutoff)
        self._t_numbers = None
        self._t_charge = None

    def do_reset(self):
        self._t_numbers = None
        self._t_charge = None
        self.charge = 0.0

    def set_charge(self, charge):
        self.charge = float(charge)

    def _make_input(self):
        coord = torch.as_tensor(self.atoms.positions).to(torch.float).to(self.device).unsqueeze(0)
        if self._t_numbers is None:
            self._t_numbers = torch.as_tensor(self.atoms.numbers).to(torch.long).to(self.device).unsqueeze(0)
            self._t_charge = torch.tensor([self.charge], dtype=torch.float, device=self.device)
        d = dict(coord=coord, numbers=self._t_numbers, charge=self._t_charge)
        return d

    def _eval_model(self, d, forces=True):
        prev = torch.is_grad_enabled()
        torch._C._set_grad_enabled(forces)
        if forces:
            d['coord'].requires_grad_(True)
        _out = self.model(d)
        ret = dict(energy=_out['energy'].item(), charges=_out['charges'].detach()[0].cpu().numpy())
        if forces:
            if 'forces' in _out:
                f = _out['forces'][0]
            else:
                f = - torch.autograd.grad(_out['energy'], d['coord'])[0][0]
            ret['forces'] = f.detach().cpu().numpy()
        torch._C._set_grad_enabled(prev)
        return ret

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=ase.calculators.calculator.all_changes):
        super().calculate(atoms, properties, system_changes)
        _in = self._make_input()
        do_forces = 'forces' in properties
        _out =  self._eval_model(_in, do_forces)

        self.results['energy'] = _out['energy']
        self.results['charges'] = _out['charges']
        if do_forces:
            self.results['forces'] = _out['forces']
    
    def get_hessian(self, atoms, properties=['energy'], system_changes=ase.calculators.calculator.all_changes):
        """ Calculate the Hessian matrix of the system.
        """
        super().calculate(atoms, properties, system_changes)
        _in = self._make_input()
        with torch.jit.optimized_execution(False):
            _in['coord'].requires_grad_(True)
            _out = self.model(_in)
            e = _out['energy']
            f = -_get_derivatives_not_none(_in['coord'], e, create_graph=True)
            h = - torch.stack([
            _get_derivatives_not_none(_in['coord'], _f, retain_graph=True)[0]
                    for _f in f.flatten().unbind()
                    ])
        hessian = h.flatten(-2, -1).to(torch.double).cpu().numpy()
        return hessian

class BatchCalculator():
    def __init__(self):
        pass

    def get_energies(self, coord, numbers, charges):
        pass

    def get_energies_forces(self, coord, numbers, charges):
        pass

    def get_energies_forces_hessians(self, coord, numbers, charges):
        pass

class AIMNET(torch.nn.Module, BatchCalculator):
    def __init__(self, model, device) -> None:
        super().__init__()
        self.model = model
        self.device = device

    def forward(self, coord, numbers, charges):
        out = self.model(dict(coord=coord, numbers=numbers, charge=charges))
        return out['energy']
    
    def get_energies_forces(self, coord, numbers, charges):
        in_dict = dict(coord=coord, numbers=numbers, charge=charges)
        in_dict['coord'].requires_grad_(True)
        with torch.jit.optimized_execution(False):
            out = self.model(in_dict)
            e = out['energy']
            f = - torch.autograd.grad([e.sum()], [in_dict['coord']])[0]
        print(f.shape)
        raise ValueError
        return e, f
    
    def get_energies_forces_hessians(self, coord, numbers, charges):
        in_dict = dict(coord=coord, numbers=numbers, charge=charges)
        in_dict['coord'].requires_grad_(True)
        with torch.jit.optimized_execution(False):
            out = self.model(in_dict)
            e = out['energy']
            g =  _get_derivatives_not_none(in_dict['coord'], e, create_graph=True)
            f = -g
            a = [
            _get_derivatives_not_none(in_dict['coord'], _f, retain_graph=True)
                    for _f in f.flatten(-2, -1).unbind(1)
                        ]
            h = - torch.stack(a, dim=1)
        return e, f, h

class Fairchem(BatchCalculator):
    def __init__(self, model_file) -> None:
        super().__init__()
        self.model = load_predict_unit(model_file)

    def unpad_coords_numbers(self, coords, numbers):
        unpadded_coords = [coord[coord.sum(-1)!=0] for coord in coords]
        unpadded_numbers = [num[:len(coord)] for num, coord in zip(numbers, unpadded_coords)]
        return unpadded_coords, unpadded_numbers

    def prep_input(self, coord, numbers, charges):
        batch_size = coord.shape[0]
        # Make a cell tensor of zeros with ones on the diagonal
        cell = torch.zeros((1, 3, 3), dtype=coord.dtype)
        cell[:, 0, 0] = 1.0
        cell[:, 1, 1] = 1.0
        cell[:, 2, 2] = 1.0
        # Convert padded coords back to list of tensors
        coord, numbers = self.unpad_coords_numbers(coord, numbers)

        # Convert charges to long tensor
        charges = charges.to(torch.long)

        # Make a list of atomic data objects for each geometry
        data_list = []
        
        for i in range(batch_size):
            natoms = coord[i].shape[0] * torch.ones((1,), dtype=torch.long)
            data = AtomicData(
                pos=coord[i],
                atomic_numbers=numbers[i],
                cell=cell,
                pbc=torch.zeros((1, 3), dtype=torch.bool),
                natoms=natoms,
                edge_index=torch.empty((2, 0), dtype=torch.long),
                cell_offsets=torch.empty((0, 3), dtype=torch.float),
                nedges=torch.tensor([0], dtype=torch.long),
                charge=charges[i],
                spin=torch.tensor([1], dtype=torch.long),
                batch=torch.zeros((coord[i].shape[0],), dtype=torch.long),
                fixed=torch.zeros(coord[i].shape[0], dtype=torch.long),
                tags=torch.zeros(coord[i].shape[0], dtype=torch.long),
                energy=None,
                forces=None,
                stress=None,
                sid="id_" + str(i),
                dataset="omol",
            )
            data_list.append(data)
        # Convert the list of atomic data objects to a batch
        data = atomicdata_list_to_batch(data_list)
        return data

    def get_energies(self, coord, numbers, charges):
        data = self.prep_input(coord, numbers, charges)
        out = self.model.predict(data)
        return out['energy']

    def get_energies_forces(self, coord, numbers, charges):
        data = self.prep_input(coord, numbers, charges)
        with torch.jit.optimized_execution(False):
            out = self.model.predict(data)
            e = out['energy']
            f = out['forces']
        f = self.repad_forces(f, coord)
        return e, f
    
    def repad_forces(self, forces, coord):
        repadded_forces = torch.zeros((len(coord), coord.shape[1], 3), device=forces.device)
        force_index = 0
        for i in range(len(coord)):
            for j in range(len(coord[i])):
                if coord[i, j].sum() == 0:
                    break
                repadded_forces[i, j, :] = forces[force_index, :]
                force_index += 1
        return repadded_forces


def _get_derivatives_not_none(x, y, retain_graph=None, create_graph=False):
    ret = torch.autograd.grad([y.sum()], [x], retain_graph=retain_graph, create_graph=create_graph)[0]
    assert ret is not None
    return ret
