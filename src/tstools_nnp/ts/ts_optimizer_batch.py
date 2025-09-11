import os
from tstools_nnp.path.path_generator_batch import PathGeneratorBatch
from tstools_nnp.utils import cheminformatics

class TSOptimizerBatch():
    def __init__(self, ts_optimizers, batch_calc, device):
        self.ts_optimizers = ts_optimizers
        self.batch_calc = batch_calc
        self.device = device

    def generate_ts_batch(self):
        os.chdir(self.ts_optimizers[0].results_directory)
        for ts_optimizer in self.ts_optimizers:
            os.makedirs(ts_optimizer.path_generator.rxn_id, exist_ok=True)
            os.chdir(ts_optimizer.path_generator.rxn_id)
            # Make directories to store final results
            os.makedirs("rp_geometries", exist_ok=True)
            os.makedirs("final_ts_guess", exist_ok=True)
            if ts_optimizer.save_paths:
                os.makedirs("path_dir", exist_ok=True)
            os.chdir(self.ts_optimizers[0].results_directory)

        success_list = []
        index_list = [0]*len(self.ts_optimizers)
        for i in range(len(self.ts_optimizers[0].reactive_complex_factors)):
            path_generators = [ts_optimizer.path_generator for ts_optimizer in self.ts_optimizers if ts_optimizer.path_generator.rxn_id not in success_list]
            [path_generator.set_reactive_complex_factor(ts_optimizer.reactive_complex_factors[i]) for path_generator, ts_optimizer in zip(path_generators, self.ts_optimizers)]
            [path_generator.reset_opt_state() for path_generator in path_generators]
            path_generator_batch = PathGeneratorBatch(path_generators, self.batch_calc, self.device)
            energies, _, paths = path_generator_batch.get_paths()

            for ts_optimizer in self.ts_optimizers:
                if ts_optimizer.path_generator.rxn_id in energies:
                    os.chdir(ts_optimizer.path_generator.rxn_id)
                    index = index_list[self.ts_optimizers.index(ts_optimizer)]
                    if ts_optimizer.save_paths:
                        cheminformatics.path_to_xyz_file(paths[ts_optimizer.path_generator.rxn_id], ts_optimizer.path_generator.atomic_symbols, f"path_dir/path_{ts_optimizer.path_generator.reactive_complex_factor}_{index}.xyz")
                    success = ts_optimizer.check_ts_guesses(energies[ts_optimizer.path_generator.rxn_id],
                                                paths[ts_optimizer.path_generator.rxn_id],
                                                ts_optimizer.path_generator.atomic_symbols,
                                                index)
                    index_list[self.ts_optimizers.index(ts_optimizer)] += 1
                    os.chdir(self.ts_optimizers[0].results_directory)
                    if success:
                        success_list.append(ts_optimizer.path_generator.rxn_id)
                    os.chdir(self.ts_optimizers[0].results_directory)

        return success_list