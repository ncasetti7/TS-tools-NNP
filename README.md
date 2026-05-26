# TS-tools-NNP

An NNP-specific implementation of TS-Tools for transition state search using neural network potentials (AIMNet2 and FAIRChem OMol).

## Setting up the environment

This project uses [pixi](https://pixi.sh) for environment management. Install pixi first if you don't have it:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Then install the default environment (all runtime dependencies):

```bash
pixi install
```

The package itself is installed in editable mode automatically. For development work (includes ruff and pytest):

```bash
pixi install -e dev
```

> **Note:** `autode` and `rdkit` are installed from conda-forge. `torch`, `ase`, `sella`, `fairchem-core`, and `pysisyphus` are installed from PyPI. If your `aimnet` package is not on PyPI, adjust the `[pypi-dependencies]` entry in `pixi.toml` accordingly (e.g., point to a local path or GitHub URL).

## Running TS searches

You need three things: a file of atom-mapped reaction SMILES (see `data/` for an example), a target directory for results, and a path to your NNP model weights (AIMNet2 or FAIRChem OMol).

```bash
pixi run python run_scripts/run_ts_searcher.py \
    --input-file data/example.txt \
    --model-path /PATH/TO/MODEL \
    --target-dir example
```

### Configuration

Runtime parameters live in `run_scripts/default_arguments.yaml`:

| Parameter | Default | Description |
|---|---|---|
| `model_type` | `AIMNET` | Model to use: `AIMNET` or `OMol` |
| `reactive_complex_factor_list_intramolecular` | `[1.3, 1.2, 1.8, 0]` | RC stretch factors for intramolecular reactions |
| `reactive_complex_factor_list_intermolecular` | `[2.5, 1.8, 2.8, 1.3]` | RC stretch factors for intermolecular reactions |
| `attempts` | `3` | Attempts per reactive complex factor |
| `batch` | `false` | Use batch processing (recommended with a GPU) |
| `batch_size` | `10` | Max structures per batch |
| `num_workers` | `5` | CPU workers (batch=false only; 0 = all CPUs) |
| `use_gpu` | `true` | Use GPU if available |
| `calc_hess` | `true` | Calculate Hessians for TS opt + IRC (AIMNet only) |

To use a custom config file:

```bash
pixi run python run_scripts/run_ts_searcher.py \
    --defaults-file my_config.yaml \
    --input-file data/example.txt \
    --model-path /PATH/TO/MODEL \
    --target-dir example
```

### Parsing results

After a run, calculate forward and reverse barriers with:

```bash
pixi run python run_scripts/parse_results.py \
    --target-dir example \
    --model-path /PATH/TO/MODEL \
    --save-results
```

This writes `barriers.txt` and `reverse_barriers.txt` to the target directory.

## Development

### Linting and formatting

Ruff is configured in `pyproject.toml` (line length 120, rules E/F/W/I).

```bash
pixi run -e dev ruff check .          # lint
pixi run -e dev ruff check --fix .    # lint + auto-fix
pixi run -e dev ruff format .         # format
```

Or use the pixi tasks:

```bash
pixi run -e dev lint
pixi run -e dev lint-fix
pixi run -e dev format
```

### Tests

```bash
pixi run -e dev test           # run tests
pixi run -e dev test-cov       # run tests with coverage report
```

Tests cover cheminformatics utilities, multiprocessing helpers, and `PathGenerator` logic. Tests that require model weights or a GPU are skipped automatically if those dependencies are not available (`pytest.importorskip`).

### CI

GitHub Actions runs two jobs on every push and pull request to `main`:

- **Lint** — `ruff check` + `ruff format --check` (Python + ruff only, fast)
- **Test** — pixi `ci-test` environment (rdkit, ase, pysisyphus, pytest; no torch/aimnet/fairchem)
