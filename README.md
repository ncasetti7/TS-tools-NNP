# TS-tools-NNP

An NNP-specific implementation of TS-Tools for transition state search using neural network potentials (AIMNet2 and FAIRChem OMol).

Given an atom-mapped reaction SMILES, the code generates a reactive path, identifies transition state guesses as maxima along that path, optimizes each guess to a saddle point with Sella, and validates it with an IRC that must connect the intended reactant and product.

---

## 1. System requirements

### Operating systems

| Platform | Status |
|---|---|
| Linux x86-64 | Supported; the only platform declared in `pixi.toml` |
| Windows via WSL2 | Tested (Ubuntu 24.04 on kernel 6.18 WSL2) |
| macOS / native Windows | Not supported — `pixi.toml` declares `platforms = ["linux-64"]` only |

### Software dependencies

Everything is pinned in `pixi.lock`, so `pixi install` reproduces the exact
environment. Versions below are those the code has been tested on:

| Package | Version tested | Source |
|---|---|---|
| Python | 3.11.15 | conda-forge |
| numpy | 2.4.6 | conda-forge |
| scipy | 1.17.1 | conda-forge |
| rdkit | 2026.03.2 | conda-forge |
| autode | 1.4.5 | conda-forge |
| torch | 2.8.0+cu128 | PyPI |
| ase | 3.28.0 | PyPI |
| sella | 2.4.2 | PyPI |
| pysisyphus | 1.0.0 | PyPI |
| fairchem-core | 2.20.0 | PyPI |
| aimnet | 0.2.0 | PyPI |

No other non-standard software is required. Model weights are **not** bundled —
you supply an AIMNet2 (`.pt` / `.jpt`) or FAIRChem OMol checkpoint via
`--model-path`.

> **Note on sella.** Every released sella (through 2.5.0) returns a zero-step IRC
> under ase >= 3.28: `Optimizer.irun` checks `gradient_converged()` while sella's
> `IRC` only overrides `converged()`, so an IRC started from a converged TS
> reports convergence before applying its imaginary-mode displacement
> ([zadorlab/sella#82](https://github.com/zadorlab/sella/issues/82)). This
> repository ships `tstools_nnp.utils.sella_compat.IRC`, a subclass applying the
> upstream fix, and uses it in `calculations.calc_irc`. No user action is needed;
> the shim can be dropped once a sella release carries the fix.

### Hardware

Runs on CPU only — no non-standard hardware is required. A CUDA GPU is optional
and enabled by default (`use_gpu: true`); selecting a GPU forces batch mode on.

| Resource | Requirement |
|---|---|
| CPU | Any x86-64. Reference machine: 12th Gen Intel Core i7-1260P, 8 cores / 16 threads |
| RAM | 16 GB is comfortable for the ~60-atom systems used here |
| Disk | ~8.6 GB for the `default` pixi environment (~17 GB if `dev` and `ci-test` are also built) |
| GPU | Optional; CUDA only. All timings in this README are CPU-only |

---

## 2. Installation guide

### Instructions

This project uses [pixi](https://pixi.sh) for environment management. Install pixi first if you don't have it:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Then clone the repository and install the default environment (all runtime dependencies):

```bash
git clone https://github.com/ncasetti7/TS-tools-NNP.git
cd TS-tools-NNP
pixi install
```

The package itself is installed in editable mode automatically. For development work (includes ruff and pytest):

```bash
pixi install -e dev
```

Verify the install:

```bash
pixi run -e dev pytest      # expect: 42 passed
```

> **Note:** `autode` and `rdkit` are installed from conda-forge. `torch`, `ase`, `sella`, `fairchem-core`, and `pysisyphus` are installed from PyPI. If your `aimnet` package is not on PyPI, adjust the `[pypi-dependencies]` entry in `pixi.toml` accordingly (e.g., point to a local path or GitHub URL).

### Typical install time on a "normal" desktop computer

Install time is dominated by download, since the `default` environment pulls
torch, fairchem, and their CUDA dependencies (~8.6 GB unpacked).

| Scenario | Time |
|---|---|
| First install, cold cache, ~100 Mbit connection | **10–25 min** (estimate; download-bound) |
| Repeat install, pixi package cache already populated | **4.2 s** (measured, fresh clone) |
| `ci-test` environment only, cold cache, 2-core CI runner | **40 s** (measured on GitHub Actions) |

The 10–25 min figure is an extrapolation from download volume, not a stopwatch
measurement; the two measured rows bracket it.

---

## 3. Demo

The repository ships a three-reaction demo input at `data/example.txt`: three
single-fragment intramolecular rearrangements of 37, 53, and 47 atoms, so all
three use the `reactive_complex_factor_list_intramolecular` factors.

### Instructions to run on the demo data

```bash
pixi run python run_scripts/run_ts_searcher.py \
    --input-file data/example.txt \
    --model-path /PATH/TO/MODEL \
    --target-dir example
```

Then compute barriers from the located transition states:

```bash
pixi run python run_scripts/parse_results.py \
    --target-dir example \
    --model-path /PATH/TO/MODEL \
    --save-results
```

### Expected output

One directory per reaction id, named after the first column of the input file:

```
example/
├── R1/
│   ├── final_ts_guess/
│   │   └── ts_guess_0.xyz        # validated transition state
│   ├── rp_geometries/
│   │   ├── reactant_0.xyz        # optimized IRC endpoint, reactant side
│   │   └── product_0.xyz         # optimized IRC endpoint, product side
│   ├── path_dir/
│   │   └── path_<factor>_<i>.xyz # reactive path, one frame per image (save_paths: true)
│   └── (reactant_check.xyz, product_check.xyz, opt.traj, tmp*.xyz — scratch)
├── R2/ ...
├── R3/ ...
├── batch_0/ ... batch_N/         # one scratch dir per CPU worker
├── barriers.txt                  # written by parse_results.py
└── reverse_barriers.txt
```

`final_ts_guess/` is populated **only** when a TS survives the IRC check, so an
empty `final_ts_guess/` means that reaction failed. `barriers.txt` holds one
`<rxn_id> <barrier_kcal_per_mol>` line per succeeded reaction, and
`reverse_barriers.txt` the same measured from the product side.

On stdout, each reaction reports its state as it advances
(`Initial Optimization` → `Stretched Optimization` → `Reactive Optimization` →
`Reaction Path Optimization` → `TS Optimization` → `IRC`), ending with:

```
Successful reactions: ['R1', 'R3']
Time taken: <seconds> seconds
```

The list names only the reactions that produced a validated TS, so it may be a
subset of the input — the search is not expected to succeed on every reaction,
and the demo's success set has not been characterized across repeat runs.

### Expected run time for the demo on a "normal" desktop computer

**Order of hours on CPU** for all three reactions (they run concurrently across
`num_workers` processes, so wall-clock is set by the slowest reaction, and each
reaction retries up to `attempts` times across four reactive-complex factors). Note that this is for a 53 atom molecule and scaling for MLIPs tends to be approximately linear.

---

## 4. Instructions for use

### How to run the software on your own data

1. **Prepare an input file.** One reaction per line, `<id> <atom-mapped reaction SMILES>`,
   whitespace-separated. Every atom — hydrogens included — must carry a map number,
   and reactant and product map numbers must correspond. Use `data/example.txt` as a
   template. Charge and multiplicity are inferred from the reactant SMILES
   (formal charge, and radical electron count + 1).

2. **Choose a model.** Point `--model-path` at your AIMNet2 checkpoint
   (`.pt` or `.jpt`), or set `model_type: OMol` in the config and pass a FAIRChem
   OMol checkpoint. Note that `calc_hess` is unavailable for OMol and is
   force-disabled with a warning.

3. **Run the search:**

   ```bash
   pixi run python run_scripts/run_ts_searcher.py \
       --input-file /path/to/my_reactions.txt \
       --model-path /PATH/TO/MODEL \
       --target-dir my_results
   ```

4. **Collect barriers:**

   ```bash
   pixi run python run_scripts/parse_results.py \
       --target-dir my_results \
       --model-path /PATH/TO/MODEL \
       --save-results
   ```

Reactions with an empty `final_ts_guess/` did not yield a validated TS;
`parse_results.py` lists these under "The following reactions were not finished".
Common remedies are adding reactive-complex factors, raising `attempts`, or
supplying a better-suited model.

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

Whether a reaction is treated as intra- or intermolecular is decided by the
presence of a `.` in the reaction SMILES.

To use a custom config file:

```bash
pixi run python run_scripts/run_ts_searcher.py \
    --defaults-file my_config.yaml \
    --input-file data/example.txt \
    --model-path /PATH/TO/MODEL \
    --target-dir example
```


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
pixi run -e dev pytest                  # run tests (42 tests, ~10 s)
pixi run -e dev pytest --cov=tstools_nnp # with coverage
```

Tests cover cheminformatics utilities, multiprocessing helpers, `PathGenerator`
logic, and the sella IRC compatibility shim. Tests requiring an optional
dependency skip automatically via `pytest.importorskip`.

> Use `pixi run -e dev pytest` rather than `pixi run test`: `pytest` belongs to the
> `dev` feature, but the `[tasks]` table binds to the default environment, so
> `pixi run test` falls through to whatever `pytest` is on `PATH`.

### CI

GitHub Actions runs two jobs on every push and pull request to `main`:

- **Lint** — `ruff check` + `ruff format --check` (Python + ruff only, fast)
- **Test** — pixi `ci-test` environment (rdkit, ase, pysisyphus, pytest; no torch/aimnet/fairchem)
