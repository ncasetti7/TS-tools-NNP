# TS-tools-NNP

This repository holds an NNP specific implementation of TS-Tools

### Setting up the environment

To set up the ts-tools conda environment:

```
conda create -n ts-tools-nnp python
conda activate ts-tools-nnp
pip install -r requirements.txt
conda install autode --channel conda-forge
```

To install the TS-tools package, activate the ts-tools environment and run the following command within the TS-tools directory:

```
pip install -e .
```

### Running TS searches

To run a TS search, you'll need 3 things: a data file with all-atom mapped reaction SMILES (see example in data), a target results directory to save results, and a file path to your NNP of choice (currently supported: AIMNet, Fairchem). With these three, running a search looks like this

```
python run_scripts/run_ts_searcher.py --input-file data/example.txt --model-path /PATH/TO/MODEL --target-dir example
```

There are are extra arguments that can be modulated (model type, gpu use, etc.). These arguments are stored in a config file (which defaults to the config file default_arguments.yaml in run_scripts). These arguments can be modified by modifiying default_arguments.yaml (not recommended) or by creation of a new config file. The arguments in the new config file can be used by including the --defaults-file argument