# Toolbox for Gromov-Wasserstein Optimal Transport (GWOT)
This toolbox implements an easy-to-use hyperparameter optimization of GWOT.  
To find good local minima in GWOT, hyperparameter tuning is important.  
For hyperparameter tuning, [Optuna](https://optuna.org/) is used.  
The GWOT optimization is based on the Python Optimal Transport toolbox ([POT])(https://pythonot.github.io/).  

## Tutorials
To use our toolbox, please run `tutorial.ipynb` in the `scripts` folder.  
You can learn how to use this toolbox on two types of sample data.  
By replacing the tutorial data with your data, you will be able to easily test GWOT for your data!  

## Requirements
Please see `pyproject.toml` for required libraries.  
If you are using poetry, please run `poetry install` first and then, install `torch`, `torchvision`, `torchaudio` that are compatible with your environment.  
For compatibility information, see the official pytorch page. https://pytorch.org/get-started/locally/.   

## Folders in this repository  

`data`: Dataset for tutorials.  
`scripts`: Scripts for the GWOT optimization tutorials.  
`src`: Core modules for GWOT hyperparameter optimization.  
`utils`: Utility modules that are not specific for GWOT optimization but are useful for general purposes.  

## Core modules in src  

`gw_alignment.py`  
`align_representations.py`  

## Data for tutorials

1. Human color similarity judgment data (from Tsuchiya Lab)  
2. THINGS data (https://things-initiative.org/)  
