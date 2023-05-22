# GW_methods

## Folder structure of this repository

~~~markdown
|-- data  
|-- scripts
|-- src
|   -- utils
 -- .gitignore  
 -- pyproject.toml  
 -- README.md  
~~~

## Roles of the folders

**data**: Where you put "minimal" data for tutorials. Do not put a large size data.  
**scripts**: Where you put scripts for various types of GW alignment tutorials.  
**src**: Where you put "core" modules for this project such as GW alignment with hyperparameter optimization or GW barycenter alignment.  
**utils**: Where you put utility modules that are not specific to GW alignment but are useful for general purposes.  

## Core modules in src

### Basic GW alignment (now mainly Sasaki & Abe)

GW alignment with the hyperparameter optimization of epsilon and initial transportation plans

0. util function for selecting CPU and GPU, changing variable types  

1. optimization for epsilon  
  a. grid search  
  b. Optuna  (Bayes)
2. optimization for initial transportation plans  
  a. diagonal  
  b. uniform (default option for POT)  
  c. random matrix  
3. evaluation function for unsupervised alignment  

### GWD + procrustes

Procrustes alignment after GW alignment (this should be easy to implement. it may be included in Basic GW class?)

### GWD + Wasserstein barycenter alignment

## Tutorials in scripts

1. Barycenter alignment using simulation data and human color similarity judgement data
2. histogram alignment using ANN natural objects similarity data  

## Data for tutorials

1. Human color similarity judgement data (Tsuchiya Lab)  <- Oizumi  
2. ANN natural objects similarity data (Alexnet and VGG?) <- Sasaki san  
3. THINGS data (THINGS team)  <- Takeda san  
