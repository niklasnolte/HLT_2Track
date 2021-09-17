Introduction
============

This package is meant to reproduce the entire pipeline to preprocess and train various models on simulated data.
Snakemake will run most scripts located in `scripts` but they can each be individually run with any configuration (if the proper input files exist.):

## configuration
the snakefile is built to process a subset of the cartesian product of all settings in config.yml.  
It is a subset because some settings do not work together. These are excluded by the heuristics in `expand_with_rules`.
To restrict yourself to a specific config, comment out the others in config.yml and call snakemake.
The snakefile is built such that when the scripts that produce an output are changed, the output is redone by snakemake.
WARNING: This dependency tracking does not extend to these scripts' imports. Therefore you may sometimes have to explicitly
reschedule execution by deleting the outputs, or, in case you want to rerun everything, `snakemake -F`.

## Preprocessing
preprocessing is made once for standalone and LHCb data at the beginning of the snakemake process.

## Training
the scripts in scripts/train train the different architectures.  

## Evaluating
We evaluate on a grid for plotting and on the data for efficiency/rate calculation.  
they are found under scripts/eval.

## Plotting
plots are made throughout the snakemake processing and can be found in plots/.

## model files
we save the trained models under savepoints/. The network that is implemented in Allen is also exported in `scripts/eval/export_model.py`

## other
the results folder contains several efficiency/rate results and their versions in latex format.

