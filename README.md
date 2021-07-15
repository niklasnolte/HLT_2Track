Introduction
============

This package is meant to reproduce the entire pipeline to preprocess and train various models on simulated data.
Snakemake will run most scripts located in `scripts` but they can each be individually run with any configuration (if the proper input files exist.):
## Preprocessing
## Training
Snakemake will run the scripts located in `scripts/train` to train various models specified the Configuration api. By default snakemake will evaluate all possbile configurations as defined by the cartesian product of `configs.yaml`.

Models that can be trained include: NN, NN with infnorm, Boosted Decision Trees, Linear/ Quadratic Discriminants, and Gaussian Naive Bayes. The entire pipeline can be extended to more models with some work.


## Evaluating
## Plotting 

