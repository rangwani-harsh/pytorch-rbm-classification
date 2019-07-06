# Restricted Boltzmann Machines (RBMs) for classification in PyTorch




## Overview

This project implements Restricted Boltzmann Machines (RBMs) for classification using PyTorch for (see `classification_rbm.py`).
This is the open source implementation of the paper [Classification using Discriminative Restricted Boltzmann Machines](www.dmi.usherb.ca/~larocheh/publications/icml-2008-discriminative-rbm.pdf). In this implementation we only test on the MNIST dataset.

## Usage

``` 
usage: train.py [-h] [--lr LR] [--epochs EPOCHS] [--batch-size BATCH_SIZE]
                [--early-stop EARLY_STOP] [--visible-units VISIBLE_UNITS]
                [--hidden-units HIDDEN_UNITS] [--no-cuda] [--cd-k CD_K]
                [--type TYPE] [--sparsity-coeffectient SPARSITY_COEFFECTIENT]
                [--data-folder DATA_FOLDER]

classification_model text classificer

optional arguments:
  -h, --help            show this help message and exit
  --lr LR               initial learning rate [default: 0.001]
  --epochs EPOCHS       number of epochs for train [default: 256]
  --batch-size BATCH_SIZE
                        batch size for training [default: 64]
  --early-stop EARLY_STOP
                        iteration numbers to stop without performance
                        increasing
  --visible-units VISIBLE_UNITS
                        Number of dimensions in input
  --hidden-units HIDDEN_UNITS
                        Number of dimensions of the hidden representation
  --no-cuda             disable the gpu
  --cd-k CD_K           The K in the contrastive Divergence Algorithm
  --type TYPE           The type of training you want to start -
                        discriminative, hybrid and generative
  --sparsity-coeffectient SPARSITY_COEFFECTIENT
                        The amount that must be subtracted from bias after
                        every update
  --data-folder DATA_FOLDER
                        Folder in which the data needs to be stored
  --generative-factor GENERATIVE_FACTOR
```

The configuration used in the paper are present in the file **reproduce.sh**
which can be run by: 
```./reproduce.sh```

