# Data Diet

This repository accompanies the paper [Deep Learning on a Data Diet: Finding Important Examples Early in Training](TODO) and contains the basic code for replicating the training and computations in it.



## Setup

The main requirements are [jax](https://github.com/google/jax), [flax](https://github.com/google/flax) and [tensorflow_datasets](https://www.tensorflow.org/datasets).

The [`Dockerfile`](Dockerfile) sets up a container with a functioning environment for this project. It can be used as a template for constructing an environment by other means. If using this container, the working directory will point to this repository. For the rest of this Readme, we use `ROOT` to reference the path to this repository.

After pulling this repository, create directories to contain the data and experiment checkpoints etc.

```sh
mkdir data
mkdir exps
```

Store the datasets in `<ROOT>/data`



## Experiment Scripts

We provide samples scripts for training networks, and computing scores in [`scripts`](scripts). Our examples are for CIFAR-10 and ResNet18 but can be easily modified for the other datasets and networks in the paper.



### Training

To train one independent run of ResNet18 on CIFAR10 (the full dataset), from `<ROOT>` execute

```sh
python scripts/run_full_data.py <ROOT:str> <EXP_NAME:str> <RUN_NUM:int>
```

This will train a model and save checkpoints and meta-data in `<ROOT>/exps/<EXP_NAME>/run_<RUN_NUM>`. `RUN_NUM` is used to identify independent runs and generate a unique seed for each run. To calculate scores, we recommend at least 10 runs. Forgetting events are tracked.

All scripts contain default hyperparameters such as seeds, dataset, network, optimizer, checkpoint frequency, etc. Changing these will generate different variants of the training run.

To train on a random subset of size `SUBSET_SIZE`, and save to `<ROOT>/exps/<EXP_NAME>/size_<SUBSET_SIZE>/run_<RUN_NUM>`, execute

```sh
python scripts/run_random_subset.py <ROOT:str> <EXP_NAME:str> <SUBSET_SIZE:int> <RUN_NUM:int>
```

To train on a subset of size `SUBSET_SIZE` comprised of maximum scores, with scores stored in a 1D numpy array at path `SCORE_PATH`, and save to  `<ROOT>/exps/<EXP_NAME>/size_<SIZE>/run_<RUN_NUM>`, execute

```sh
python scripts/run_keep_max_scores.py <ROOT:str> <EXP_NAME:str> <SCORE_PATH:str> <SUBSET_SIZE:int> <RUN_NUM:int>
```

To train on a subset of size `SUBSET_SIZE` comprised of smallest scores after an offset given by `OFFSET`, with scores stored in a 1D numpy array at path `SCORE_PATH`, and save to  `<ROOT>/exps/<EXP_NAME>/size_<SUBSET_SIZE>.offset_<OFFSET>/run_<RUN_NUM>`, execute

```sh
python scripts/run_offset_subset.py <ROOT:str> <EXP_NAME:str> <SCORE_PATH:str> <SUBSET_SIZE:int> <OFFSET:int> <RUN_NUM:int>
```

For a variation of any of the above but with a fraction of randomized labels given by `RAND_LABEL_FRAC`, (specify a seed for the randomness) change the corresponding script by adding to the script

```python
args.random_label_fraction = RAND_LABEL_FRAC
args.random_label_seed = RAND_LABEL_SEED
```



**Variants.**  Currently, the models `resnet18_lowres` and `resnet50_lowres` and datasets `cifar10`, `cinic10`, `cifar100` are supported.



### Scores

To calculate either the EL2N or GraNd scores for a network saved in `<ROOT>/exps/<EXP_NAME>/run_<RUN_NUM>` and checkpoint at step `STEP`,

```sh
python scripts/get_run_score.py <ROOT:str> <EXP_NAME:str> <RUN_NUM:int> <STEP:int> <BATCH_SZ:int> <TYPE:str>
```

`TYPE` is either `l2_error` for EL2N scores or `grad_norm` for GraNd scores.`BATCH_SZ` can be adjusted to fit the computation in GPU memory.

To calculate the average EL2N, GraNd or forget scores over multiple runs in an experiment saved in `<ROOT>/exps/<EXP_NAME>` (we assume that the `RUN_NUMS` are 0, 1, 2, ..., `N_RUNS-1`)

```sh
python scripts/get_mean_score.py <ROOT:str> <EXP_NAME:str> <N_RUNS:int> <STEP:int> <TYPE:str>
```

`TYPE` can be `l2_error`, `grad_norm`, or `forget`. This will save the score in `<ROOT>/exps/<EXP_NAME>`. 
