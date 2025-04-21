"""
Model configuration loader for COMB-PSO.
Loads dataset, initializes classifier, sets swarm and RDC parameters.
"""

import numpy as np
from os.path import join
from sklearn.ensemble import ExtraTreesClassifier

from algorithms import config
from algorithms.utilities import subset_accuracy
from datasets import load, filter


def init_model(dataset_name: str):
    print(f" Initializing model for dataset: {dataset_name}")

    # Main control parameters
    config.epochs = 3
    config.swarm_size = 5
    config.max_nbr_iter = 5
    config.feature_init = 50

    # Load dataset and metadata
    if dataset_name not in load._loads:
        raise ValueError(f"Dataset '{dataset_name}' not found in load._loads")

    config.X, config.y, config.genes, config.classes, config.folder = load._loads[dataset_name]()
    
    # Feature selection filtering
    config.filtered_genes = filter.ad_filter()
    config.X = config.X[:, config.filtered_genes]
    config.genes = config.genes[config.filtered_genes]
    config.particle_size = len(config.filtered_genes)

    # Fitness function & model
    config.fitness_function = subset_accuracy
    config.clf = ExtraTreesClassifier(n_estimators=20, random_state=0)

    # Load RDC (non-linear dependency) values
    rdc_path = join(config.folder, f"{dataset_name}_irdc.txt")
    try:
        with open(rdc_path, 'r') as fr:
            config.irdc = np.array(fr.read().splitlines()).astype(float)
            config.irdc = config.irdc[config.filtered_genes]
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing RDC file: {rdc_path}")

    # RDC correlation matrix
    config.crdc = np.zeros((config.particle_size, config.particle_size))

    # Stability (frequency tracking)
    config.freq = np.zeros(config.particle_size)

    print("Model initialized successfully.\n")
