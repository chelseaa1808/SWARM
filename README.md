# Combinatorial Particle Swarm Optimization (COMB-PSO)

Updated for health research data workflows (2025)

Originally designed for high-dimensional feature selection using swarm intelligence.

COMB-PSO is a modular and extensible implementation of a combinatorial Particle Swarm Optimization (PSO) algorithm, now adapted to support modern health research datasets including CSV, EHR, and genomics.

It is suitable for:

Biomedical and health informatics researchers

Feature selection in high-dimensional datasets

Experimentation with swarm intelligence in Python

Rapid prototyping of metaheuristic optimization workflows

**Key Features**

Modular Particle Swarm Optimization algorithm

Built-in objective functions for benchmarking

Data support for CSV / tabular formats (e.g., healthcare data)

Health-focused examples: EHR feature selection, gene expression analysis

Plotting tools for cost evolution and particle tracking

Hyperparameter tuning tools

Clean, extensible code for new objective functions or topologies

*Installation*

`pip install combpso`


Compatible with Python 3.7+

Consider using a virtual environment

**Running the Optimizer**

*Basic usage:*



`python main.py combpso monks single True`


*General syntax:*



`code` python main.py combpso <dataset> <mode> <plot_flag>


*Example for a health dataset:*



` python main.py combpso ehr_2022 binary True`


*Directory Overview*

<code>
.
├── algorithms/         # Core PSO algorithm variants and custom implementations
├── datats/             # Input datasets (CSV, health data, benchmark sets)
├── evaluation/         # Performance metrics, result analysis, and logs
├── examples/           # Example scripts and usage demos (e.g. EHR feature selection)
├── main.py             # CLI entry point for launching optimization runs
├── README.md           # Project documentation
└── setup.py            # Installation script
</code>


**For Developers & Researchers**

This project provides a highly-extensible API for integrating:

Custom objective functions

Domain-specific constraints (e.g., EHR imputation, sparsity)

Alternative PSO topologies (ring, star, dynamic)

*License*

MIT License – free to use, modify, and distribute.

*Acknowledgments*

Original design by hdhrif

Modernized for 2025 use cases 
