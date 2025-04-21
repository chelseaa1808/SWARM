import sys
import pickle
from datetime import datetime
from os.path import join
import numpy as np

from algorithms.combpso import combpso
from algorithms.swarm import Swarm
from algorithms import config


def run_combpso(dataset_name=None):
    """
    Runs the COMB-PSO algorithm and saves the best feature subset.
    """
    if dataset_name is None:
        print("Error: Please provide a dataset name as an argument.")
        sys.exit(1)
    
    if config.particle_size is None or config.folder is None:
        print("Error: 'particle_size' and 'folder' must be set in config before running.")
        sys.exit(1)

    print(f"Starting COMB-PSO runs for dataset: {dataset_name}\n")
    start_time = datetime.now()

    SW = []
    for i in range(config.epochs):
        print(f" Run {i + 1}/{config.epochs}")
        sw = combpso()
        SW.append(sw)

    # Combine all gbest solutions
    print("\nConsolidating global bests...\n")
    gbest_u = np.zeros(config.particle_size, dtype=int)
    for sw in SW:
        gbest_u = np.bitwise_or(gbest_u, sw._gbest_b)

    sw_u = Swarm()
    sw_u._gbest_b = gbest_u
    sw_u._gbest_nbf = gbest_u.sum()
    sw_u._local_search()
    SW.append(sw_u)

    print("Final selected features:")
    print(sw_u._final_str())

    print(f"\n Total time elapsed: {datetime.now() - start_time}")

    output_path = join(config.folder, f"{dataset_name}.pickle")
    selected_genes = config.genes[sw_u._gbest_b == 1]

    print(f"\n Saving selected genes to: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(selected_genes, f)


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else None
    run_combpso(dataset)
