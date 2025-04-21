from os.path import dirname, join
import numpy as np
from functools import reduce
from typing import List


def _load_filtered_indices(folder_name: str, threshold: float = 0.25) -> np.ndarray:
    path = join(dirname(__file__), folder_name, f"{folder_name}_irdc.txt")
    with open(path, 'r') as fr:
        values = fr.read().splitlines()
        ranked = sorted(enumerate(values), key=lambda x: float(x[1]), reverse=True)
        filtered = [i for i, v in ranked if float(v) > threshold]
    return np.array(filtered)


def ad_filter() -> List[int]:
    """
    Returns list of feature indices that pass RDC threshold (0.25) across all AD datasets.
    """
    sets = [
        _load_filtered_indices('ad_blood1'),
        _load_filtered_indices('ad_blood2'),
        _load_filtered_indices('ad_kronos'),
        _load_filtered_indices('ad_rush')
    ]
    return reduce(np.intersect1d, sets).astype(int).tolist()
