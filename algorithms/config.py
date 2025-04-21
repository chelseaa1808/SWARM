"""
COMB-PSO Configuration 
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional


@dataclass
class SwarmParams:
    v_min: float = -3.0
    v_max: float = 1.0
    x_min: float = -3.0
    x_max: float = 1.0
    w_min: float = 0.4
    w_max: float = 1.2
    c_min: float = 1.7
    c_max: float = 2.1
    w_a: float = 0.5
    w_b: float = 4.0
    swarm_size: Optional[int] = None
    particle_size: Optional[int] = None
    max_nbr_iter: Optional[int] = None
    gbest_idle_counter: int = 0
    gbest_idle_max_counter: int = 5
    swarm_reinit_frac: float = 0.1


@dataclass
class FeatureSelectionParams:
    X: List[Any] = field(default_factory=list)
    y: List[Any] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    genes: List[str] = field(default_factory=list)
    folder: Optional[str] = None
    fitness_function: Optional[Callable] = None
    clf: Any = None
    alpha_1: float = 0.8
    alpha_2: float = 0.1
    alpha_3: float = 0.1
    cv: int = 5
    filtered_genes: List[str] = field(default_factory=list)


@dataclass
class RDCParams:
    rdc_f: Callable = np.sin
    rdc_k: int = 20
    rdc_s: float = 1/6.
    rdc_n: int = 1
    irdc: List[Any] = field(default_factory=list)
    crdc: List[Any] = field(default_factory=list)
    ls_threshold: int = 15


@dataclass
class SubsetStabilityParams:
    freq: List[int] = field(default_factory=list)


# Master config object
swarm_params = SwarmParams()
fs_params = FeatureSelectionParams()
rdc_params = RDCParams()
stability_params = SubsetStabilityParams()

# Epochs or experiment settings
epochs = 5
