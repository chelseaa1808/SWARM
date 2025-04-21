import numpy as np
from algorithms.swarm import Swarm, Particle
from algorithms import config


def combpso():
    sw = Swarm()
    rng = np.random.default_rng()
    
    # Initialize swarm
    sw._R = rng.random((1, config.particle_size))
    p = Particle(sw)
    p.b = np.ones(config.particle_size)
    p._nbf = config.particle_size
    sw._gbest_b = p.b
    sw._gbest_nbf = p._nbf
    sw._gbest_cost, sw._gbest_cp = config.fitness_function(p, sw)

    swarm_size = config.swarm_size
    i = 0
    n = 0

    while i < swarm_size:
        p = Particle(sw)
        p.update_pbest(sw)
        p.init_gbest(sw)
        if i < config.swarm_size:
            sw._P.append(p)
        if i >= config.swarm_size + n * 50 - 1 and not sw._gbest_x:
            swarm_size += 50
            n += 1
        i += 1

    # Iterative optimization
    for t in range(config.max_nbr_iter):
        progress_ratio = t / (config.max_nbr_iter * config.w_a)
        denom = 1 + progress_ratio**config.w_b

        sw._w = config.w_min + (config.w_max - config.w_min) / denom
        sw._c1 = config.c_min + (config.c_max - config.c_min) / denom
        sw._c2 = config.c_max - (config.c_max - config.c_min) / denom
        sw._R = rng.random((3, config.particle_size))

        for p in sw._P:
            p.update_particle(sw)
            p.update_pbest(sw)
            p.update_gbest(sw)

        if sw._gbest_idle_counter >= sw._gbest_idle_max_counter:
            sw._reinit_partial_swarm()
        sw._gbest_idle_counter += 1

    print('\n ### Final gbest ###\n')
    print(sw)
    if sw._local_search():
        print(sw._final_str())

    return sw
