import numpy as np
from copy import copy
from heapq import heappop, heappush
from math import ceil
from operator import attrgetter

from algorithms.utilities import sigmoid, RDC
from algorithms import config


class Swarm:
    def __init__(self):
        self.cfg = config
        self.particles = []
        self.rand_matrix = []
        self._w = None
        self._c1 = None
        self._c2 = None

        self._gbest_idle_counter = self.cfg.gbest_idle_counter
        self._gbest_idle_max_counter = self.cfg.gbest_idle_max_counter
        self._swarm_reinit_frac = self.cfg.swarm_reinit_frac

        self._gbest_x = []
        self._gbest_v = []
        self._gbest_b = []
        self._gbest_nbf = 0
        self._gbest_cost = 1.0
        self._gbest_cp = 0.0

        self._f_calls = 0
        self._nbr_reinit = 0
        self.heap = []

        self.X = self.cfg.X
        self.y = self.cfg.y
        self.clf = self.cfg.clf
        self.cv = self.cfg.cv
        self.alpha_1 = self.cfg.alpha_1
        self.alpha_2 = self.cfg.alpha_2
        self.alpha_3 = self.cfg.alpha_3
        self.irdc = self.cfg.irdc
        self.freq = self.cfg.freq

        self._init_cost = 1.0
        self._init_cp = 0.0

    def _reinit_partial_swarm(self):
        k = ceil(self.cfg.swarm_size * self._swarm_reinit_frac)
        selection_mask = [1]*k + [0]*(self.cfg.swarm_size - k)
        np.random.shuffle(selection_mask)

        for p in np.array(self.particles)[np.array(selection_mask) == 1]:
            p.v = np.random.uniform(self.cfg.v_min, self.cfg.v_max, size=self.cfg.particle_size)

        self._gbest_idle_counter = 0
        self._nbr_reinit += 1

    def _local_search(self):
        if self._gbest_nbf < self.cfg.ls_threshold:
            p = Particle(self)
            p.b = np.ones(self.cfg.particle_size)

            for i in range(self._gbest_nbf):
                for j in range(self._gbest_nbf):
                    if i != j:
                        if self.cfg.crdc[i, j] == 0:
                            self.cfg.crdc[i, j] = RDC(
                                self.cfg.X[:, [i, j]], self.cfg.y,
                                k=self.cfg.rdc_k, n=self.cfg.rdc_n
                            )
                        if self.irdc[i] >= self.irdc[j] and self.irdc[i] >= max(self.cfg.crdc[i, j], self.cfg.crdc[j, i]):
                            p.b[j] = 0
                            assert p.b.shape == self._gbest_b.shape
                            p._cost, p._cp = self.cfg.fitness_function(p, self)
                            if p._cp >= self._gbest_cp:
                                self._gbest_b[j] = 0
                                self._gbest_cost = p._cost
                                self._gbest_cp = p._cp
            return True
        return False

    def __len__(self):
        return self.cfg.swarm_size

    def __str__(self):
        selected_genes = {
            self.cfg.genes[i]: [self.irdc[i], self.freq[i]]
            for i, val in enumerate(self._gbest_b) if val == 1
        }
        s = f"After {self._f_calls} evaluations and {self._nbr_reinit} partial reinits\n"
        s += f"Best subset has {self._gbest_nbf} features with score {self._gbest_cp:.2f}\n"
        if self._gbest_nbf < 20:
            s += f"Selected features:\n{selected_genes}\n"
        return s

    def _final_str(self):
        selected_genes = {
            self.cfg.genes[i]: [self.irdc[i], self.freq[i]]
            for i, val in enumerate(self._gbest_b) if val == 1
        }
        s = f"Final subset: {self._gbest_nbf} features with score {self._gbest_cp:.2f}\n"
        if self._gbest_nbf < 20:
            s += f"Selected features:\n{selected_genes}\n"
        return s


class Particle:
    def __init__(self, swarm):
        cfg = swarm.cfg

        self.v = np.random.uniform(cfg.v_min, cfg.v_max, size=cfg.particle_size)
        initial_bits = np.array([0] * (cfg.particle_size - cfg.feature_init) + [1] * cfg.feature_init)
        np.random.shuffle(initial_bits)

        self.x = np.where(initial_bits == 1, cfg.x_max, cfg.x_min)
        self.b = (swarm.rand_matrix[0] < sigmoid(self.x)).astype(int)
        self._nbf = self.b.sum()

        self._cost = None
        self._cp = None

        self._pbest_x = self.x
        self._pbest_cost = 1.0
        self._pbest_cp = None

    def update_particle(self, swarm):
        cfg = swarm.cfg

        self.v = (
            swarm._w * self.v
            + swarm._c1 * np.multiply(swarm.rand_matrix[1], self._pbest_x - self.x)
            + swarm._c2 * np.multiply(swarm.rand_matrix[2], swarm._gbest_x - self.x)
        )
        self.v = np.clip(self.v, cfg.v_min, cfg.v_max)
        self.x = np.clip(self.x + self.v, cfg.x_min, cfg.x_max)
        self.b = (swarm.rand_matrix[0] < sigmoid(self.x)).astype(int)
        self._nbf = self.b.sum()

    def update_pbest(self, swarm):
        if self._nbf > 0:
            self._cost, self._cp = swarm.cfg.fitness_function(self, swarm)
            swarm._f_calls += 1

            if self._cost < self._pbest_cost:
                if self._pbest_cost != 1.0:
                    past = Particle(swarm)
                    past.x = copy(self._pbest_x)
                    past._cost = self._pbest_cost
                    past._cp = self._pbest_cp
                    heappush(swarm.heap, past)

                self._pbest_x = self.x
                self._pbest_cost = self._cost
                self._pbest_cp = self._cp

            elif swarm.heap:
                best_old = heappop(swarm.heap)
                while swarm.heap and best_old._cost > self._cost:
                    best_old = heappop(swarm.heap)
                if best_old._cost < self._cost:
                    self._pbest_x = copy(best_old.x)
                    self._pbest_cost = best_old._cost
                    self._pbest_cp = best_old._cp

    def init_gbest(self, swarm):
        if self._cost < swarm._gbest_cost:
            swarm._gbest_x = self.x
            swarm._gbest_v = self.v
            swarm._gbest_cost = self._cost
            swarm._gbest_cp = self._cp
            swarm._gbest_b = self.b
            swarm._gbest_nbf = self._nbf

    def update_gbest(self, swarm):
        if self._cost < swarm._gbest_cost:
            if swarm._gbest_cost < 1.0:
                old = Particle(swarm)
                old.x = copy(swarm._gbest_x)
                old._cost = swarm._gbest_cost
                old._cp = swarm._gbest_cp
                heappush(swarm.heap, old)

            swarm._gbest_x = self.x
            swarm._gbest_v = self.v
            swarm._gbest_cost = self._cost
            swarm._gbest_cp = self._cp
            swarm._gbest_b = self.b
            swarm._gbest_nbf = self._nbf
            swarm._gbest_idle_counter = 0

            for i in range(swarm.cfg.particle_size):
                if self.b[i]:
                    swarm.cfg.freq[i] += 1

            print(f"[GBEST] Updated: {self._nbf} features, Score = {self._cp:.4f}")

    def __lt__(self, other):
        return self._cost < other._cost

    def __le__(self, other):
        return self._cost <= other._cost

    def __len__(self):
        return len(self.x)

    def __str__(self):
        return f"Position: {self.x.tolist()} | Cost: {self._cp}"
