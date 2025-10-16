"""
PSO.py — Enjambre de Partículas (random-keys para permutaciones)
=================================================================
- Posición: vector continuo de claves en [0,1].
- Decodificación: ordeno ids por esas claves → obtengo la permutación.
- Dinámica: actualizo velocidades y posiciones (w, c_cog, c_soc).
- Evaluación: decoder+fitness (mismo núcleo).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional
import random

from core import Instance, decode_permutation_to_schedule, fitness


@dataclass
class PSOConfig:
    """
    Parámetros PSO:
      - swarm_size: tamaño del enjambre.
      - iterations: iteraciones totales.
      - w_inertia: peso inercial (0.6–0.9 típico).
      - c_cog, c_soc: componentes cognitiva y social (1.2–2.0 típico).
      - seed: reproducibilidad.
    """
    swarm_size: int = 40
    iterations: int = 150
    w_inertia: float = 0.7
    c_cog: float = 1.4
    c_soc: float = 1.4
    seed: Optional[int] = 99


def _keys_to_perm(keys: List[float], ids: List[int]) -> List[int]:
    "Ordena ids según sus claves (random-keys)."
    return [pid for _, pid in sorted(zip(keys, ids), key=lambda t: t[0])]


def run_pso(inst: Instance, cfg: PSOConfig, fitness_kwargs: Optional[Dict] = None):
    """
    Núcleo PSO random-keys:
    - Inicializo claves aleatorias por partícula y velocidades 0.
    - Por iteración: actualizo velocidad, posición, clip a [0,1], evalúo y actualizo pbest/gbest.
    - Devuelvo mejor permutación, calendario, fitness y traza de gbest.
    """
    rng = random.Random(cfg.seed)
    fitness_kwargs = fitness_kwargs or {}
    ids = [s.sid for s in inst.surgeries]
    n = len(ids)
    if n == 0:
        empty_sched = decode_permutation_to_schedule(inst, [])
        return [], empty_sched, fitness(inst, empty_sched, **fitness_kwargs), []

    swarm_keys = [[rng.random() for _ in range(n)] for _ in range(cfg.swarm_size)]
    velocities = [[0.0] * n for _ in range(cfg.swarm_size)]
    pbest_keys = [k[:] for k in swarm_keys]
    pbest_fit = []
    pbest_sched = []

    for k in pbest_keys:
        perm = _keys_to_perm(k, ids)
        sched = decode_permutation_to_schedule(inst, perm)
        pbest_sched.append(sched)
        pbest_fit.append(fitness(inst, sched, **fitness_kwargs))

    g = min(range(cfg.swarm_size), key=lambda i: pbest_fit[i])
    gbest_keys = pbest_keys[g][:]
    gbest_sched = pbest_sched[g]
    gbest_fit = pbest_fit[g]
    history = [gbest_fit]

    for _ in range(cfg.iterations):
        for i in range(cfg.swarm_size):
            # Actualizar velocidad y posición (claves)
            for d in range(n):
                r1, r2 = rng.random(), rng.random()
                velocities[i][d] = (
                    cfg.w_inertia * velocities[i][d]
                    + cfg.c_cog * r1 * (pbest_keys[i][d] - swarm_keys[i][d])
                    + cfg.c_soc * r2 * (gbest_keys[d] - swarm_keys[i][d])
                )
                swarm_keys[i][d] += velocities[i][d]
            # Mantener claves en [0,1] (estabiliza el método)
            swarm_keys[i] = [min(1.0, max(0.0, x)) for x in swarm_keys[i]]

            # Evaluar
            perm = _keys_to_perm(swarm_keys[i], ids)
            sched = decode_permutation_to_schedule(inst, perm)
            fit = fitness(inst, sched, **fitness_kwargs)

            # pbest
            if fit < pbest_fit[i]:
                pbest_fit[i] = fit
                pbest_keys[i] = swarm_keys[i][:]
                pbest_sched[i] = sched

            # gbest
            if fit < gbest_fit:
                gbest_fit = fit
                gbest_keys = swarm_keys[i][:]
                gbest_sched = sched

        history.append(gbest_fit)

    best_perm = _keys_to_perm(gbest_keys, ids)
    return best_perm, gbest_sched, gbest_fit, history
