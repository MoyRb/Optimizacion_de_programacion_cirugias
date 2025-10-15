""""
- Genoma: permutación de ids de cirugía (sin separadores).
- Evaluación: reutilizo `decode_permutation_to_schedule` y `fitness` del núcleo.
- Operadores:
    * Selección: torneo (k configurable).
    * Crossover: OX (Order Crossover), conserva bloques y orden relativo.
    * Mutación: swap + barajar segmento (para explorar localmente).
- Elitismo: conservo la mejor fracción de la población cada generación.

Justificación:
- OX funciona muy bien cuando el orden relativo de elementos importa (es nuestro caso).
- La combinación swap + shuffle da un buen balance entre exploración y explotación.
"""

# GA.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import random, itertools
from core import Instance, Schedule, decode_permutation_to_schedule, fitness

@dataclass
class GAConfig:
    pop_size: int = 60
    generations: int = 150
    elite_frac: float = 0.1
    cx_prob: float = 0.9
    mut_prob: float = 0.3
    tournament_k: int = 3
    seed: Optional[int] = 42

def init_population(inst: Instance, pop_size: int, seed: Optional[int]=None) -> List[List[int]]:
    rng = random.Random(seed)
    ids = [s.sid for s in inst.surgeries]
    prio_rank = {'H':0,'M':1,'L':2}
    pop = []
    pop.append(sorted(ids, key=lambda sid:(prio_rank[inst.surgeries[sid].priority], inst.surgeries[sid].duration)))
    pop.append(sorted(ids, key=lambda sid: inst.surgeries[sid].duration))
    pop.append(sorted(ids, key=lambda sid: inst.surgeries[sid].arrival))
    while len(pop) < pop_size:
        p = ids[:]; rng.shuffle(p); pop.append(p)
    return pop

def tournament_selection(pop, fits, k=3, rng=None):
    rng = rng or random
    k = max(2, min(k, len(pop)))
    idxs = rng.sample(range(len(pop)), k)
    return pop[min(idxs, key=lambda i: fits[i])][:]

def order_crossover_OX(p1: List[int], p2: List[int], rng=None):
    rng = rng or random
    n = len(p1)
    if n < 2: return p1[:], p2[:]
    a,b = sorted(rng.sample(range(n),2))
    def ox(pa, pb):
        child = [None]*n
        child[a:b+1] = pa[a:b+1]
        filled = set(child[a:b+1])
        pos = (b+1)%n
        for g in itertools.chain(pb[b+1:], pb[:b+1]):
            if g not in filled:
                child[pos]=g; pos=(pos+1)%n
        return [x for x in child if x is not None]
    return ox(p1,p2), ox(p2,p1)

def mutate(perm: List[int], rng=None, p_swap=0.7, p_shuffle=0.3):
    rng = rng or random
    n = len(perm)
    if n>=2 and rng.random()<p_swap:
        i,j = rng.sample(range(n),2); perm[i],perm[j]=perm[j],perm[i]
    if n>=3 and rng.random()<p_shuffle:
        a,b = sorted(rng.sample(range(n),2)); seg = perm[a:b+1]; rng.shuffle(seg); perm[a:b+1]=seg

def run_ga(inst: Instance, cfg: GAConfig, fitness_kwargs: Optional[Dict]=None):
    rng = random.Random(cfg.seed)
    fitness_kwargs = fitness_kwargs or {}
    pop = init_population(inst, cfg.pop_size, seed=cfg.seed)

    def evalp(p):
        sched = decode_permutation_to_schedule(inst, p)
        return fitness(inst, sched, **fitness_kwargs), sched

    fits_scheds = [evalp(p) for p in pop]
    fits = [fs[0] for fs in fits_scheds]
    best_i = min(range(len(pop)), key=lambda i: fits[i])
    best_perm = pop[best_i][:]
    best_sched = fits_scheds[best_i][1]
    best_fit = fits[best_i]
    history = [best_fit]
    elite = max(1, int(cfg.elite_frac*cfg.pop_size))

    for _ in range(cfg.generations):
        ranked = sorted(zip(pop, fits_scheds), key=lambda pf: pf[1][0])
        new_pop = [p[:] for p,_ in ranked[:elite]]
        while len(new_pop) < cfg.pop_size:
            p1 = tournament_selection(pop, fits, cfg.tournament_k, rng)
            p2 = tournament_selection(pop, fits, cfg.tournament_k, rng)
            c1,c2 = (order_crossover_OX(p1,p2,rng) if rng.random()<cfg.cx_prob else (p1[:],p2[:]))
            if rng.random()<cfg.mut_prob: mutate(c1,rng)
            if rng.random()<cfg.mut_prob: mutate(c2,rng)
            new_pop.append(c1); 
            if len(new_pop)<cfg.pop_size: new_pop.append(c2)
        pop = new_pop
        fits_scheds = [evalp(p) for p in pop]; fits = [fs[0] for fs in fits_scheds]
        i = min(range(len(pop)), key=lambda k: fits[k])
        if fits[i] < best_fit: best_fit, best_perm, best_sched = fits[i], pop[i][:], fits_scheds[i][1]
        history.append(best_fit)
    return best_perm, best_sched, best_fit, history
