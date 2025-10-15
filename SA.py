# SA.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional
import random, math
from core import Instance, decode_permutation_to_schedule, fitness

@dataclass
class SAConfig:
    t0: float = 0.1
    alpha: float = 0.95
    tmin: float = 1e-4
    iters_per_T: int = 200
    seed: Optional[int] = 123
    p_move: float = 0.35  # prob. de usar insert vs swap

def initial_solution(inst: Instance) -> List[int]:
    ids = [s.sid for s in inst.surgeries]
    if not ids: return []
    pr = {'H':0,'M':1,'L':2}
    cands = [
        sorted(ids, key=lambda sid:(pr[inst.surgeries[sid].priority], inst.surgeries[sid].duration)),
        sorted(ids, key=lambda sid: inst.surgeries[sid].duration),
        sorted(ids, key=lambda sid: inst.surgeries[sid].arrival),
    ]
    best, bestf = cands[0], float('inf')
    for p in cands:
        f = fitness(inst, decode_permutation_to_schedule(inst, p))
        if f < bestf: best, bestf = p, f
    return best[:]

def n_swap(p, rng): 
    if len(p)<2: return p[:]
    i,j = rng.sample(range(len(p)),2); q=p[:]; q[i],q[j]=q[j],q[i]; return q
def n_insert(p, rng): 
    if len(p)<2: return p[:]
    i,j = rng.sample(range(len(p)),2); q=p[:]; g=q.pop(i); q.insert(j,g); return q

def run_sa(inst: Instance, cfg: SAConfig, fitness_kwargs: Optional[Dict]=None, start_perm: Optional[List[int]]=None):
    rng = random.Random(cfg.seed)
    fitness_kwargs = fitness_kwargs or {}

    perm = start_perm[:] if start_perm is not None else initial_solution(inst)
    sched = decode_permutation_to_schedule(inst, perm)
    F = fitness(inst, sched, **fitness_kwargs)
    best_perm, best_sched, best_F = perm[:], sched, F
    history = [best_F]

    T = cfg.t0
    while T > cfg.tmin:
        for _ in range(cfg.iters_per_T):
            cand = n_insert(perm,rng) if rng.random()<cfg.p_move else n_swap(perm,rng)
            sched2 = decode_permutation_to_schedule(inst, cand); F2 = fitness(inst, sched2, **fitness_kwargs)
            dF = F2 - F
            if dF <= 0 or rng.random() < math.exp(-dF/max(T,1e-12)):
                perm, sched, F = cand, sched2, F2
                if F < best_F: best_perm, best_sched, best_F = perm[:], sched, F
        T *= cfg.alpha
        history.append(best_F)
    return best_perm, best_sched, best_F, history
