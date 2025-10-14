"""
Módulo: Algoritmo Genético (GA) para Programación de Cirugías
=============================================================

Documentación del **código** (sin planteamiento del problema):

Estructura del módulo
---------------------
- **Clases de datos**: `Surgery`, `Instance`, `PlacedCase`, `Schedule`.
- **Decodificador**: `decode_permutation_to_schedule` — convierte una permutación en un calendario factible multi‑día/multi‑quirófano.
- **Fitness**: `fitness` — calcula F a partir del calendario, con normalización y pesos configurables.
- **Operadores GA**: `init_population`, `tournament_selection`, `order_crossover_OX`, `mutate`.
- **Bucle GA**: `run_ga` con elitismo, torneo, OX y mutación.
- **Utilidades**: `make_example_instance`, `make_edge_instance_empty_days`, `print_schedule`.
- **Pruebas de humo**: bloque `__main__` ejecuta una demo y tests rápidos para verificar que el flujo funciona y no hay `KeyError`.

Cómo parametrizar (puntos de entrada)
-------------------------------------
- **Datos**: edita/rehace `make_example_instance()` o crea un objeto `Instance` propio.
- **Parámetros GA**: modifica `GAConfig` o la variable `cfg` en `__main__`.
- **Pesos/priordades del fitness**: pasa `fitness_kwargs` en la llamada a `run_ga(...)`.
- **Impresión**: usa `print_schedule(inst, sched, show_empty=...)`.

Notas de depuración
-------------------
- Se corrigió el acceso a claves de `Schedule` en `print_schedule` para usar `(room, day)` y `.get(...)` con valores por defecto.
- Se añadieron pruebas de humo para validar impresión con días/salas vacíos.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import random
import math
import itertools

# =============================
# Modelo de datos (clases y tipos)
# =============================

Priority = str  # "H"|"M"|"L"

@dataclass
class Surgery:
    sid: int              # id único [0..n-1]
    duration: int         # minutos
    priority: Priority    # 'H'/'M'/'L'
    team: str             # equipo requerido, p.ej. 'Cardio', 'Ortho', ...
    arrival: int = 0      # minutos desde el inicio del día (0 si está desde la mañana)

@dataclass
class Instance:
    Q: int                                 # quirófanos
    D: int                                 # días del horizonte
    H: int                                 # jornada (minutos)
    surgeries: List[Surgery]               # lista de cirugías
    team_avail: Dict[Tuple[str,int], int]  # (team, day) -> 0/1 disponibilidad

    def avail(self, team: str, d: int) -> int:
        return self.team_avail.get((team, d), 0)

# =============================
# Representación del calendario que devuelve el decodificador
# =============================

@dataclass
class PlacedCase:
    sid: int
    day: int     # 1..D
    room: int    # 1..Q
    start: int   # minuto desde inicio del día
    end: int     # start + duration

@dataclass
class Schedule:
    placed: List[PlacedCase]                   # una entrada por cirugía
    room_day_end: Dict[Tuple[int,int], int]    # (room,day) -> fin del último caso
    overtime: Dict[Tuple[int,int], int]        # (room,day) -> minutos sobre H
    idle: Dict[Tuple[int,int], int]            # (room,day) -> minutos ociosos dentro [0,H]
    violations: int                            # conteo de violaciones duras

# =============================
# Decodificador (list‑scheduling: inserción en la posición factible más temprana)
# =============================

def decode_permutation_to_schedule(inst: Instance, perm: List[int]) -> Schedule:
    """Convierte una permutación en un calendario multi‑día/multi‑quirófano.

    Estrategia:
    - Para cada (room, day) mantenemos una lista de casos (en orden temporal) y un cursor de fin.
    - Recorremos las cirugías en el orden de `perm` y elegimos el **(día, quirófano, inicio)** con **inicio más temprano** que cumpla:
      * disponibilidad del equipo ese día;
      * `start >= arrival`;
      * sin solaparse (en este decodificador, se agrega al final de la cola del quirófano/día).
    - Tras ubicar todo, calculamos `overtime` y `idle` por (room,day).

    Nota: si ningún día del horizonte tiene al equipo disponible, se coloca al final del **último día** y se suma una violación.
    """
    # Estructuras de control: inicializamos TODAS las claves (q,d)
    timeline: Dict[Tuple[int,int], List[PlacedCase]] = {(q,d): [] for q in range(1, inst.Q+1) for d in range(1, inst.D+1)}
    endtime: Dict[Tuple[int,int], int] = {(q,d): 0 for q in range(1, inst.Q+1) for d in range(1, inst.D+1)}

    placed: List[PlacedCase] = []
    violations = 0

    def next_available_day(team: str, d0: int) -> Optional[int]:
        for d in range(d0, inst.D+1):
            if inst.avail(team, d) == 1:
                return d
        return None

    for sid in perm:
        s = inst.surgeries[sid]
        best_tuple: Optional[Tuple[int,int,int]] = None  # (start, day, room)

        d_start = next_available_day(s.team, 1)
        d_candidates = [inst.D] if d_start is None else list(range(d_start, inst.D+1))
        if d_start is None:
            violations += 1

        for d in d_candidates:
            if inst.avail(s.team, d) != 1:
                continue
            for q in range(1, inst.Q+1):
                start_candidate = max(endtime[(q,d)], s.arrival)
                tup = (start_candidate, d, q)
                if best_tuple is None or tup < best_tuple:
                    best_tuple = tup
            if best_tuple is not None and best_tuple[0] == s.arrival == 0:
                break

        if best_tuple is None:
            # sin disponibilidad: coloca en último día, sala con menor cola
            d = inst.D
            q = min(range(1, inst.Q+1), key=lambda r: endtime[(r,d)])
            start = max(endtime[(q,d)], s.arrival)
        else:
            start, d, q = best_tuple

        end = start + s.duration
        case = PlacedCase(sid=sid, day=d, room=q, start=start, end=end)
        timeline[(q,d)].append(case)
        endtime[(q,d)] = end
        placed.append(case)

    # Métricas por (room,day)
    overtime: Dict[Tuple[int,int], int] = {}
    idle: Dict[Tuple[int,int], int] = {}
    for q in range(1, inst.Q+1):
        for d in range(1, inst.D+1):
            cases = timeline[(q,d)]
            last_end = endtime[(q,d)]
            ot = max(0, last_end - inst.H)
            worked = sum(min(c.end, inst.H) - min(c.start, inst.H) for c in cases if c.start < inst.H)
            idle_time = max(0, inst.H - worked)
            overtime[(q,d)] = ot
            idle[(q,d)] = idle_time

    return Schedule(placed=placed, room_day_end=endtime, overtime=overtime, idle=idle, violations=violations)

# =============================
# Función de fitness (menor es mejor)
# =============================

def fitness(inst: Instance, sched: Schedule,
            w_wait: float = 0.5, w_ot: float = 0.3, w_idle: float = 0.2,
            alpha_H: float = 3.0, alpha_M: float = 1.0, alpha_L: float = 0.5,
            penalty_per_violation: float = 0.25) -> float:
    """Calcula el fitness ponderado y normalizado.

    Componentes:
      - W: espera ponderada por prioridad = sum(α_prio * max(0, s_i - r_i))
      - O: total de horas extra
      - U: total de tiempo ocioso dentro de la jornada

    Cotas de normalización (simples):
      UB_W = |I| * H
      UB_O = |Q| * |D| * H
      UB_U = |Q| * |D| * H

    Se agrega una pequeña penalización por cada violación dura reportada por el decodificador.
    """
    start_by_sid = {c.sid: c.start for c in sched.placed}
    prio_weight = {'H': alpha_H, 'M': alpha_M, 'L': alpha_L}

    W = 0.0
    for s in inst.surgeries:
        w = max(0, start_by_sid.get(s.sid, 0) - s.arrival)
        W += prio_weight.get(s.priority, 1.0) * w

    O = sum(sched.overtime.values())
    U = sum(sched.idle.values())

    UB_W = len(inst.surgeries) * inst.H
    UB_O = inst.Q * inst.D * inst.H
    UB_U = inst.Q * inst.D * inst.H

    Wn = W / UB_W if UB_W > 0 else 0.0
    On = O / UB_O if UB_O > 0 else 0.0
    Un = U / UB_U if UB_U > 0 else 0.0

    base = w_wait * Wn + w_ot * On + w_idle * Un
    penalty = sched.violations * penalty_per_violation
    return base + penalty

# =============================
# Operadores del GA
# =============================

def init_population(inst: Instance, pop_size: int, seed: Optional[int] = None) -> List[List[int]]:
    """Crea la población inicial de permutaciones (mezcla de heurísticas + aleatorias)."""
    rng = random.Random(seed)
    ids = [s.sid for s in inst.surgeries]

    pop: List[List[int]] = []

    # Heurística 1: prioridad (H>M>L), luego duración
    prio_rank = {'H': 0, 'M': 1, 'L': 2}
    sorted1 = sorted(ids, key=lambda sid: (prio_rank[inst.surgeries[sid].priority], inst.surgeries[sid].duration))
    pop.append(sorted1)

    # Heurística 2: SPT (duración más corta primero)
    sorted2 = sorted(ids, key=lambda sid: inst.surgeries[sid].duration)
    pop.append(sorted2)

    # Heurística 3: menor arribo primero
    sorted3 = sorted(ids, key=lambda sid: inst.surgeries[sid].arrival)
    pop.append(sorted3)

    # Resto: aleatorias
    while len(pop) < pop_size:
        perm = ids[:]
        rng.shuffle(perm)
        pop.append(perm)

    return pop


def tournament_selection(pop: List[List[int]], fits: List[float], k: int = 3, rng: Optional[random.Random] = None) -> List[int]:
    rng = rng or random
    k = max(2, min(k, len(pop)))  # robustez
    idxs = rng.sample(range(len(pop)), k)
    best_idx = min(idxs, key=lambda i: fits[i])
    return pop[best_idx][:]


def order_crossover_OX(p1: List[int], p2: List[int], rng: Optional[random.Random] = None) -> Tuple[List[int], List[int]]:
    """Crossover de Orden (OX) para permutaciones (devuelve dos hijos)."""
    rng = rng or random
    n = len(p1)
    if n < 2:
        return p1[:], p2[:]
    a, b = sorted(rng.sample(range(n), 2))

    def ox(pa: List[int], pb: List[int]) -> List[int]:
        child: List[Optional[int]] = [None] * n
        child[a:b+1] = pa[a:b+1]
        filled = set(child[a:b+1])
        pos = (b + 1) % n
        for gene in itertools.chain(pb[b+1:], pb[:b+1]):
            if gene not in filled:
                child[pos] = gene
                pos = (pos + 1) % n
        # assert no None remains
        return [g for g in child if g is not None]

    return ox(p1, p2), ox(p2, p1)


def mutate(perm: List[int], rng: Optional[random.Random] = None, p_swap: float = 0.7, p_shuffle: float = 0.3) -> None:
    """Mutaciones simples: swap y barajar un segmento (in‑place)."""
    rng = rng or random
    n = len(perm)
    if n >= 2 and rng.random() < p_swap:
        i, j = rng.sample(range(n), 2)
        perm[i], perm[j] = perm[j], perm[i]
    if n >= 3 and rng.random() < p_shuffle:
        a, b = sorted(rng.sample(range(n), 2))
        seg = perm[a:b+1]
        rng.shuffle(seg)
        perm[a:b+1] = seg

# =============================
# Bucle principal del GA
# =============================

@dataclass
class GAConfig:
    """Parámetros del algoritmo genético.

    Atributos:
        pop_size: Tamaño de la población.
        generations: Número de generaciones a ejecutar.
        elite_frac: Fracción de élite que se copia tal cual a la siguiente generación.
        cx_prob: Probabilidad de aplicar crossover (OX).
        mut_prob: Probabilidad de aplicar mutación.
        tournament_k: Tamaño del torneo para selección.
        seed: Semilla para reproducibilidad.
    """
    pop_size: int = 60
    generations: int = 150
    elite_frac: float = 0.1
    cx_prob: float = 0.9
    mut_prob: float = 0.3
    tournament_k: int = 3
    seed: Optional[int] = 42

@dataclass
class GAResult:
    """Resultado del GA y trazas útiles para análisis.

    Atributos:
        best_perm: Mejor permutación encontrada.
        best_sched: Calendario correspondiente a `best_perm`.
        best_fitness: Valor de fitness de la mejor solución.
        history: Mejor fitness por generación (para graficar convergencia).
    """
    best_perm: List[int]
    best_sched: Schedule
    best_fitness: float
    history: List[float]  # mejor fitness por generación


def run_ga(inst: Instance, cfg: GAConfig,
           fitness_kwargs: Optional[Dict] = None) -> GAResult:
    rng = random.Random(cfg.seed)
    fitness_kwargs = fitness_kwargs or {}

    pop = init_population(inst, cfg.pop_size, seed=cfg.seed)

    def eval_perm(perm: List[int]) -> Tuple[float, Schedule]:
        sched = decode_permutation_to_schedule(inst, perm)
        fit = fitness(inst, sched, **fitness_kwargs)
        return fit, sched

    fits_scheds = [eval_perm(p) for p in pop]
    fits = [fs[0] for fs in fits_scheds]

    best_idx = min(range(len(pop)), key=lambda i: fits[i])
    best_perm = pop[best_idx][:]
    best_sched = fits_scheds[best_idx][1]
    best_fit = fits[best_idx]

    history = [best_fit]
    elite_count = max(1, int(cfg.elite_frac * cfg.pop_size))

    for _ in range(cfg.generations):
        # Elitismo
        ranked = sorted(zip(pop, fits_scheds), key=lambda pf: pf[1][0])
        new_pop = [p[:] for p, _ in ranked[:elite_count]]

        # Reproducción
        while len(new_pop) < cfg.pop_size:
            p1 = tournament_selection(pop, fits, cfg.tournament_k, rng)
            p2 = tournament_selection(pop, fits, cfg.tournament_k, rng)

            if rng.random() < cfg.cx_prob:
                c1, c2 = order_crossover_OX(p1, p2, rng)
            else:
                c1, c2 = p1[:], p2[:]

            if rng.random() < cfg.mut_prob:
                mutate(c1, rng)
            if rng.random() < cfg.mut_prob:
                mutate(c2, rng)

            new_pop.append(c1)
            if len(new_pop) < cfg.pop_size:
                new_pop.append(c2)

        pop = new_pop
        fits_scheds = [eval_perm(p) for p in pop]
        fits = [fs[0] for fs in fits_scheds]

        gen_best_idx = min(range(len(pop)), key=lambda i: fits[i])
        gen_best_fit = fits[gen_best_idx]
        if gen_best_fit < best_fit:
            best_fit = gen_best_fit
            best_perm = pop[gen_best_idx][:]
            best_sched = fits_scheds[gen_best_idx][1]
        history.append(best_fit)

    return GAResult(best_perm=best_perm, best_sched=best_sched, best_fitness=best_fit, history=history)

# =============================
# Utilidades para experimentar / imprimir resultados
# =============================

def make_example_instance() -> Instance:
    """Instancia pequeña de ejemplo (3 quirófanos, 1 día, 480 min, 12 cirugías)."""
    Q, D, H = 3, 1, 480
    teams = ['A','B','C']
    team_avail = {(t, 1): 1 for t in teams}  # todos disponibles en el día 1

    data = [
        (0, 90,  'H','A', 0), (1, 60,  'M','A', 0), (2, 120, 'L','B', 0), (3, 45,  'H','B',  30),
        (4, 180, 'M','A', 0), (5, 75,  'L','C', 0), (6, 150, 'H','C', 60), (7, 30,  'M','A', 0),
        (8, 60,  'L','B', 0), (9, 90,  'M','C', 0), (10, 50, 'H','A', 0), (11, 120,'M','B', 0),
    ]
    surgeries = [Surgery(*row) for row in data]

    return Instance(Q=Q, D=D, H=H, surgeries=surgeries, team_avail=team_avail)


def make_edge_instance_empty_days() -> Instance:
    """Instancia de borde: 2 quirófanos, 2 días, sin cirugías (prueba de impresión y claves)."""
    Q, D, H = 2, 2, 480
    teams = ['A']
    team_avail = {("A", 1): 1, ("A", 2): 1}
    surgeries: List[Surgery] = []
    return Instance(Q=Q, D=D, H=H, surgeries=surgeries, team_avail=team_avail)


def print_schedule(inst: Instance, sched: Schedule, show_empty: bool = True) -> None:
    """Imprime el calendario por día y quirófano con KPIs.

    ✅ Corrige el KeyError usando claves **(room,day) = (q,d)** y `.get(...)` con valores por defecto.
    - `show_empty=False` oculta quirófanos sin casos y sin actividad (útil para instancias grandes).
    """
    # Agrupar por (room,day) para consistencia con Schedule
    by_room_day: Dict[Tuple[int,int], List[PlacedCase]] = {}
    for c in sched.placed:
        by_room_day.setdefault((c.room, c.day), []).append(c)

    for d in range(1, inst.D+1):
        print(f"\n=== Day {d} ===")
        for q in range(1, inst.Q+1):
            key = (q, d)  # (room, day)
            cases = sorted(by_room_day.get(key, []), key=lambda c: c.start)
            end_ = sched.room_day_end.get(key, 0)
            ot_ = sched.overtime.get(key, 0)
            idle_ = sched.idle.get(key, inst.H)

            if not show_empty and (not cases) and end_ == 0 and ot_ == 0 and idle_ == inst.H:
                continue  # no imprime salas completamente vacías/inactivas

            print(f"  OR {q}: end={end_} ot={ot_} idle={idle_}")
            for c in cases:
                s = inst.surgeries[c.sid]
                print(f"    S{c.sid:02d} ({s.priority}, team {s.team}, dur {s.duration})  {c.start:>3d}-{c.end:>3d}")

    total_ot = sum(sched.overtime.get((q,d), 0) for q in range(1, inst.Q+1) for d in range(1, inst.D+1))
    total_idle = sum(sched.idle.get((q,d), 0) for q in range(1, inst.Q+1) for d in range(1, inst.D+1))
    print(f"\nTotal overtime: {total_ot}  |  Total idle: {total_idle}  |  Violations: {sched.violations}")


# =============================
# Demo y pruebas de humo (solo al ejecutar este archivo)
# =============================
if __name__ == "__main__":
    # --- Demo principal con instancia de ejemplo ---
    inst = make_example_instance()
    cfg = GAConfig(pop_size=60, generations=120, elite_frac=0.15, seed=7)

    result = run_ga(inst, cfg, fitness_kwargs=dict(
        w_wait=0.6, w_ot=0.25, w_idle=0.15,
        alpha_H=3.0, alpha_M=1.0, alpha_L=0.5,
        penalty_per_violation=0.25,
    ))

    print(f"Best fitness: {result.best_fitness:.4f}")
    print_schedule(inst, result.best_sched)

    # --- Pruebas de humo adicionales ---
    # 1) Instancia sin cirugías, 2 días, para verificar que no hay KeyError y que imprime por defecto
    edge_inst = make_edge_instance_empty_days()
    empty_sched = decode_permutation_to_schedule(edge_inst, perm=[])  # sin cirugías
    print("\n[Smoke Test] Instancia vacía (2 días, 2 quirófanos) — sin KeyError esperado")
    print_schedule(edge_inst, empty_sched)

    # 2) Impresión ocultando quirófanos vacíos
    print("\n[Smoke Test] Ocultando quirófanos vacíos (show_empty=False)")
    print_schedule(edge_inst, empty_sched, show_empty=False)

    # 3) Traza de convergencia (primeros y últimos valores)
    print("\nHistory (best per generation):")
    print([round(x,4) for x in result.history[:10]], "...", [round(x,4) for x in result.history[-10:]])
