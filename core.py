"""
core.py — Núcleo común (modelo, decodificador y fitness)
========================================================

Resumen (lo que implementé)
- Clases de datos: `Surgery`, `Instance`, `PlacedCase`, `Schedule`.
- Decodificador *list-scheduling* que transforma una permutación de cirugías en
  un calendario factible multi-día / multi-quirófano (sin solapes).
- Función de `fitness` que combina (espera ponderada por prioridad, horas extra,
  tiempo ocioso) con normalización y pesos configurables.
- Impresor robusto `print_schedule` para inspección y debugging (sin KeyError).

Decisiones de diseño (por qué así)
- Representación compacta: una **permutación** de ids; el decodificador decide día y sala.
  Esto permite reutilizar la misma representación en GA/SA/PSO y comparar de forma justa.
- Política de factibilidad: el decodificador **siempre** devuelve un calendario factible;
  si un equipo no está disponible en todo el horizonte, coloco la cirugía al final del
  último día y sumo una pequeña penalización en `violations`.
- Métricas separadas por (quirófano,día) para auditar: `overtime`, `idle`, `room_day_end`.

Complejidad (aprox.)
- Decodificar N cirugías en Q salas y D días: O(N·Q·D) en el peor caso (revisar candidatos).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


# -------------------------
# Clases de datos
# -------------------------

Priority = str  # 'H'|'M'|'L'


@dataclass
class Surgery:
    """
    Cirugía atómica en el modelo.

    Atributos:
        sid: identificador entero [0..N-1]
        duration: minutos estimados (>0)
        priority: 'H'/'M'/'L' (peso en la espera dentro del fitness)
        team: etiqueta del equipo requerido (p.ej., 'Cardio')
        arrival: minuto de arribo dentro del día (0 si el paciente está desde la mañana)
    """
    sid: int
    duration: int
    priority: Priority
    team: str
    arrival: int = 0


@dataclass
class Instance:
    """
    Instancia del problema.

    Atributos:
        Q: número de quirófanos
        D: número de días del horizonte
        H: jornada por día (minutos)
        surgeries: lista de `Surgery`
        team_avail: disponibilidad binaria por (equipo, día) → {0,1}
    """
    Q: int
    D: int
    H: int
    surgeries: List[Surgery]
    team_avail: Dict[Tuple[str, int], int]

    def avail(self, team: str, d: int) -> int:
        "Devuelve 1 si el equipo está disponible el día d; en otro caso 0."
        return self.team_avail.get((team, d), 0)


@dataclass
class PlacedCase:
    """
    Cirugía colocada en el calendario.

    Atributos:
        sid: id de la cirugía
        day: día asignado (1..D)
        room: quirófano asignado (1..Q)
        start: minuto de inicio dentro del día
        end: minuto de fin (start + duration)
    """
    sid: int
    day: int
    room: int
    start: int
    end: int


@dataclass
class Schedule:
    """
    Calendario resultado de decodificar una permutación.

    Atributos:
        placed: lista de casos colocados (uno por cirugía)
        room_day_end[(q,d)]: fin del último caso en la sala q día d
        overtime[(q,d)]: minutos por encima de la jornada H
        idle[(q,d)]: minutos ociosos dentro de la jornada H
        violations: conteo de violaciones duras (equipo nunca disponible en todo D)
    """
    placed: List[PlacedCase]
    room_day_end: Dict[Tuple[int, int], int]
    overtime: Dict[Tuple[int, int], int]
    idle: Dict[Tuple[int, int], int]
    violations: int


# -------------------------
# Decodificador
# -------------------------

def decode_permutation_to_schedule(inst: Instance, perm: List[int]) -> Schedule:
    """
    Decodificador list-scheduling (inserción al final):
    --------------------------------------------------
    Recorro la permutación y para cada cirugía busco el par (día, sala) con
    el **inicio factible más temprano** cumpliendo:
      - team disponible ese día,
      - start >= arrival,
      - sin solapes (añado al final de la cola del quirófano/día).

    Si ningún día tiene el equipo disponible, coloco la cirugía en la cola del
    último día (mantengo la solución medible) y aumento `violations`.

    Métricas por (quirófano, día):
      - overtime(q,d) = max(0, end_last - H)
      - idle(q,d)     = H - trabajo_dentro_de_[0,H]  (acotado a [0,H])
    """
    # Inicializar todas las claves (q,d) para evitar KeyError
    timeline: Dict[Tuple[int, int], List[PlacedCase]] = {
        (q, d): [] for q in range(1, inst.Q + 1) for d in range(1, inst.D + 1)
    }
    endtime: Dict[Tuple[int, int], int] = {
        (q, d): 0 for q in range(1, inst.Q + 1) for d in range(1, inst.D + 1)
    }

    placed: List[PlacedCase] = []
    violations = 0

    def next_available_day(team: str, d0: int) -> Optional[int]:
        for d in range(d0, inst.D + 1):
            if inst.avail(team, d) == 1:
                return d
        return None

    for sid in perm:
        s = inst.surgeries[sid]
        best_tuple: Optional[Tuple[int, int, int]] = None  # (start, day, room)

        d_start = next_available_day(s.team, 1)
        d_candidates = [inst.D] if d_start is None else list(range(d_start, inst.D + 1))
        if d_start is None:
            violations += 1

        for d in d_candidates:
            if inst.avail(s.team, d) != 1:
                continue
            for q in range(1, inst.Q + 1):
                start_candidate = max(endtime[(q, d)], s.arrival)
                tup = (start_candidate, d, q)
                if best_tuple is None or tup < best_tuple:
                    best_tuple = tup
            # early exit: si puedo empezar exactamente al arribo 0, es óptimo para este decoder
            if best_tuple is not None and best_tuple[0] == s.arrival == 0:
                break

        if best_tuple is None:
            # sin disponibilidad: coloca en último día, sala con menor cola
            d = inst.D
            q = min(range(1, inst.Q + 1), key=lambda r: endtime[(r, d)])
            start = max(endtime[(q, d)], s.arrival)
        else:
            start, d, q = best_tuple

        end = start + s.duration
        case = PlacedCase(sid=sid, day=d, room=q, start=start, end=end)
        timeline[(q, d)].append(case)
        endtime[(q, d)] = end
        placed.append(case)

    # Métricas por (room,day)
    overtime: Dict[Tuple[int, int], int] = {}
    idle: Dict[Tuple[int, int], int] = {}
    for q in range(1, inst.Q + 1):
        for d in range(1, inst.D + 1):
            cases = timeline[(q, d)]
            last_end = endtime[(q, d)]
            ot = max(0, last_end - inst.H)
            worked = sum(
                min(c.end, inst.H) - min(c.start, inst.H) for c in cases if c.start < inst.H
            )
            idle_time = max(0, inst.H - worked)
            overtime[(q, d)] = ot
            idle[(q, d)] = idle_time

    return Schedule(
        placed=placed,
        room_day_end=endtime,
        overtime=overtime,
        idle=idle,
        violations=violations,
    )


# -------------------------
# Fitness (menor es mejor)
# -------------------------

def fitness(
    inst: Instance,
    sched: Schedule,
    w_wait: float = 0.5,
    w_ot: float = 0.3,
    w_idle: float = 0.2,
    alpha_H: float = 3.0,
    alpha_M: float = 1.0,
    alpha_L: float = 0.5,
    penalty_per_violation: float = 0.25,
) -> float:
    """
    Fitness (menor es mejor):
    -------------------------
    F = w_wait * Wn + w_ot * On + w_idle * Un + penalizaciones

    Donde:
      - W = sum(alpha[prio_i] * max(0, start_i - arrival_i))
      - O = sum_{q,d} overtime(q,d)
      - U = sum_{q,d} idle(q,d)

    Normalización:
      UB_W = |I| * H, UB_O = |Q|*|D|*H, UB_U = |Q|*|D|*H
      -> Wn = W/UB_W, On = O/UB_O, Un = U/UB_U
    """
    start_by_sid = {c.sid: c.start for c in sched.placed}
    prio_weight = {"H": alpha_H, "M": alpha_M, "L": alpha_L}

    # Espera ponderada
    W = 0.0
    for s in inst.surgeries:
        w = max(0, start_by_sid.get(s.sid, 0) - s.arrival)
        W += prio_weight.get(s.priority, 1.0) * w

    # Horas extra e Idle
    O = sum(sched.overtime.values())
    U = sum(sched.idle.values())

    # Normalización simple
    UB_W = len(inst.surgeries) * inst.H
    UB_O = inst.Q * inst.D * inst.H
    UB_U = inst.Q * inst.D * inst.H

    Wn = W / UB_W if UB_W > 0 else 0.0
    On = O / UB_O if UB_O > 0 else 0.0
    Un = U / UB_U if UB_U > 0 else 0.0

    base = w_wait * Wn + w_ot * On + w_idle * Un
    penalty = sched.violations * penalty_per_violation
    return base + penalty


# -------------------------
# Impresor (debugging)
# -------------------------

def print_schedule(inst: Instance, sched: Schedule, show_empty: bool = True) -> None:
    """
    Impresor de calendario (robusto):
    - Usa claves (room, day) consistentemente para evitar KeyError.
    - `.get(..., default)` para salas/días sin actividad.
    - `show_empty=False` oculta quirófanos vacíos (útil en instancias grandes).
    """
    by_room_day: Dict[Tuple[int, int], List[PlacedCase]] = {}
    for c in sched.placed:
        by_room_day.setdefault((c.room, c.day), []).append(c)

    for d in range(1, inst.D + 1):
        print(f"\n=== Día {d} ===")
        for q in range(1, inst.Q + 1):
            key = (q, d)
            cases = sorted(by_room_day.get(key, []), key=lambda c: c.start)
            end_ = sched.room_day_end.get(key, 0)
            ot_ = sched.overtime.get(key, 0)
            idle_ = sched.idle.get(key, inst.H)

            if not show_empty and (not cases) and end_ == 0 and ot_ == 0 and idle_ == inst.H:
                continue

            print(f"  QF {q}: end={end_} ot={ot_} idle={idle_}")
            for c in cases:
                s = inst.surgeries[c.sid]
                print(
                    f"    S{c.sid:02d} ({s.priority}, team {s.team}, dur {s.duration})  {c.start}-{c.end}"
                )
    print(
        f"\nOT total={sum(sched.overtime.values())} | "
        f"Idle total={sum(sched.idle.values())} | "
        f"Violations={sched.violations}"
    )


# -------------------------
# Instancia ejemplo (para pruebas rápidas)
# -------------------------

def make_example_instance() -> Instance:
    "Instancia pequeña de ejemplo (3 quirófanos, 1 día, 480 min, 12 cirugías)."
    Q, D, H = 3, 1, 480
    teams = ["A", "B", "C"]
    team_avail = {(t, 1): 1 for t in teams}

    data = [
        (0, 90, "H", "A", 0),
        (1, 60, "M", "A", 0),
        (2, 120, "L", "B", 0),
        (3, 45, "H", "B", 30),
        (4, 180, "M", "A", 0),
        (5, 75, "L", "C", 0),
        (6, 150, "H", "C", 60),
        (7, 30, "M", "A", 0),
        (8, 60, "L", "B", 0),
        (9, 90, "M", "C", 0),
        (10, 50, "H", "A", 0),
        (11, 120, "M", "B", 0),
    ]
    surgeries = [Surgery(*row) for row in data]
    return Instance(Q=Q, D=D, H=H, surgeries=surgeries, team_avail=team_avail)
