# core.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math

Priority = str  # 'H'|'M'|'L'

@dataclass
class Surgery:
    sid: int
    duration: int
    priority: Priority
    team: str
    arrival: int = 0

@dataclass
class Instance:
    Q: int
    D: int
    H: int
    surgeries: List[Surgery]
    team_avail: Dict[Tuple[str,int], int]  # (team, day) -> 0/1

    def avail(self, team: str, d: int) -> int:
        return self.team_avail.get((team, d), 0)

@dataclass
class PlacedCase:
    sid: int
    day: int     # 1..D
    room: int    # 1..Q
    start: int
    end: int

@dataclass
class Schedule:
    placed: List[PlacedCase]
    room_day_end: Dict[Tuple[int,int], int]  # (room,day)
    overtime: Dict[Tuple[int,int], int]      # (room,day)
    idle: Dict[Tuple[int,int], int]          # (room,day)
    violations: int

# ---------- Decodificador (list scheduling) ----------
def decode_permutation_to_schedule(inst: Instance, perm: List[int]) -> Schedule:
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
        best = None  # (start, day, room)
        d0 = next_available_day(s.team, 1)
        d_candidates = [inst.D] if d0 is None else list(range(d0, inst.D+1))
        if d0 is None:
            violations += 1

        for d in d_candidates:
            if inst.avail(s.team, d) != 1:
                continue
            for q in range(1, inst.Q+1):
                start = max(endtime[(q,d)], s.arrival)
                tup = (start, d, q)
                if best is None or tup < best:
                    best = tup
            if best is not None and best[0] == s.arrival == 0:
                break

        if best is None:
            d = inst.D
            q = min(range(1, inst.Q+1), key=lambda r: endtime[(r,d)])
            start = max(endtime[(q,d)], s.arrival)
        else:
            start, d, q = best

        end = start + s.duration
        case = PlacedCase(sid=sid, day=d, room=q, start=start, end=end)
        timeline[(q,d)].append(case)
        endtime[(q,d)] = end
        placed.append(case)

    overtime: Dict[Tuple[int,int], int] = {}
    idle: Dict[Tuple[int,int], int] = {}
    for q in range(1, inst.Q+1):
        for d in range(1, inst.D+1):
            cases = timeline[(q,d)]
            last_end = endtime[(q,d)]
            ot = max(0, last_end - inst.H)
            worked = sum(min(c.end, inst.H) - min(c.start, inst.H) for c in cases if c.start < inst.H)
            idle[(q,d)] = max(0, inst.H - worked)
            overtime[(q,d)] = ot

    return Schedule(placed=placed, room_day_end=endtime, overtime=overtime, idle=idle, violations=violations)

# ---------- Fitness (menor es mejor) ----------
def fitness(inst: Instance, sched: Schedule,
            w_wait=0.5, w_ot=0.3, w_idle=0.2,
            alpha_H=3.0, alpha_M=1.0, alpha_L=0.5,
            penalty_per_violation=0.25) -> float:
    prio_w = {'H': alpha_H, 'M': alpha_M, 'L': alpha_L}
    start = {c.sid: c.start for c in sched.placed}

    W = 0.0
    for s in inst.surgeries:
        W += prio_w.get(s.priority, 1.0) * max(0, start.get(s.sid, 0) - s.arrival)
    O = sum(sched.overtime.values())
    U = sum(sched.idle.values())

    UB_W = len(inst.surgeries) * inst.H
    UB_O = inst.Q * inst.D * inst.H
    UB_U = inst.Q * inst.D * inst.H

    Wn = W/UB_W if UB_W else 0.0
    On = O/UB_O if UB_O else 0.0
    Un = U/UB_U if UB_U else 0.0

    base = w_wait*Wn + w_ot*On + w_idle*Un
    return base + sched.violations*penalty_per_violation

# ---------- Utilidades ----------
def print_schedule(inst: Instance, sched: Schedule, show_empty=True) -> None:
    by_room_day: Dict[Tuple[int,int], List[PlacedCase]] = {}
    for c in sched.placed:
        by_room_day.setdefault((c.room,c.day), []).append(c)

    for d in range(1, inst.D+1):
        print(f"\n=== Día {d} ===")
        for q in range(1, inst.Q+1):
            key = (q,d)
            cases = sorted(by_room_day.get(key, []), key=lambda c: c.start)
            end_  = sched.room_day_end.get(key, 0)
            ot_   = sched.overtime.get(key, 0)
            idle_ = sched.idle.get(key, inst.H)
            if not show_empty and not cases and end_ == 0 and ot_ == 0 and idle_ == inst.H:
                continue
            print(f"  QF {q}: end={end_} ot={ot_} idle={idle_}")
            for c in cases:
                s = inst.surgeries[c.sid]
                print(f"    S{c.sid:02d} ({s.priority}, {s.team}, dur {s.duration}) {c.start}-{c.end}")
    print(f"\nOT total={sum(sched.overtime.values())} | Idle total={sum(sched.idle.values())} | Violaciones={sched.violations}")

# Un dataset pequeño para pruebas rápidas
def make_example_instance() -> Instance:
    Q, D, H = 3, 1, 480
    teams = ['A','B','C']
    team_avail = {(t,1):1 for t in teams}
    data = [
        (0,90,'H','A',0),(1,60,'M','A',0),(2,120,'L','B',0),(3,45,'H','B',30),
        (4,180,'M','A',0),(5,75,'L','C',0),(6,150,'H','C',60),(7,30,'M','A',0),
        (8,60,'L','B',0),(9,90,'M','C',0),(10,50,'H','A',0),(11,120,'M','B',0)
    ]
    surgeries = [Surgery(*row) for row in data]
    return Instance(Q=Q, D=D, H=H, surgeries=surgeries, team_avail=team_avail)
