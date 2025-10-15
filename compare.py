# compare.py
from core import make_example_instance, print_schedule, fitness
from GA import GAConfig, run_ga
from SA import SAConfig, run_sa
from PSO import PSOConfig, run_pso

if __name__ == "__main__":
    inst = make_example_instance()

    fit_kwargs = dict(w_wait=0.6, w_ot=0.25, w_idle=0.15,
                      alpha_H=3.0, alpha_M=1.0, alpha_L=0.5,
                      penalty_per_violation=0.25)

    # GA
    ga_cfg = GAConfig(pop_size=60, generations=120, elite_frac=0.15, seed=7)
    ga_perm, ga_sched, ga_fit, ga_hist = run_ga(inst, ga_cfg, fitness_kwargs=fit_kwargs)
    print("\n=== GA ===")
    print("Fitness:", ga_fit)
    print_schedule(inst, ga_sched)

    # SA (arranca desde GA para acelerar)
    sa_cfg = SAConfig(t0=0.1, alpha=0.95, tmin=1e-4, iters_per_T=250, seed=11, p_move=0.35)
    sa_perm, sa_sched, sa_fit, sa_hist = run_sa(inst, sa_cfg, fitness_kwargs=fit_kwargs, start_perm=ga_perm)
    print("\n=== SA ===")
    print("Fitness:", sa_fit)
    print_schedule(inst, sa_sched)

    # PSO
    pso_cfg = PSOConfig(swarm_size=40, iterations=150, seed=19)
    pso_perm, pso_sched, pso_fit, pso_hist = run_pso(inst, pso_cfg, fitness_kwargs=fit_kwargs)
    print("\n=== PSO ===")
    print("Fitness:", pso_fit)
    print_schedule(inst, pso_sched)

    print("\n=== RESUMEN ===")
    print(f"GA : {ga_fit:.4f}   | iters={len(ga_hist)}")
    print(f"SA : {sa_fit:.4f}   | nivelesT={len(sa_hist)}")
    print(f"PSO: {pso_fit:.4f}  | iters={len(pso_hist)}")
