# app.py — Interfaz Streamlit para programar cirugías con GA / SA / PSO
# Ejecuta: streamlit run app.py

import json
from io import BytesIO
import pandas as pd
import streamlit as st
import altair as alt

from core import Instance, Surgery, decode_permutation_to_schedule, fitness
from GA import GAConfig, run_ga
from SA import SAConfig, run_sa
from PSO import PSOConfig, run_pso

st.set_page_config(page_title="Programación de Cirugías", layout="wide")
st.title("🩺 Programación de Cirugías (GA / SA / PSO)")

# ---------------------------
# 1) Parámetros del horizonte
# ---------------------------
st.sidebar.header("Parámetros del horizonte")
Q = st.sidebar.number_input("Quirófanos (Q)", min_value=1, value=3, step=1)
D = st.sidebar.number_input("Días (D)", min_value=1, value=1, step=1)
H = st.sidebar.number_input("Jornada (min, H)", min_value=1, value=480, step=15)

st.sidebar.header("Pesos del fitness")
w_wait = st.sidebar.slider("w_wait (espera)", 0.0, 1.0, 0.60, 0.05)
w_ot   = st.sidebar.slider("w_ot (horas extra)", 0.0, 1.0, 0.25, 0.05)
w_idle = st.sidebar.slider("w_idle (idle)", 0.0, 1.0, 0.15, 0.05)
total_w = w_wait + w_ot + w_idle
if total_w == 0:
    st.sidebar.warning("Ajusta al menos un peso > 0")
alpha_H = st.sidebar.number_input("α Alta", min_value=0.0, value=3.0, step=0.5)
alpha_M = st.sidebar.number_input("α Media", min_value=0.0, value=1.0, step=0.5)
alpha_L = st.sidebar.number_input("α Baja", min_value=0.0, value=0.5, step=0.5)

# ---------------------------
# 2) Datos (tablas)
# ---------------------------
st.header("1) Datos del problema")

st.markdown("### Cirugías")
default_surgeries = pd.DataFrame([
    {"sid": 0, "duration": 90,  "priority": "H", "team": "A", "arrival": 0},
    {"sid": 1, "duration": 60,  "priority": "M", "team": "A", "arrival": 0},
    {"sid": 2, "duration": 120, "priority": "L", "team": "B", "arrival": 0},
    {"sid": 3, "duration": 45,  "priority": "H", "team": "B", "arrival": 30},
    {"sid": 4, "duration": 180, "priority": "M", "team": "A", "arrival": 0},
])
surgeries_df = st.data_editor(
    default_surgeries, num_rows="dynamic",
    column_config={
        "sid": st.column_config.NumberColumn("sid", step=1),
        "duration": st.column_config.NumberColumn("duración (min)", step=5),
        "priority": st.column_config.SelectboxColumn("prioridad", options=["H","M","L"]),
        "team": st.column_config.TextColumn("equipo"),
        "arrival": st.column_config.NumberColumn("arribo (min)", step=5),
    },
    use_container_width=True
)

st.markdown("### Disponibilidad de equipos por día")
teams = sorted({row["team"] for _, row in surgeries_df.iterrows() if row.get("team")})
team_avail_df = pd.DataFrame(
    [{"team": t, "day": d, "avail": 1} for t in teams for d in range(1, D+1)]
) if teams else pd.DataFrame(columns=["team","day","avail"])
team_avail_df = st.data_editor(
    team_avail_df, num_rows="dynamic",
    column_config={
        "team": st.column_config.TextColumn("equipo"),
        "day": st.column_config.NumberColumn("día", step=1, min_value=1),
        "avail": st.column_config.NumberColumn("disponible (0/1)", min_value=0, max_value=1, step=1),
    },
    use_container_width=True
)

# ---------------------------
# 3) Construir instancia
# ---------------------------
def build_instance(Q, D, H, surgeries_df, team_avail_df) -> Instance:
    "Valida y construye la instancia desde las tablas de la UI."
    if surgeries_df.empty:
        return Instance(Q=Q, D=D, H=H, surgeries=[], team_avail={})
    if surgeries_df["sid"].duplicated().any():
        st.error("Hay 'sid' duplicados en cirugías."); st.stop()
    if not set(surgeries_df["priority"]).issubset({"H","M","L"}):
        st.error("Prioridades deben ser H/M/L."); st.stop()
    if (surgeries_df["duration"] <= 0).any() or (surgeries_df["arrival"] < 0).any():
        st.error("Duración debe ser >0 y arribo >=0."); st.stop()

    surgeries = [Surgery(int(r.sid), int(r.duration), str(r.priority), str(r.team), int(r.arrival))
                 for _, r in surgeries_df.iterrows()]
    team_av = {}
    for _, r in team_avail_df.iterrows():
        try:
            d = int(r.day)
            team_av[(str(r.team), d)] = int(r.avail)
        except Exception:
            pass
    return Instance(Q=Q, D=D, H=H, surgeries=surgeries, team_avail=team_av)

inst = build_instance(Q, D, H, surgeries_df, team_avail_df)

# ---------------------------
# 4) Elegir algoritmo / config
# ---------------------------
st.header("2) Algoritmo y configuración")

algo = st.selectbox("Elige algoritmo", ["GA", "SA", "PSO"], index=0)
seed = st.number_input("Semilla", value=7, step=1)

fit_kwargs = dict(
    w_wait=w_wait/total_w if total_w>0 else 0.0,
    w_ot=w_ot/total_w if total_w>0 else 0.0,
    w_idle=w_idle/total_w if total_w>0 else 0.0,
    alpha_H=alpha_H, alpha_M=alpha_M, alpha_L=alpha_L,
    penalty_per_violation=0.25
)

col1, col2 = st.columns(2)
if algo == "GA":
    with col1:
        pop_size = st.number_input("Población", 10, 2000, 60, 5)
        generations = st.number_input("Generaciones", 10, 10000, 120, 10)
        elite_frac = st.slider("Elite frac", 0.0, 0.5, 0.15, 0.01)
    with col2:
        cx_prob = st.slider("Prob. crossover", 0.0, 1.0, 0.9, 0.05)
        mut_prob = st.slider("Prob. mutación", 0.0, 1.0, 0.3, 0.05)
        tournament_k = st.number_input("Torneo k", 2, 20, 3, 1)
elif algo == "SA":
    with col1:
        t0 = st.number_input("T0", min_value=0.0001, value=0.1, step=0.05, format="%.5f")
        alpha = st.slider("alpha (enfriamiento)", 0.80, 0.99, 0.95, 0.01)
        tmin = st.number_input("Tmin", min_value=1e-6, value=1e-4, step=1e-5, format="%.6f")
    with col2:
        iters_per_T = st.number_input("Iteraciones por T", 10, 10000, 250, 10)
        p_move = st.slider("Prob. usar INSERT (vs SWAP)", 0.0, 1.0, 0.35, 0.05)
else:
    with col1:
        swarm_size = st.number_input("Tamaño del enjambre", 10, 5000, 40, 5)
        iterations = st.number_input("Iteraciones", 10, 10000, 150, 10)
    with col2:
        w_inertia = st.slider("w_inertia", 0.0, 1.5, 0.7, 0.05)
        c_cog = st.slider("c_cog", 0.0, 3.0, 1.4, 0.1)
        c_soc = st.slider("c_soc", 0.0, 3.0, 1.4, 0.1)

# ---------------------------
# 5) Utilidades: KPIs y Excel
# ---------------------------
def kpis_from_schedule(inst, sched, alpha_H, alpha_M, alpha_L):
    "Calcula KPIs a partir del calendario (útil para tablas/gráficos/export)."
    start_map = {c.sid: c.start for c in sched.placed}
    prio_w = {'H': alpha_H, 'M': alpha_M, 'L': alpha_L}
    wait_weighted = 0.0
    for s in inst.surgeries:
        wait_weighted += prio_w[s.priority] * max(0, start_map.get(s.sid, 0) - s.arrival)
    return {
        "wait_weighted": wait_weighted,
        "overtime": sum(sched.overtime.values()),
        "idle": sum(sched.idle.values()),
        "violations": sched.violations
    }

def export_results_to_excel(inst, results_dict, fit_kwargs) -> bytes:
    """
    Crea un Excel en memoria con:
      - 'Resumen': KPIs por algoritmo + gráfico de barras y convergencia.
      - 'Calendario_<ALG>': calendario ordenado por día/sala + fitness.
      - 'Hist_<ALG>': historial de convergencia.
      - 'Quartiles_GA' si existe.
      - 'Params': Q, D, H y pesos del fitness.
    """
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        book = writer.book

        # Resumen KPIs
        resumen_rows = []
        for algo, d in results_dict.items():
            k = d["kpi_df"].iloc[0].to_dict()
            resumen_rows.append({
                "Algoritmo": algo,
                "Fitness": d.get("fitness"),
                "Wait_weighted": k.get("wait_weighted", 0),
                "Overtime": k.get("overtime", 0),
                "Idle": k.get("idle", 0),
                "Violations": k.get("violations", 0)
            })
        resumen_df = pd.DataFrame(resumen_rows)
        resumen_df.to_excel(writer, sheet_name="Resumen", index=False, startrow=0)

        sheet = writer.sheets["Resumen"]
        # Barra de KPIs
        chart_bar = book.add_chart({"type": "column"})
        for col, name in zip([2, 3, 4, 5], ["Wait_weighted", "Overtime", "Idle", "Violations"]):
            chart_bar.add_series({
                "name": name,
                "categories": ["Resumen", 1, 0, len(resumen_df), 0],
                "values": ["Resumen", 1, col, len(resumen_df), col],
            })
        chart_bar.set_title({"name": "KPIs por algoritmo"})
        chart_bar.set_x_axis({"name": "Algoritmo"})
        chart_bar.set_y_axis({"name": "Valor"})
        sheet.insert_chart("H2", chart_bar, {"x_scale": 1.2, "y_scale": 1.2})

        # Convergencia
        hist_max = max(len(d["history"]) for d in results_dict.values())
        hist_df = pd.DataFrame({"Iter": list(range(hist_max))})
        for algo, d in results_dict.items():
            h = d["history"]
            hist_df[algo] = [h[i] if i < len(h) else None for i in range(hist_max)]
        start_hist_row = len(resumen_df) + 4
        hist_df.to_excel(writer, sheet_name="Resumen", index=False, startrow=start_hist_row)
        chart_line = book.add_chart({"type": "line"})
        for idx, algo in enumerate(results_dict.keys()):
            chart_line.add_series({
                "name": algo,
                "categories": ["Resumen", start_hist_row + 1, 0, start_hist_row + hist_max, 0],
                "values": ["Resumen", start_hist_row + 1, idx + 1, start_hist_row + hist_max, idx + 1],
            })
        chart_line.set_title({"name": "Convergencia (menor es mejor)"})
        chart_line.set_x_axis({"name": "Iteración"})
        chart_line.set_y_axis({"name": "Fitness"})
        sheet.insert_chart("H22", chart_line, {"x_scale": 1.2, "y_scale": 1.2})

        # Hojas por algoritmo
        for algo, d in results_dict.items():
            d["schedule_df"].to_excel(writer, sheet_name=f"Calendario_{algo}", index=False)
            pd.DataFrame({"best_fitness": [d.get("fitness")]}).to_excel(
                writer, sheet_name=f"Calendario_{algo}", index=False, startrow=len(d["schedule_df"]) + 2
            )
            pd.DataFrame({"best": d["history"]}).to_excel(writer, sheet_name=f"Hist_{algo}", index=False)
            # Cuartiles del GA (si existen)
            if algo == "GA" and "quartiles" in d:
                d["quartiles"].to_excel(writer, sheet_name=f"Quartiles_{algo}", index=False)

        # Parámetros
        params_df = pd.DataFrame([{
            "Q": inst.Q, "D": inst.D, "H": inst.H,
            "w_wait": fit_kwargs.get("w_wait"),
            "w_ot": fit_kwargs.get("w_ot"),
            "w_idle": fit_kwargs.get("w_idle"),
            "alpha_H": fit_kwargs.get("alpha_H"),
            "alpha_M": fit_kwargs.get("alpha_M"),
            "alpha_L": fit_kwargs.get("alpha_L")
        }])
        params_df.to_excel(writer, sheet_name="Params", index=False)

    buffer.seek(0)
    return buffer.getvalue()

# ---------------------------
# 6) Ejecutar / comparar
# ---------------------------
st.header("3) Ejecutar y ver resultados")

colA, colB = st.columns(2)
run = colA.button("🚀 Ejecutar algoritmo seleccionado", type="primary")
run_all = colB.button("📊 Comparar GA + SA + PSO y exportar Excel", type="secondary")

# Ejecutar UNO
if run:
    if algo == "GA":
        cfg = GAConfig(pop_size=int(pop_size), generations=int(generations),
                       elite_frac=float(elite_frac), cx_prob=float(cx_prob),
                       mut_prob=float(mut_prob), tournament_k=int(tournament_k),
                       seed=int(seed))
        # GA devuelve 5 valores (incluye estadísticos por generación)
        best_perm, best_sched, best_fit, history, ga_stats = run_ga(inst, cfg, fitness_kwargs=fit_kwargs)
    elif algo == "SA":
        cfg = SAConfig(t0=float(t0), alpha=float(alpha), tmin=float(tmin),
                       iters_per_T=int(iters_per_T), seed=int(seed), p_move=float(p_move))
        best_perm, best_sched, best_fit, history = run_sa(inst, cfg, fitness_kwargs=fit_kwargs, start_perm=None)
        ga_stats = None
    else:
        cfg = PSOConfig(swarm_size=int(swarm_size), iterations=int(iterations),
                        w_inertia=float(w_inertia), c_cog=float(c_cog), c_soc=float(c_soc), seed=int(seed))
        best_perm, best_sched, best_fit, history = run_pso(inst, cfg, fitness_kwargs=fit_kwargs)
        ga_stats = None

    st.success(f"Mejor fitness ({algo}): {best_fit:.6f}")

    # KPIs
    kpi_dict = kpis_from_schedule(inst, best_sched, alpha_H, alpha_M, alpha_L)
    st.subheader("KPIs")
    st.dataframe(pd.DataFrame([kpi_dict]), use_container_width=True)

    # Calendario
    cal_df = pd.DataFrame([{"sid": c.sid, "day": c.day, "room": c.room, "start": c.start, "end": c.end}
                           for c in best_sched.placed]).sort_values(["day","room","start"]).reset_index(drop=True)
    st.subheader("Calendario resultante")
    st.dataframe(cal_df, use_container_width=True)

    # Convergencia (mejor histórico)
    st.subheader("Convergencia")
    st.line_chart(pd.DataFrame({"best": history}), height=250)

    # DISTRIBUCIÓN (solo GA) con leyenda + tooltips (ahora sí, ya existe ga_stats)
    if algo == "GA" and ga_stats:
        st.subheader("Distribución del fitness por generación (GA)")
        stats_df = pd.DataFrame(ga_stats)  # gen,min,q1,median,q3,max,mean,std,best

        area_iqr = alt.Chart(stats_df).mark_area(opacity=0.25).encode(
            x=alt.X('gen:Q', title='Generación'),
            y=alt.Y('q1:Q', title='Fitness'),
            y2='q3:Q'
        )

        line_data = stats_df[['gen', 'median', 'best', 'min', 'max']].melt(
            'gen', var_name='Serie', value_name='Fitness'
        )

        color_scale = alt.Scale(
            domain=['best', 'median', 'min', 'max'],
            range=['#2ca02c', '#000000', '#888888', '#888888']
        )

        line_chart = alt.Chart(line_data).mark_line().encode(
            x='gen:Q',
            y='Fitness:Q',
            color=alt.Color('Serie:N', title='Serie', scale=color_scale),
            tooltip=[
                alt.Tooltip('gen:Q', title='Generación'),
                alt.Tooltip('Serie:N', title='Serie'),
                alt.Tooltip('Fitness:Q', title='Fitness', format='.5f')
            ]
        )

        points = alt.Chart(line_data).mark_circle(size=30, opacity=0.6).encode(
            x='gen:Q',
            y='Fitness:Q',
            color=alt.Color('Serie:N', scale=color_scale, legend=None),
            tooltip=[
                alt.Tooltip('gen:Q', title='Generación'),
                alt.Tooltip('Serie:N', title='Serie'),
                alt.Tooltip('Fitness:Q', title='Fitness', format='.5f')
            ]
        )

        st.altair_chart(
            (area_iqr + line_chart + points).properties(height=280),
            use_container_width=True
        )

        st.markdown("""
**Cómo leer la gráfica (GA):**
- **Verde (best)**: mejor **histórico** hasta cada generación.
- **Negro (median)**: **mediana** de la población en esa generación.
- **Gris inferior (min)**: mejor individuo **de esa generación**.
- **Gris superior (max)**: peor individuo **de esa generación**.
- **Banda sombreada**: rango **Q1–Q3** (50% central de la población).

**Recuerda:** fitness más **bajo** es **mejor**.  
Banda **ancha** = **diversidad**; banda **estrecha** = **convergencia**.
Si el **min** no toca el **best**, el mejor histórico no está en esa cohorte.
""")

# Ejecutar TODOS + Excel
if run_all:
    cfg_ga  = GAConfig(pop_size=60, generations=120, elite_frac=0.15, cx_prob=0.9, mut_prob=0.3, tournament_k=3, seed=int(seed))
    cfg_sa  = SAConfig(t0=0.1, alpha=0.95, tmin=1e-4, iters_per_T=250, seed=int(seed)+1, p_move=0.35)
    cfg_pso = PSOConfig(swarm_size=40, iterations=150, w_inertia=0.7, c_cog=1.4, c_soc=1.4, seed=int(seed)+2)

    ga_perm, ga_sched, ga_fit, ga_hist, ga_stats = run_ga(inst, cfg_ga, fitness_kwargs=fit_kwargs)
    ga_kpi = pd.DataFrame([kpis_from_schedule(inst, ga_sched, alpha_H, alpha_M, alpha_L)])
    ga_cal = pd.DataFrame([{"sid": c.sid, "day": c.day, "room": c.room, "start": c.start, "end": c.end}
                           for c in ga_sched.placed]).sort_values(["day","room","start"]).reset_index(drop=True)

    sa_perm, sa_sched, sa_fit, sa_hist = run_sa(inst, cfg_sa, fitness_kwargs=fit_kwargs, start_perm=ga_perm)
    sa_kpi = pd.DataFrame([kpis_from_schedule(inst, sa_sched, alpha_H, alpha_M, alpha_L)])
    sa_cal = pd.DataFrame([{"sid": c.sid, "day": c.day, "room": c.room, "start": c.start, "end": c.end}
                           for c in sa_sched.placed]).sort_values(["day","room","start"]).reset_index(drop=True)

    pso_perm, pso_sched, pso_fit, pso_hist = run_pso(inst, cfg_pso, fitness_kwargs=fit_kwargs)
    pso_kpi = pd.DataFrame([kpis_from_schedule(inst, pso_sched, alpha_H, alpha_M, alpha_L)])
    pso_cal = pd.DataFrame([{"sid": c.sid, "day": c.day, "room": c.room, "start": c.start, "end": c.end}
                           for c in pso_sched.placed]).sort_values(["day","room","start"]).reset_index(drop=True)

    st.success("Comparación completada")
    st.subheader("KPIs por algoritmo")
    resumen = pd.DataFrame([
        {"Algoritmo":"GA",  **ga_kpi.iloc[0].to_dict(),  "fitness": ga_fit},
        {"Algoritmo":"SA",  **sa_kpi.iloc[0].to_dict(),  "fitness": sa_fit},
        {"Algoritmo":"PSO", **pso_kpi.iloc[0].to_dict(), "fitness": pso_fit},
    ])
    st.bar_chart(resumen.set_index("Algoritmo")[["wait_weighted","overtime","idle","violations"]])

    st.subheader("Convergencia (menor es mejor)")
    conv_df = pd.DataFrame({"Iter": list(range(max(len(ga_hist), len(sa_hist), len(pso_hist))))})
    conv_df["GA"]  = [ga_hist[i]  if i<len(ga_hist)  else None for i in range(len(conv_df))]
    conv_df["SA"]  = [sa_hist[i]  if i<len(sa_hist)  else None for i in range(len(conv_df))]
    conv_df["PSO"] = [pso_hist[i] if i<len(pso_hist) else None for i in range(len(conv_df))]
    st.line_chart(conv_df.set_index("Iter"))

    # Exportar Excel (incluye cuartiles del GA)
    results_dict = {
        "GA":  {"schedule_df": ga_cal,  "kpi_df": ga_kpi,  "history": ga_hist,  "fitness": ga_fit,
                "quartiles": pd.DataFrame(ga_stats)},
        "SA":  {"schedule_df": sa_cal,  "kpi_df": sa_kpi,  "history": sa_hist,  "fitness": sa_fit},
        "PSO": {"schedule_df": pso_cal, "kpi_df": pso_kpi, "history": pso_hist, "fitness": pso_fit},
    }
    xlsx_bytes = export_results_to_excel(inst, results_dict, fit_kwargs)
    st.download_button("⬇️ Descargar Excel comparativo",
                       data=xlsx_bytes,
                       file_name="comparativa_programacion.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
