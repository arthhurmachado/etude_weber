# -*- coding: utf-8 -*-
# Streamlit app para "Ligne de 4 convoyeurs — 2 barquettes — règle « pointe avant »"
# Conversão do seu código para web (sliders no navegador + 2 gráficos matplotlib).

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import streamlit as st

st.set_page_config(page_title="Convoyeurs — 2 barquettes (pointe avant)", layout="wide")

# =========================
# Parâmetros fixos
# =========================
L1, L2, L3, L4 = 9.0, 1.50, 1.50, 1.10   # [m]
v2, v3, v4 = 13.0, 20.0, 11.0            # [m/min]
ECART_CIBLE = 0.15                       # 15 cm

# =========================
# Funções do modelo (mesmas do seu script, adaptadas)
# =========================
def construire_ligne():
    longueurs = np.array([L1, L2, L3, L4], dtype=float)
    debuts    = np.cumsum([0.0] + longueurs[:-1].tolist())
    fins      = np.cumsum(longueurs)
    return longueurs, debuts, fins, float(fins[-1])

def indice_segment_depuis_pointe(s_centre, Lb, debuts, fins, S_total):
    s_pointe = s_centre + Lb/2.0
    if s_pointe >= S_total:
        return len(debuts) - 1
    for i, (a, b) in enumerate(zip(debuts, fins)):
        if a <= s_pointe < b:
            return i
    return 0

def simuler_positions(s0, vitesses_mpm, temps, Lb, debuts, fins, S_total):
    v_mps = np.array(vitesses_mpm, dtype=float) / 60.0
    s = np.empty_like(temps, dtype=float); s[0] = s0
    if len(temps) == 1:
        return s
    dt = float(temps[1] - temps[0])
    for k in range(1, len(temps)):
        i = indice_segment_depuis_pointe(s[k-1], Lb, debuts, fins, S_total)
        v = float(v_mps[i])
        s_suiv = s[k-1] + v*dt
        # Clamp no fim da linha, considerando a meia barqueta (pointe avant)
        if s_suiv + Lb/2.0 > S_total:
            s_suiv = S_total - Lb/2.0
        s[k] = s_suiv
    return s

def ecart_signe(sA, sB, Lb):
    # e = ponta traseira da dianteira - ponta dianteira da traseira
    avant, arriere = (sA, sB) if sA >= sB else (sB, sA)
    return (avant - Lb/2.0) - (arriere + Lb/2.0)

def axe_temps(S_total):
    vmin_mps = 5.0/60.0  # menor v1 possível no UI = 5 m/min
    T_max = S_total / max(vmin_mps, 1e-6) + 5.0
    dt = 0.02
    return np.arange(0.0, T_max, dt), dt

def calculer_trajectoires(Lb_cm, g0_cm, v1_mpm):
    longueurs, debuts, fins, S_total = construire_ligne()
    temps, dt = axe_temps(S_total)
    Lb = Lb_cm/100.0
    g0 = g0_cm/100.0

    # Duas barquetas no início de C1 (A à frente, B atrás)
    sA0 = L1/2.0
    sB0 = sA0 - (Lb + g0)

    vitesses_mpm = [v1_mpm, v2, v3, v4]
    sA = simuler_positions(sA0, vitesses_mpm, temps, Lb, debuts, fins, S_total)
    sB = simuler_positions(sB0, vitesses_mpm, temps, Lb, debuts, fins, S_total)

    return dict(sA=sA, sB=sB, temps=temps, dt=dt, Lb=Lb, g0=g0,
                debuts=debuts, fins=fins, S_total=S_total,
                v1=v1_mpm, v2=v2, v3=v3, v4=v4)

def g0_ideal_cm(Lb_cm, v1_mpm, v4_mpm=v4, gf=ECART_CIBLE):
    r = v4_mpm / max(v1_mpm, 1e-9)
    g0_m = (gf - (Lb_cm/100.0)*(r - 1.0)) / r
    return 100.0 * g0_m  # cm

# =========================
# UI — controles
# =========================
st.title("Ligne de 4 convoyeurs — 2 barquettes — règle « pointe avant »")

with st.sidebar:
    st.subheader("Parâmetros")
    Lb_cm = st.slider("L_b [cm]", 8.0, 40.0, 21.0, 1.0)
    g0_cm = st.slider("g₀ [cm]", 0.0, 50.0, 0.0, 1.0)
    v1_mpm = st.slider("v1 [m/min]", 5.0, 20.0, 9.0, 0.1)

# Estado com os parâmetros atuais
etat = calculer_trajectoires(Lb_cm, g0_cm, v1_mpm)
t_max = float(etat["temps"][-1])

# Slider de tempo (na barra de topo da página)
t = st.slider("t [s]", 0.0, t_max, 0.0, 0.02)

# =========================
# Render do gráfico principal (Matplotlib)
# =========================
def render_main_plot(etat, t):
    # Preparar figura
    fig, ax = plt.subplots(figsize=(10, 4.8))

    # Desenhar os 4 convoyeurs
    for a, b in zip(etat["debuts"], etat["fins"]):
        ax.plot([a, b], [0, 0], linewidth=10)
    for i, (a, b) in enumerate(zip(etat["debuts"], etat["fins"]), start=1):
        ax.axvline(b, linestyle='--', linewidth=1)
        ax.text((a+b)/2.0, -0.85, f"C{i}", ha='center', va='top')

    ax.text(0.99, 0.02, "Objectif indicatif : écart ≥ 0,15 m à l’entrée C4",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9)
    ax.set_xlabel("Position curviligne s [m] (origine = début C1)")

    # Índice temporal
    idx = int(np.clip(round(t/etat["dt"]), 0, len(etat["temps"])-1))
    sAt = etat["sA"][idx]; sBt = etat["sB"][idx]

    # Barquetas como retângulos
    rectA = Rectangle((sAt - etat["Lb"]/2.0, -0.30), etat["Lb"], 0.60,
                      label="Barquette A (avant)", alpha=0.9, color="tab:blue")
    rectB = Rectangle((sBt - etat["Lb"]/2.0, -0.30), etat["Lb"], 0.60,
                      label="Barquette B (arrière)", alpha=0.9, color="tab:orange")
    ax.add_patch(rectA); ax.add_patch(rectB)
    ax.legend(loc="upper right", title="Légende")

    # Título com métricas
    e_signe = ecart_signe(sAt, sBt, etat["Lb"])
    e_pos = max(e_signe, 0.0)
    ax.set_title(
        f"t = {t:5.2f} s   |   Écart instantané (signé) = {e_signe:.3f} m   "
        f"|   Écart ≥ 0,15 m ? {'OUI' if e_pos >= ECART_CIBLE else 'NON'}"
    )

    # Limites
    marge = max(0.5, etat["Lb"])
    ax.set_xlim(-marge, etat["S_total"] + marge)
    ax.set_ylim(-1.0, 1.2)
    ax.set_yticks([])
    return fig

def render_aux_plot(Lb_cm, g0_cm, v1_mpm):
    fig_g, ax_g0 = plt.subplots(figsize=(7.5, 4.5))
    Lb_range = np.linspace(8.0, 40.0, 250)
    courbe = g0_ideal_cm(Lb_range, v1_mpm)
    ax_g0.plot(Lb_range, courbe, lw=2, label="g₀ idéal (v1 courant)")

    g0_ideal_actuel = g0_ideal_cm(Lb_cm, v1_mpm)
    ax_g0.axvline(Lb_cm, linestyle="--", linewidth=1, label=f"L_b actuel = {Lb_cm:.0f} cm")
    ax_g0.plot([Lb_cm], [g0_ideal_actuel], marker="o", ms=7,
               label=f"g₀ idéal @ L_b actuel ≈ {g0_ideal_actuel:.1f} cm")
    ax_g0.plot([Lb_cm], [g0_cm], marker="s", ms=7, label=f"g₀ actuel = {g0_cm:.1f} cm")

    ax_g0.set_title("g₀ idéal pour écart final = 15 cm (C4 fixé à 11 m/min)")
    ax_g0.set_xlabel("Longueur de la barquette L_b [cm]")
    ax_g0.set_ylabel("g₀ idéal [cm]")
    ax_g0.grid(True)
    ax_g0.legend(loc="best")
    return fig_g

col1, col2 = st.columns([2, 1], gap="large")
with col1:
    st.pyplot(render_main_plot(etat, t), clear_figure=True)
with col2:
    st.pyplot(render_aux_plot(Lb_cm, g0_cm, v1_mpm), clear_figure=True)

# Rodapé informativo
st.caption(
    f"v1={etat['v1']:.1f}, v2={etat['v2']:.1f}, v3={etat['v3']:.1f}, v4={etat['v4']:.1f} [m/min]   "
    f"|   L_b={etat['Lb']*100:.0f} cm   |   g₀={etat['g0']*100:.0f} cm"
)
