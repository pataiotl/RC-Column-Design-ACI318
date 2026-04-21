import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import math
import numpy as np
from fpdf import FPDF
import tempfile
import os

# ============================================================
# CONSTANTS
# ============================================================
ES = 200_000
ECU = 0.003
PHI_TIED = 0.65
PHI_TENSION = 0.90
PHI_SHEAR = 0.75

# ============================================================
# PAGE CONFIG & GLOBAL CSS
# ============================================================
st.set_page_config(
    page_title="RC Column Designer",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* ── Base font size reduction ── */
html, body, [class*="css"] { font-size: 13px; }

/* ── Main title ── */
h1 { font-size: 1.35rem !important; font-weight: 700 !important; letter-spacing: -0.5px; }
h2 { font-size: 1.05rem !important; font-weight: 600 !important; }
h3 { font-size: 0.95rem !important; font-weight: 600 !important; }

/* ── Sidebar tightening ── */
[data-testid="stSidebar"] { min-width: 260px !important; max-width: 280px !important; }
[data-testid="stSidebar"] .block-container { padding: 0.75rem 0.75rem 1rem !important; }
[data-testid="stSidebar"] label { font-size: 11px !important; color: #555 !important; font-weight: 500; }
[data-testid="stSidebar"] .stNumberInput input,
[data-testid="stSidebar"] .stSelectbox select { font-size: 12px !important; padding: 2px 6px !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { font-size: 0.78rem !important; text-transform: uppercase;
    letter-spacing: 0.08em; color: #888; margin: 0.8rem 0 0.25rem; font-weight: 600 !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] { background: #f7f8fa; border: 1px solid #e5e7eb;
    border-radius: 8px; padding: 10px 14px !important; }
[data-testid="stMetricLabel"] { font-size: 10px !important; color: #777 !important;
    text-transform: uppercase; letter-spacing: 0.06em; }
[data-testid="stMetricValue"] { font-size: 1.4rem !important; font-weight: 700 !important; }

/* ── Caption & info ── */
.stCaption { font-size: 11px !important; color: #888; }
.stAlert p { font-size: 12px !important; }

/* ── Dataframe ── */
.stDataFrame { font-size: 11px !important; }

/* ── Buttons ── */
.stButton button, .stDownloadButton button {
    font-size: 12px !important; padding: 6px 16px !important;
    border-radius: 6px !important;
}

/* ── Tab labels ── */
.stTabs [data-baseweb="tab"] { font-size: 12px !important; padding: 6px 14px !important; }

/* ── Section divider helper ── */
.section-label {
    font-size: 10px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.1em; color: #aaa; margin: 0 0 4px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# ENGINEERING ENGINES  (unchanged from original)
# ============================================================

def beta1(fc):
    return 0.85 if fc <= 28 else max(0.65, 0.85 - 0.05 * (fc - 28) / 7)

def get_Po(b, h, fc, fy, Ast):
    return 0.85 * fc * (b * h - Ast) + fy * Ast

def check_rho_g(b, h, Ast):
    rho = Ast / (b * h)
    return round(rho * 100, 2), rho >= 0.01, rho <= 0.08

def check_bar_fit(b, h, nw, nd, dia, cover, tie_d):
    min_clear = max(1.5 * dia, 40.0)
    avail_w = b - 2 * cover - 2 * tie_d
    avail_h = h - 2 * cover - 2 * tie_d
    req_w = nw * dia + (nw - 1) * min_clear
    req_h = nd * dia + (nd - 1) * min_clear
    return req_w <= avail_w and req_h <= avail_h, round(req_w, 1), round(req_h, 1)

def calculate_tie_spacing(b, h, long_dia, tie_d):
    s = min(16 * long_dia, 48 * tie_d, min(b, h))
    return int(math.floor(s / 25) * 25)

def build_layers(depth, d_prime, n_width, n_depth, bar_area):
    layers = [{'area': n_width * bar_area, 'd': d_prime}]
    if n_depth > 2:
        spacing = (depth - 2 * d_prime) / (n_depth - 1)
        for i in range(1, n_depth - 1):
            layers.append({'area': 2 * bar_area, 'd': d_prime + i * spacing})
    layers.append({'area': n_width * bar_area, 'd': depth - d_prime})
    return layers

def generate_pm_curve(width, depth, fc, fy, cover, n_width, n_depth, bar_dia, tie_d):
    b1 = beta1(fc)
    ey = fy / ES
    d_prime = cover + tie_d + bar_dia / 2
    bar_area = math.pi * bar_dia ** 2 / 4
    layers = build_layers(depth, d_prime, n_width, n_depth, bar_area)
    Ast = sum(L['area'] for L in layers)
    Po = get_Po(width, depth, fc, fy, Ast)
    Pn_max = 0.80 * Po

    points = []
    c = depth * 2.0
    c_step = max(depth / 400, 1.0)

    while c > c_step:
        a = min(b1 * c, depth)
        Cc = 0.85 * fc * a * width
        Mc = Cc * (depth / 2 - a / 2)
        Fs = Ms = 0.0
        for L in layers:
            eps = ECU * (c - L['d']) / c
            fs = min(fy, max(-fy, eps * ES))
            F = L['area'] * fs
            if L['d'] <= a and eps > 0:
                F -= L['area'] * 0.85 * fc
            Fs += F
            Ms += F * (depth / 2 - L['d'])
        Pn = Cc + Fs
        Mn = Mc + Ms
        eps_t = ECU * (layers[-1]['d'] - c) / c
        if eps_t <= ey:
            phi = PHI_TIED
        elif eps_t >= ey + 0.003:
            phi = PHI_TENSION
        else:
            phi = PHI_TIED + (PHI_TENSION - PHI_TIED) * (eps_t - ey) / 0.003
        design_P = min(phi * Pn, PHI_TIED * Pn_max) / 1_000
        design_M = abs(phi * Mn) / 1_000_000
        points.append({'Moment_kNm': round(design_M, 1), 'Axial_kN': round(design_P, 1)})
        c -= c_step

    Pt = -(PHI_TENSION * Ast * fy) / 1_000
    points.append({'Moment_kNm': 0.0, 'Axial_kN': round(Pt, 1)})
    return pd.DataFrame(points)

def get_dc_ratio(pm_df, P_demand, M_demand):
    P_max = pm_df['Axial_kN'].max()
    P_min = pm_df['Axial_kN'].min()
    M_abs = abs(M_demand)
    if P_demand > P_max:
        return round(P_demand / P_max, 3)
    if P_demand < P_min:
        return 9.99
    if M_abs < 1e-6:
        return round(P_demand / P_max, 3) if P_demand >= 0 else round(abs(P_demand) / abs(P_min), 3)
    e_demand = M_abs / (abs(P_demand) + 1e-6)
    pm = pm_df.copy()
    pm['e'] = pm['Moment_kNm'] / (pm['Axial_kN'].abs() + 1e-6)
    side = pm[pm['Axial_kN'] >= 0].copy() if P_demand >= 0 else pm[pm['Axial_kN'] < 0].copy()
    if side.empty:
        return 9.99
    side['R_cap'] = (side['Moment_kNm'] ** 2 + side['Axial_kN'] ** 2) ** 0.5
    R_demand = (M_abs ** 2 + P_demand ** 2) ** 0.5
    side = side.sort_values('e').reset_index(drop=True)
    idx = side['e'].searchsorted(e_demand)
    if idx == 0:
        R_cap = side.loc[0, 'R_cap']
    elif idx >= len(side):
        R_cap = side.loc[len(side) - 1, 'R_cap']
    else:
        e0, e1 = side.loc[idx - 1, 'e'], side.loc[idx, 'e']
        R0, R1 = side.loc[idx - 1, 'R_cap'], side.loc[idx, 'R_cap']
        t = (e_demand - e0) / (e1 - e0) if abs(e1 - e0) > 1e-9 else 0.5
        R_cap = R0 + t * (R1 - R0)
    return 9.99 if R_cap <= 0 else round(R_demand / R_cap, 3)

def magnify_moment(Pu, Mu, width, depth, fc, lu, k, Cm, beta_dns):
    if Pu <= 0:
        return round(abs(Mu), 1), 1.0
    M_min = Pu * (15 + 0.03 * depth) / 1_000
    Mu_eff = max(abs(Mu), 1e-3)
    r = 0.3 * depth
    klu_r = (k * lu) / r
    if klu_r <= 34:
        return round(max(Mu_eff, M_min), 1), 1.0
    Ec = 4_700 * math.sqrt(fc)
    Ig = width * depth ** 3 / 12
    EI = 0.4 * Ec * Ig / (1 + beta_dns)
    Pc = math.pi ** 2 * EI / (k * lu) ** 2 / 1_000
    if Pu >= 0.75 * Pc:
        return 9_999.9, 999.9
    delta = max(Cm / (1 - Pu / (0.75 * Pc)), 1.0)
    return round(max(delta * Mu_eff, M_min), 1), round(delta, 3)

def dynamic_alpha(Pu, Po_kN):
    if Pu <= 0 or Po_kN <= 0:
        return 1.15
    return round(min(max(1.15 + 0.35 * min(Pu / (PHI_TIED * Po_kN), 1.0), 1.15), 1.50), 3)

def biaxial_pmm(dc2, dc3, alpha):
    if dc2 >= 9.0 or dc3 >= 9.0:
        return 9.99
    return round((dc2 ** alpha + dc3 ** alpha) ** (1 / alpha), 3)

def column_shear_capacity(b, d, fc, Pu, Ag, fyt, Av, s):
    Nu = Pu * 1_000
    Vc = max(0.17 * (1 + Nu / (14 * Ag)) * math.sqrt(fc) * b * d, 0)
    Vs = (Av * fyt * d) / s if s > 0 else 0
    return {'phi_Vn_kN': round(PHI_SHEAR * (Vc + Vs) / 1_000, 1),
            'phi_Vc_kN': round(PHI_SHEAR * Vc / 1_000, 1)}

def run_optimizer(df, b, h, fc, fy, cover, lu_2, lu_3, k_2, k_3, Cm_2, Cm_3, beta_dns, tie_d):
    Ag = b * h
    bars = {'DB16': 16, 'DB20': 20, 'DB25': 25, 'DB28': 28, 'DB32': 32}
    configs = []
    for name, dia in bars.items():
        area = math.pi * dia ** 2 / 4
        for nw in range(3, 15):
            for nd in range(3, 15):
                total = 2 * nw + 2 * (nd - 2)
                Ast = total * area
                rho = Ast / Ag
                fits, _, _ = check_bar_fit(b, h, nw, nd, dia, cover, tie_d)
                if 0.01 <= rho <= 0.08 and fits:
                    configs.append({'label': f"{total}-{name} ({nw}×{nd})",
                                    'Ast': Ast, 'nw': nw, 'nd': nd, 'dia': dia, 'total_bars': total})
    configs.sort(key=lambda x: (round(x['Ast'], -2), x['total_bars']))
    for cfg in configs:
        pm2 = generate_pm_curve(h, b, fc, fy, cover, cfg['nd'], cfg['nw'], cfg['dia'], tie_d)
        pm3 = generate_pm_curve(b, h, fc, fy, cover, cfg['nw'], cfg['nd'], cfg['dia'], tie_d)
        Po_kN = get_Po(b, h, fc, fy, cfg['Ast']) / 1_000
        ok = True
        for _, row in df.iterrows():
            P = row['P_Demand_kN']
            Mc2, _ = magnify_moment(P, row['M2_Demand_kNm'], h, b, fc, lu_2, k_2, Cm_2, beta_dns)
            Mc3, _ = magnify_moment(P, row['M3_Demand_kNm'], b, h, fc, lu_3, k_3, Cm_3, beta_dns)
            dc2 = get_dc_ratio(pm2, P, Mc2)
            dc3 = get_dc_ratio(pm3, P, Mc3)
            if biaxial_pmm(dc2, dc3, dynamic_alpha(P, Po_kN)) > 1.0:
                ok = False
                break
        if ok:
            return cfg
    return None

def create_pdf(frame_name, b, h, fc, fy, layout_text, tie_text, rho_g,
               max_ratio, max_combo, klu_r_2, klu_r_3, fig):
    pdf = FPDF()
    pdf.add_page()
    W = 190
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(W, 9, f"RC Column Design — Frame: {frame_name}", ln=True, align='C')
    pdf.set_font("Arial", '', 8)
    pdf.set_text_color(130, 130, 130)
    pdf.cell(W, 5, "ACI 318-19 | Non-sway | PCA Load Contour biaxial | Radial D/C interpolation", ln=True, align='C')
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)

    def sec(t):
        pdf.set_font("Arial", 'B', 10)
        pdf.set_fill_color(235, 237, 240)
        pdf.cell(W, 7, f"  {t}", ln=True, fill=True)
        pdf.set_font("Arial", '', 10)
        pdf.ln(1)

    def row(lbl, val):
        pdf.cell(75, 5, f"  {lbl}", border=0)
        pdf.cell(W - 75, 5, str(val), ln=True)

    sec("Section")
    row("Dimensions", f"{b} × {h} mm")
    row("f'c / fy", f"{fc} MPa / {fy} MPa")
    row("Ag", f"{int(b * h):,} mm²")
    pdf.ln(2)
    sec("Reinforcement")
    row("Longitudinal", layout_text)
    row("ρg", f"{rho_g} %")
    row("Ties", tie_text)
    pdf.ln(2)
    sec("Slenderness")
    row("klu/r axis-2 / axis-3", f"{round(klu_r_2,1)} / {round(klu_r_3,1)}")
    row("Slenderness limit", "34 (non-sway, conservative)")
    pdf.ln(2)
    sec("Result")
    status = "FAIL" if max_ratio > 1.0 else "PASS"
    pdf.set_text_color(*(180, 30, 30) if max_ratio > 1.0 else (20, 140, 60))
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(W, 7, f"  {status}   |   Biaxial PMM = {max_ratio}", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 10)
    row("Governing combo", max_combo)
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(130, 130, 130)
    pdf.cell(W, 5, "  * α = 1.15–1.50 dynamic; D/C via radial eccentricity interpolation", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)
    sec("P-M Interaction Diagram")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        fig.savefig(tmp.name, format="png", bbox_inches="tight", dpi=180)
        img_path = tmp.name
    pdf.image(img_path, x=30, w=140)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp2:
        pdf.output(tmp2.name)
        pdf_path = tmp2.name
    with open(pdf_path, "rb") as f:
        data = f.read()
    for p in (img_path, pdf_path):
        try: os.remove(p)
        except: pass
    return data


# ============================================================
# IMPROVED PMM CHART
# ============================================================

def make_pmm_chart(pm2, pm3, df, frame_id, layout_text):
    """
    Clean, professional PMM interaction diagram:
    - Both axes plotted on the same chart with distinct colours
    - Demand points sized and coloured by PMM ratio (red = failing)
    - Balance point annotated
    - Tight axis limits with 5% padding
    - Gridlines subdued; axes through origin prominent
    """
    BLUE   = "#1a6fad"
    ORANGE = "#e07b00"
    GRAY   = "#c0c4cc"

    fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=110)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")

    # ── PM curves ──────────────────────────────────────────
    for pm, color, lbl in [(pm2, BLUE, "Axis-2  (M2–P)"),
                            (pm3, ORANGE, "Axis-3  (M3–P)")]:
        s = pm.sort_values('Axial_kN').drop_duplicates()
        ax.plot(s['Moment_kNm'], s['Axial_kN'],
                color=color, linewidth=2.2, label=lbl, zorder=3)
        # Shade inside the curve lightly
        ax.fill_betweenx(s['Axial_kN'], 0, s['Moment_kNm'],
                          color=color, alpha=0.04, zorder=1)

    # ── Demand points ──────────────────────────────────────
    pmm_vals = df['PMM'].clip(0, 2.0)
    # Colourmap: green (< 0.7) → yellow → red (> 1.0)
    cmap = plt.cm.RdYlGn_r
    norm = plt.Normalize(0.5, 1.2)

    # M2 demands (squares)
    sc2 = ax.scatter(df['M2_Mag'], df['P_Demand_kN'],
                     c=pmm_vals, cmap=cmap, norm=norm,
                     s=55, marker='s', zorder=5,
                     edgecolors='white', linewidths=0.6,
                     label='M2 demands')
    # M3 demands (triangles)
    sc3 = ax.scatter(df['M3_Mag'], df['P_Demand_kN'],
                     c=pmm_vals, cmap=cmap, norm=norm,
                     s=55, marker='^', zorder=5,
                     edgecolors='white', linewidths=0.6,
                     label='M3 demands')

    # Highlight governing (max PMM) point
    gov_idx = df['PMM'].idxmax()
    gov_M2  = df.loc[gov_idx, 'M2_Mag']
    gov_M3  = df.loc[gov_idx, 'M3_Mag']
    gov_P   = df.loc[gov_idx, 'P_Demand_kN']
    gov_pmm = df.loc[gov_idx, 'PMM']
    ax.scatter([gov_M2, gov_M3], [gov_P, gov_P],
               s=130, marker='*', color='#d00000', zorder=7,
               label=f'Governing  PMM={gov_pmm:.3f}', edgecolors='white', linewidths=0.5)

    # ── Reference lines ────────────────────────────────────
    ax.axhline(0, color='#333', linewidth=0.9, zorder=2)
    ax.axvline(0, color='#333', linewidth=0.9, zorder=2)

    # ── Colourbar ──────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, aspect=22)
    cbar.set_label('PMM ratio', fontsize=8, labelpad=4)
    cbar.ax.tick_params(labelsize=7)
    cbar.ax.axhline(y=1.0, color='#d00000', linewidth=1.2, linestyle='--')

    # ── Axes labels & ticks ────────────────────────────────
    ax.set_xlabel("φMn  (kNm)", fontsize=9, labelpad=4)
    ax.set_ylabel("φPn  (kN)",  fontsize=9, labelpad=4)
    ax.tick_params(axis='both', labelsize=8, length=3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(6, integer=True))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(7, integer=True))

    # Tight limits with padding
    all_M = pd.concat([pm2['Moment_kNm'], pm3['Moment_kNm'], df['M2_Mag'], df['M3_Mag']])
    all_P = pd.concat([pm2['Axial_kN'],   pm3['Axial_kN'],   df['P_Demand_kN']])
    m_pad = (all_M.max() - all_M.min()) * 0.07
    p_pad = (all_P.max() - all_P.min()) * 0.07
    ax.set_xlim(-m_pad, all_M.max() + m_pad)
    ax.set_ylim(all_P.min() - p_pad, all_P.max() + p_pad)

    # ── Grid ───────────────────────────────────────────────
    ax.grid(True, linestyle=':', linewidth=0.5, color='#ccc', zorder=0)

    # ── Legend ─────────────────────────────────────────────
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=7.5, framealpha=0.9,
              loc='lower right', handlelength=1.8,
              borderpad=0.6, labelspacing=0.4)

    ax.set_title(f"{frame_id}   ·   {layout_text}",
                 fontsize=9, fontweight='600', pad=8, color='#333')

    plt.tight_layout(pad=0.8)
    return fig


# ============================================================
# SIDEBAR
# ============================================================
sb = st.sidebar

sb.markdown("### Section")
b  = sb.number_input("Width b (mm) — axis-2", value=800, step=50, min_value=200)
h  = sb.number_input("Depth h (mm) — axis-3", value=800, step=50, min_value=200)
fc = sb.number_input("f'c (MPa)", value=40, step=5, min_value=20)
fy = sb.number_input("fy  (MPa)", value=500, step=10, min_value=300)

sb.markdown("### Ties")
tie_map  = {'RB9': 9, 'DB10': 10, 'DB12': 12}
sel_tie  = sb.selectbox("Tie size", list(tie_map.keys()), index=1)
tie_d    = tie_map[sel_tie]
cover    = sb.number_input("Clear cover (mm)", value=40, step=5, min_value=20)

sb.markdown("### Longitudinal")
auto_opt = sb.checkbox("🚀 Auto-optimise", value=False)
bar_map  = {'DB16': 16, 'DB20': 20, 'DB25': 25, 'DB28': 28, 'DB32': 32}

if not auto_opt:
    col_a, col_b = sb.columns(2)
    nw      = col_a.number_input("Bars (width)", value=4, step=1, min_value=2)
    nd      = col_b.number_input("Bars (depth)", value=4, step=1, min_value=2)
    sel_bar = sb.selectbox("Bar size", list(bar_map.keys()), index=2)
    bar_dia = bar_map[sel_bar]
    total_bars = 2 * nw + 2 * (nd - 2)
    Ast = total_bars * math.pi * bar_dia ** 2 / 4
    rho_pct, above_min, below_max = check_rho_g(b, h, Ast)
    fits, req_w, req_h = check_bar_fit(b, h, nw, nd, bar_dia, cover, tie_d)
    rho_icon = "✅" if above_min and below_max else "❌"
    sb.markdown(f"**{total_bars} bars · {round(Ast,0):.0f} mm² · ρg {rho_pct}% {rho_icon}**")
    if not above_min: sb.error("ρg < 1% — ACI §10.6.1.1")
    if not below_max: sb.error("ρg > 8% — ACI §10.6.1.1")
    if not fits:      sb.error(f"Bars don't fit — needs {req_w}/{req_h} mm")

sb.markdown("### Slenderness")
c1, c2 = sb.columns(2)
lu_2 = c1.number_input("lu-2 (mm)", value=3000, step=100, min_value=500)
lu_3 = c2.number_input("lu-3 (mm)", value=3000, step=100, min_value=500)
c3, c4 = sb.columns(2)
k_2 = c3.number_input("k (axis 2)", value=1.0, step=0.05, min_value=0.5)
k_3 = c4.number_input("k (axis 3)", value=1.0, step=0.05, min_value=0.5)
sb.caption("Cm = 0.6−0.4·(M1/M2)  — use 1.0 if unknown")
c5, c6 = sb.columns(2)
Cm_2 = c5.slider("Cm-2", 0.20, 1.00, 1.00, 0.05)
Cm_3 = c6.slider("Cm-3", 0.20, 1.00, 1.00, 0.05)
beta_dns = sb.slider("βdns", 0.0, 1.0, 0.0, 0.05)

sb.markdown("### Shear (optional)")
check_shear = sb.checkbox("Check shear capacity")
fyt_shear = sb.number_input("Tie fy for shear (MPa)", value=400, step=10, min_value=250) if check_shear else 400


# ============================================================
# MAIN AREA — HEADER
# ============================================================
st.title("🏗️ RC Column Designer — ACI 318-19")
st.caption("Non-sway · Discrete 4-face rebar · PCA biaxial (Bresler-Parme) · Radial D/C interpolation")
st.warning("**Scope:** Non-sway frames only. Seismic (Ch. 18) not checked. Preliminary design only.", icon="⚠️")

# ============================================================
# FILE UPLOAD
# ============================================================
uploaded = st.file_uploader("Upload SAP2000 frame-forces CSV", type=["csv"],
                             label_visibility="collapsed",
                             help="Expected columns: Frame, OutputCase, P, M2, M3")
if uploaded is None:
    st.info("Upload a SAP2000 frame-forces CSV to begin.  Required columns: Frame, OutputCase, P, M2, M3.")
    st.stop()

df_raw = pd.read_csv(uploaded)
if 'Frame' in df_raw.columns:
    first = str(df_raw['Frame'].iloc[0]).strip().lower()
    if first in ('text', 'unitless'):
        df_raw = df_raw.drop(0).reset_index(drop=True)
for col in ['P', 'M2', 'M3', 'V2', 'V3']:
    if col in df_raw.columns:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
if 'Frame' not in df_raw.columns:
    st.error("CSV must contain a 'Frame' column.")
    st.stop()

sb.markdown("### Select column")
frame_id = sb.selectbox("Frame", df_raw['Frame'].unique())

df = df_raw[df_raw['Frame'] == frame_id].copy()
df['Load_Combo']     = df['OutputCase']
df['P_Demand_kN']   = df['P'] * -1
df['M2_Demand_kNm'] = df['M2']
df['M3_Demand_kNm'] = df['M3']

st.caption("⚠️ SAP2000 sign: P is negated (compression → positive). Verify before use.")

# ============================================================
# AUTO-OPTIMISE
# ============================================================
if auto_opt:
    with st.spinner("Optimising rebar grid…"):
        best = run_optimizer(df, b, h, fc, fy, cover, lu_2, lu_3, k_2, k_3, Cm_2, Cm_3, beta_dns, tie_d)
    if best is None:
        st.error("Optimisation failed — section cannot satisfy demands within 1–8% ρg.")
        st.stop()
    nw, nd, bar_dia = best['nw'], best['nd'], best['dia']
    Ast = best['Ast']
    total_bars = best['total_bars']
    rho_pct = round(Ast / (b * h) * 100, 2)
    st.success(f"Optimised: **{best['label']}**  ·  ρg = {rho_pct}%")
else:
    if not fits:
        st.error("Fix bar-fit error in sidebar.")
        st.stop()
    if not (above_min and below_max):
        st.error("Fix ρg error in sidebar.")
        st.stop()

# ============================================================
# CALCULATIONS
# ============================================================
tie_spacing = calculate_tie_spacing(b, h, bar_dia, tie_d)
layout_text = f"{total_bars}-DB{bar_dia} ({nw}×{nd})"
tie_text    = f"{sel_tie} @ {tie_spacing} mm"

r2 = 0.3 * b
r3 = 0.3 * h
klu_r_2 = (k_2 * lu_2) / r2
klu_r_3 = (k_3 * lu_3) / r3

df[['M2_Mag','Delta_2']] = df.apply(
    lambda r: pd.Series(magnify_moment(r['P_Demand_kN'], r['M2_Demand_kNm'], h, b, fc, lu_2, k_2, Cm_2, beta_dns)), axis=1)
df[['M3_Mag','Delta_3']] = df.apply(
    lambda r: pd.Series(magnify_moment(r['P_Demand_kN'], r['M3_Demand_kNm'], b, h, fc, lu_3, k_3, Cm_3, beta_dns)), axis=1)

pm2 = generate_pm_curve(h, b, fc, fy, cover, nd, nw, bar_dia, tie_d)
pm3 = generate_pm_curve(b, h, fc, fy, cover, nw, nd, bar_dia, tie_d)
Po_kN = get_Po(b, h, fc, fy, Ast) / 1_000

df['DC_2']  = df.apply(lambda r: get_dc_ratio(pm2, r['P_Demand_kN'], r['M2_Mag']), axis=1)
df['DC_3']  = df.apply(lambda r: get_dc_ratio(pm3, r['P_Demand_kN'], r['M3_Mag']), axis=1)
df['Alpha'] = df.apply(lambda r: dynamic_alpha(r['P_Demand_kN'], Po_kN), axis=1)
df['PMM']   = df.apply(lambda r: biaxial_pmm(r['DC_2'], r['DC_3'], r['Alpha']), axis=1)

max_idx   = df['PMM'].idxmax()
max_ratio = df.loc[max_idx, 'PMM']
max_combo = df.loc[max_idx, 'Load_Combo']

# Shear
shear_info = {}
if check_shear and 'V2' in df.columns:
    Ag = b * h
    d2 = b - cover - tie_d - bar_dia / 2
    d3 = h - cover - tie_d - bar_dia / 2
    Av = 2 * math.pi * tie_d ** 2 / 4
    Pu_avg = df[df['P_Demand_kN'] > 0]['P_Demand_kN'].mean() if df['P_Demand_kN'].gt(0).any() else 0
    sv2 = column_shear_capacity(h, d2, fc, Pu_avg, Ag, fyt_shear, Av, tie_spacing)
    sv3 = column_shear_capacity(b, d3, fc, Pu_avg, Ag, fyt_shear, Av, tie_spacing)
    V2_max = df['V2'].abs().max()
    V3_max = df['V3'].abs().max() if 'V3' in df.columns else 0
    shear_info = {
        'V2_max': round(V2_max,1), 'phi_Vn2': sv2['phi_Vn_kN'],
        'V3_max': round(V3_max,1), 'phi_Vn3': sv3['phi_Vn_kN'],
        'pass2': V2_max <= sv2['phi_Vn_kN'],
        'pass3': V3_max <= sv3['phi_Vn_kN'],
    }

# ============================================================
# RESULTS
# ============================================================
st.markdown("---")

# ── Metric strip ───────────────────────────────────────────
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("PMM ratio", max_ratio)
m2.metric("ρg", f"{rho_pct}%")
m3.metric("klu/r  (2)", round(klu_r_2,1))
m4.metric("klu/r  (3)", round(klu_r_3,1))
m5.metric("Tie spacing", f"{tie_spacing} mm")
m6.metric("Total bars", total_bars)

# ── Status banner ──────────────────────────────────────────
if max_ratio > 1.0:
    st.error(f"**FAIL** — PMM {max_ratio}  ·  {max_combo}  ·  {layout_text}")
elif max_ratio > 0.90:
    st.warning(f"**MARGINAL** — PMM {max_ratio}  ·  {max_combo}  ·  {layout_text}")
else:
    st.success(f"**PASS** — PMM {max_ratio}  ·  {max_combo}  ·  {layout_text}  ·  Ties: {tie_text}")

# ── Slenderness warnings ────────────────────────────────────
max_delta = max(df['Delta_2'].max(), df['Delta_3'].max())
for axis, klu_r in [("axis-2", klu_r_2), ("axis-3", klu_r_3)]:
    if klu_r > 100:
        st.error(f"klu/r = {round(klu_r,1)} ({axis}) — extremely slender. Use second-order analysis.")
    elif klu_r > 34:
        st.warning(f"klu/r = {round(klu_r,1)} ({axis}) — slender; magnification applied.")
if max_delta >= 990:
    st.error("**BUCKLING FAILURE** — Pu ≥ 0.75·Pc. Increase section or reduce lu.")
elif max_delta > 1.4:
    st.warning(f"Max δ = {round(max_delta,2)} > 1.4 — significant slenderness amplification.")

# ── Shear check ────────────────────────────────────────────
if shear_info:
    st.markdown("**Shear  (ACI §22.5.6.1)**")
    sh1, sh2 = st.columns(2)
    sh1.metric(f"{'✅' if shear_info['pass2'] else '❌'} φVn (axis-2)",
               f"{shear_info['phi_Vn2']} kN", f"Vu = {shear_info['V2_max']} kN")
    sh2.metric(f"{'✅' if shear_info['pass3'] else '❌'} φVn (axis-3)",
               f"{shear_info['phi_Vn3']} kN", f"Vu = {shear_info['V3_max']} kN")
    if not shear_info['pass2'] or not shear_info['pass3']:
        st.error("Shear demand exceeds capacity — increase tie size/frequency or enlarge section.")

st.markdown("---")

# ── Two-column layout: chart | table ───────────────────────
chart_col, table_col = st.columns([5, 7])

with chart_col:
    st.markdown("**P-M Interaction Diagram**")
    st.caption("Both axes plotted · demand colour = PMM ratio · ★ = governing combo")
    fig = make_pmm_chart(pm2, pm3, df, frame_id, layout_text)
    st.pyplot(fig, use_container_width=True)
    st.caption(
        "ℹ️ Biaxial: PCA Load Contour $(DC_2)^α+(DC_3)^α=1$, "
        "α = 1.15–1.50 (axial-dependent). "
        "Points inside a curve may still fail the biaxial check."
    )

with table_col:
    st.markdown("**Load-combination results**")
    disp = ['Load_Combo','P_Demand_kN','M2_Mag','Delta_2','DC_2','M3_Mag','Delta_3','DC_3','Alpha','PMM']

    def _hl_d(v):
        if isinstance(v, (int,float)):
            if v >= 990: return 'background:#6b0000;color:white'
            if v > 1.4:  return 'background:#ffe0e0;color:#900'
        return ''

    def _hl_pmm(v):
        if isinstance(v, (int,float)):
            if v > 1.0: return 'background:#6b0000;color:white;font-weight:bold'
            if v > 0.9: return 'background:#fff0c0;color:#7a5000'
        return ''

    styled = (df[disp].style
              .map(_hl_d,   subset=['Delta_2','Delta_3'])
              .map(_hl_pmm, subset=['PMM'])
              .format({'P_Demand_kN':'{:.1f}','M2_Mag':'{:.1f}','M3_Mag':'{:.1f}',
                       'Delta_2':'{:.3f}','Delta_3':'{:.3f}',
                       'DC_2':'{:.3f}','DC_3':'{:.3f}',
                       'Alpha':'{:.2f}','PMM':'{:.3f}'}))
    st.dataframe(styled, use_container_width=True, height=420)

# ── PDF export ─────────────────────────────────────────────
st.markdown("---")
pdf_col, _ = st.columns([2, 5])
with pdf_col:
    pdf_bytes = create_pdf(frame_id, b, h, fc, fy, layout_text, tie_text,
                           rho_pct, max_ratio, max_combo, klu_r_2, klu_r_3, fig)
    st.download_button("📥 Download PDF report", data=pdf_bytes,
                       file_name=f"Column_{frame_id}.pdf", mime="application/pdf")
