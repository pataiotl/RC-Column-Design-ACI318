import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from fpdf import FPDF
import tempfile
import os

# ============================================================
# CONSTANTS
# ============================================================
ES = 200_000       # MPa — steel modulus
ECU = 0.003        # ACI 318-19 §22.2.2.1 — ultimate concrete strain
PHI_TIED = 0.65    # ACI Table 21.2.1 — tied column
PHI_TENSION = 0.90 # ACI Table 21.2.1 — tension-controlled
PHI_SHEAR = 0.75   # ACI §21.2.1

# ============================================================
# 1.  MATERIAL / SECTION HELPERS
# ============================================================

def beta1(fc: float) -> float:
    """ACI 318-19 Table 22.2.2.4.3"""
    return 0.85 if fc <= 28 else max(0.65, 0.85 - 0.05 * (fc - 28) / 7)


def eps_y(fy: float) -> float:
    return fy / ES


def get_Po(b: float, h: float, fc: float, fy: float, Ast: float) -> float:
    """Nominal pure-axial capacity Po in N — ACI §22.4.2.2"""
    Ag = b * h
    return 0.85 * fc * (Ag - Ast) + fy * Ast


def check_rho_g(b: float, h: float, Ast: float) -> tuple[float, bool, bool]:
    """Returns (rho_g, above_min, below_max) — ACI §10.6.1.1"""
    rho = Ast / (b * h)
    return round(rho * 100, 2), rho >= 0.01, rho <= 0.08


def check_bar_fit(b: float, h: float, nw: int, nd: int,
                  dia: float, cover: float, tie_d: float) -> tuple[bool, float, float]:
    """
    Checks all four faces independently.
    ACI §25.2.3: clear ≥ max(1.5db, 40 mm) for columns.
    Returns (fits, required_width_face, required_depth_face).
    """
    min_clear = max(1.5 * dia, 40.0)
    avail_w = b - 2 * cover - 2 * tie_d
    avail_h = h - 2 * cover - 2 * tie_d
    req_w = nw * dia + (nw - 1) * min_clear
    req_h = nd * dia + (nd - 1) * min_clear
    return req_w <= avail_w and req_h <= avail_h, round(req_w, 1), round(req_h, 1)


def calculate_tie_spacing(b: float, h: float, long_dia: float, tie_d: float) -> int:
    """ACI §25.7.2.1 — non-seismic tied column maximum spacing."""
    s = min(16 * long_dia, 48 * tie_d, min(b, h))
    return int(math.floor(s / 25) * 25)   # round down to 25 mm


# ============================================================
# 2.  P-M INTERACTION CURVE  (ACI 318-19, Design strength)
# ============================================================

def build_layers(depth: float, d_prime: float,
                 n_width: int, n_depth: int, bar_area: float) -> list[dict]:
    """
    Discrete 4-face rebar array.
    Bending axis is about the section centroid (depth/2).
    """
    layers = [{'area': n_width * bar_area, 'd': d_prime}]
    if n_depth > 2:
        spacing = (depth - 2 * d_prime) / (n_depth - 1)
        for i in range(1, n_depth - 1):
            layers.append({'area': 2 * bar_area, 'd': d_prime + i * spacing})
    layers.append({'area': n_width * bar_area, 'd': depth - d_prime})
    return layers


def generate_pm_curve(width: float, depth: float,
                      fc: float, fy: float,
                      cover: float, n_width: int, n_depth: int,
                      bar_dia: float, tie_d: float) -> pd.DataFrame:
    """
    ACI 318-19 design P-M curve.
    Moment is taken about the section centroid.
    Sweeps c from 2×depth (pure compression) down to near zero.
    """
    b1 = beta1(fc)
    ey = eps_y(fy)
    d_prime = cover + tie_d + bar_dia / 2
    bar_area = math.pi * bar_dia ** 2 / 4
    layers = build_layers(depth, d_prime, n_width, n_depth, bar_area)
    Ast = sum(L['area'] for L in layers)

    Po = get_Po(width, depth, fc, fy, Ast)
    Pn_max = 0.80 * Po          # ACI §22.4.2.1  (tied)

    points = []
    c = depth * 2.0
    c_step = max(depth / 400, 1.0)   # finer step — ≥1 mm, ≤depth/400

    while c > c_step:
        a = min(b1 * c, depth)
        Cc = 0.85 * fc * a * width
        Mc = Cc * (depth / 2 - a / 2)   # moment of concrete about centroid

        Fs = Ms = 0.0
        for L in layers:
            eps = ECU * (c - L['d']) / c
            fs = min(fy, max(-fy, eps * ES))
            F = L['area'] * fs
            # subtract displaced concrete only inside stress block AND in compression
            if L['d'] <= a and eps > 0:
                F -= L['area'] * 0.85 * fc
            Fs += F
            Ms += F * (depth / 2 - L['d'])   # moment about centroid (sign is automatic)

        Pn = Cc + Fs
        Mn = Mc + Ms          # can be negative for some layers — correct by sign

        # eps_t at extreme tension layer
        eps_t = ECU * (c - layers[-1]['d']) / c

        # phi — ACI Table 21.2.2 (column bracket)
        if eps_t <= ey:
            phi = PHI_TIED
        elif eps_t >= ey + 0.003:
            phi = PHI_TENSION
        else:
            phi = PHI_TIED + (PHI_TENSION - PHI_TIED) * (eps_t - ey) / 0.003

        # Apply caps — φPn cannot exceed φ_tied × Pn_max (ACI §22.4.2.1)
        design_P = min(phi * Pn, PHI_TIED * Pn_max) / 1_000     # → kN
        design_M = abs(phi * Mn) / 1_000_000                    # → kNm  (always positive)

        points.append({'Moment_kNm': round(design_M, 1),
                       'Axial_kN':   round(design_P, 1)})
        c -= c_step

    # Pure tension point — ACI §21.2.1, φ = 0.90
    Pt = -(PHI_TENSION * Ast * fy) / 1_000
    points.append({'Moment_kNm': 0.0, 'Axial_kN': round(Pt, 1)})

    return pd.DataFrame(points)


# ============================================================
# 3.  D/C RATIO  (radial / eccentricity method — robust)
# ============================================================

def get_dc_ratio(pm_df: pd.DataFrame, P_demand: float, M_demand: float) -> float:
    """
    Computes demand/capacity ratio using a ray from the origin
    through the demand point (M, P) intersected with the PM curve.
    This is robust for non-monotonic curves.
    Falls back to axial ratio when the demand is above the curve cap.
    """
    P_max = pm_df['Axial_kN'].max()
    P_min = pm_df['Axial_kN'].min()
    M_abs = abs(M_demand)

    # Above axial cap
    if P_demand > P_max:
        return round(P_demand / P_max, 3)
    # Below tension limit
    if P_demand < P_min:
        return 9.99

    # Pure axial demand (no moment) — simple axial ratio
    if M_abs < 1e-6:
        if P_demand >= 0:
            return round(P_demand / P_max, 3)
        return round(abs(P_demand) / abs(P_min), 3)

    # Ray eccentricity
    e_demand = M_abs / (abs(P_demand) + 1e-6)   # avoid div/0 for P near 0

    # Compute eccentricity at every point on the curve
    pm = pm_df.copy()
    pm['e'] = pm['Moment_kNm'] / (pm['Axial_kN'].abs() + 1e-6)

    # Find the two curve points that bracket the demand eccentricity
    # among points on the same side (compression or tension)
    if P_demand >= 0:
        side = pm[pm['Axial_kN'] >= 0].copy()
    else:
        side = pm[pm['Axial_kN'] < 0].copy()

    if side.empty:
        return 9.99

    # Radial distance from origin for each curve point
    side['R_cap'] = (side['Moment_kNm'] ** 2 + side['Axial_kN'] ** 2) ** 0.5
    R_demand = (M_abs ** 2 + P_demand ** 2) ** 0.5

    # Sort by eccentricity and find bracketing points
    side = side.sort_values('e').reset_index(drop=True)
    idx = side['e'].searchsorted(e_demand)

    if idx == 0:
        R_cap = side.loc[0, 'R_cap']
    elif idx >= len(side):
        R_cap = side.loc[len(side) - 1, 'R_cap']
    else:
        # Linear interpolation of radial capacity between two bracketing eccentricities
        e0, e1 = side.loc[idx - 1, 'e'], side.loc[idx, 'e']
        R0, R1 = side.loc[idx - 1, 'R_cap'], side.loc[idx, 'R_cap']
        if abs(e1 - e0) < 1e-9:
            R_cap = (R0 + R1) / 2
        else:
            t = (e_demand - e0) / (e1 - e0)
            R_cap = R0 + t * (R1 - R0)

    if R_cap <= 0:
        return 9.99
    return round(R_demand / R_cap, 3)


# ============================================================
# 4.  MOMENT MAGNIFICATION  (ACI 318-19 §6.6.4 — non-sway)
# ============================================================

def magnify_moment(Pu: float, Mu: float,
                   width: float, depth: float,
                   fc: float, lu: float, k: float,
                   Cm: float, beta_dns: float) -> tuple[float, float]:
    """
    Returns (Mc_kNm, delta).
    Pu in kN, Mu in kNm, dimensions in mm.
    Non-sway frame: ACI §6.6.4.4.
    """
    # Tension / zero axial — no magnification needed
    if Pu <= 0:
        return round(abs(Mu), 1), 1.0

    # Minimum eccentricity moment — ACI §6.6.4.5.4
    M_min = Pu * (15 + 0.03 * depth) / 1_000    # kNm

    Mu_eff = max(abs(Mu), 1e-3)    # avoid log(0) in slenderness

    # Radius of gyration — ACI §6.2.5: r = 0.3h for rectangular
    r = 0.3 * depth
    klu_r = (k * lu) / r

    # ACI §6.6.4.3 — non-sway slenderness limit
    # Conservative limit: 34 (= 34 − 12×(M1/M2) when M1/M2 unknown and single curvature assumed)
    # Using 34 avoids being overly conservative for typical cases
    SLENDER_LIMIT = 34
    if klu_r <= SLENDER_LIMIT:
        return round(max(Mu_eff, M_min), 1), 1.0

    # Stiffness — ACI Eq. 6.6.4.4.4b (conservative, no need for Ieff)
    Ec = 4_700 * math.sqrt(fc)
    Ig = width * depth ** 3 / 12
    EI = 0.4 * Ec * Ig / (1 + beta_dns)

    # Critical buckling load — ACI Eq. 6.6.4.4.2
    Pc = math.pi ** 2 * EI / (k * lu) ** 2 / 1_000   # kN

    # Instability guard — ACI §6.6.4.5.2
    if Pu >= 0.75 * Pc:
        return 9_999.9, 999.9

    # Moment magnifier — ACI Eq. 6.6.4.5.2
    denom = 1 - Pu / (0.75 * Pc)
    delta = max(Cm / denom, 1.0)

    Mc = delta * Mu_eff
    return round(max(Mc, M_min), 1), round(delta, 3)


# ============================================================
# 5.  BIAXIAL CHECK  (PCA Load Contour / Bresler-Parme)
# ============================================================

def dynamic_alpha(Pu: float, Po_kN: float) -> float:
    """
    α transitions linearly from 1.15 (pure bending) to 1.50 (near pure compression).
    Reference: PCA Load Contour Method; MacGregor & Wight.
    """
    if Pu <= 0 or Po_kN <= 0:
        return 1.15
    ratio = min(Pu / (PHI_TIED * Po_kN), 1.0)
    return round(min(max(1.15 + 0.35 * ratio, 1.15), 1.50), 3)


def biaxial_pmm(dc2: float, dc3: float, alpha: float) -> float:
    """
    Load contour: (M2/M2o)^α + (M3/M3o)^α = 1.0 at capacity.
    Returns ratio; > 1.0 means failure.
    """
    # Guard against 9.99 sentinel values propagating as finite numbers
    if dc2 >= 9.0 or dc3 >= 9.0:
        return 9.99
    return round((dc2 ** alpha + dc3 ** alpha) ** (1 / alpha), 3)


# ============================================================
# 6.  SHEAR CAPACITY CHECK  (ACI 318-19 §22.5 — column)
# ============================================================

def column_shear_capacity(b: float, d: float, fc: float,
                           Pu: float, Ag: float,
                           fyt: float, Av: float, s: float) -> dict:
    """
    Pu in kN (positive = compression).
    Returns phi*Vn in kN.
    ACI §22.5.6.1 — simplified Vc for columns.
    """
    Nu = Pu * 1_000   # N
    # ACI Eq. 22.5.6.1
    Vc = 0.17 * (1 + Nu / (14 * Ag)) * math.sqrt(fc) * b * d  # N
    Vc = max(Vc, 0)

    # Vs from provided ties
    Vs = (Av * fyt * d) / s if s > 0 else 0   # N

    phi_Vn = PHI_SHEAR * (Vc + Vs) / 1_000   # kN
    phi_Vc = PHI_SHEAR * Vc / 1_000
    return {'phi_Vn_kN': round(phi_Vn, 1), 'phi_Vc_kN': round(phi_Vc, 1)}


# ============================================================
# 7.  OPTIMIZER
# ============================================================

def run_optimizer(df: pd.DataFrame, b: float, h: float,
                  fc: float, fy: float, cover: float,
                  lu: float, k: float, Cm: float, beta_dns: float,
                  tie_d: float) -> dict | None:
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
                    configs.append({
                        'label': f"{total}-{name} ({nw}×{nd})",
                        'Ast': Ast, 'nw': nw, 'nd': nd, 'dia': dia,
                        'total_bars': total
                    })

    # Sort: minimum steel first, then fewer bars (more practical)
    configs.sort(key=lambda x: (round(x['Ast'], -2), x['total_bars']))

    for cfg in configs:
        pm2 = generate_pm_curve(h, b, fc, fy, cover, cfg['nd'], cfg['nw'], cfg['dia'], tie_d)
        pm3 = generate_pm_curve(b, h, fc, fy, cover, cfg['nw'], cfg['nd'], cfg['dia'], tie_d)
        Po_kN = get_Po(b, h, fc, fy, cfg['Ast']) / 1_000

        ok = True
        for _, row in df.iterrows():
            P = row['P_Demand_kN']
            Mc2, _ = magnify_moment(P, row['M2_Demand_kNm'], h, b, fc, lu, k, Cm, beta_dns)
            Mc3, _ = magnify_moment(P, row['M3_Demand_kNm'], b, h, fc, lu, k, Cm, beta_dns)
            dc2 = get_dc_ratio(pm2, P, Mc2)
            dc3 = get_dc_ratio(pm3, P, Mc3)
            alpha = dynamic_alpha(P, Po_kN)
            if biaxial_pmm(dc2, dc3, alpha) > 1.0:
                ok = False
                break
        if ok:
            return cfg
    return None


# ============================================================
# 8.  PDF REPORT
# ============================================================

def create_pdf(frame_name: str, b: float, h: float, fc: float, fy: float,
               layout_text: str, tie_text: str, rho_g: float,
               max_ratio: float, max_combo: str,
               klu_r_2: float, klu_r_3: float,
               fig) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    W = 190   # usable page width

    # ---- Header ----
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(W, 10, f"RC Column Design Report — Frame: {frame_name}", ln=True, align='C')
    pdf.set_font("Arial", '', 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(W, 6, "ACI 318-19 | Non-sway frame | Biaxial: PCA Load Contour (Bresler-Parme, dynamic α)", ln=True, align='C')
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    def section(title):
        pdf.set_font("Arial", 'B', 12)
        pdf.set_fill_color(230, 230, 230)
        pdf.cell(W, 8, f"  {title}", ln=True, fill=True)
        pdf.set_font("Arial", '', 11)
        pdf.ln(1)

    def row(label, value):
        pdf.cell(80, 6, f"  {label}", border=0)
        pdf.cell(W - 80, 6, str(value), ln=True)

    section("1. Section Properties")
    row("Dimensions", f"{b} × {h} mm")
    row("Concrete f'c", f"{fc} MPa")
    row("Steel fy", f"{fy} MPa")
    row("Gross area Ag", f"{int(b * h)} mm²")
    pdf.ln(3)

    section("2. Reinforcement")
    row("Longitudinal layout", layout_text)
    row("Steel ratio ρg", f"{rho_g} %")
    row("Transverse ties", tie_text)
    pdf.ln(3)

    section("3. Slenderness (non-sway, ACI §6.6.4)")
    row("klu/r about axis 2", f"{round(klu_r_2, 1)}")
    row("klu/r about axis 3", f"{round(klu_r_3, 1)}")
    row("Slenderness limit", "34 (conservative, M1/M2 unknown)")
    pdf.ln(3)

    section("4. Design Result")
    status = "FAIL" if max_ratio > 1.0 else "PASS"
    colour = (220, 53, 69) if max_ratio > 1.0 else (40, 167, 69)
    pdf.set_text_color(*colour)
    pdf.set_font("Arial", 'B', 13)
    pdf.cell(W, 8, f"  Status: {status}   |   Max Biaxial PMM Ratio: {max_ratio}", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 11)
    row("Governing combo", max_combo)
    pdf.set_font("Arial", 'I', 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(W, 5, "  * Biaxial check per PCA Load Contour Method, alpha = 1.15 – 1.50 (dynamic, axial-load dependent)", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)

    section("5. P-M Interaction Diagram")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
        fig.savefig(tmp_img.name, format="png", bbox_inches="tight", dpi=200)
        img_path = tmp_img.name
        
    # Shrink width to 120mm and center it (210mm total width - 120mm / 2 = 45mm x-margin)
    pdf.image(img_path, x=45, w=120)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        pdf.output(tmp_pdf.name)
        pdf_path = tmp_pdf.name

    with open(pdf_path, "rb") as f:
        data = f.read()

    for p in (img_path, pdf_path):
        try: os.remove(p)
        except Exception: pass

    return data


# ============================================================
# 9.  STREAMLIT UI
# ============================================================

st.set_page_config(page_title="RC Column Designer", page_icon="🏗️", layout="wide")
st.title("🏗️ RC Column Designer — ACI 318-19")
st.caption(
    "Non-sway frame | Discrete 4-face rebar | "
    "PCA biaxial method | Radial D/C interpolation | "
    "Slenderness per §6.6.4"
)
st.warning(
    "**Scope notice:** This tool covers non-sway frames only. "
    "Seismic (Chapter 18) provisions are NOT checked. "
    "Output is for preliminary design — verify with a licensed engineer.",
    icon="⚠️"
)

# ---- Sidebar ----
sb = st.sidebar
sb.header("1 · Section")
b = sb.number_input("Width b (Axis-2 direction, mm)", value=800, step=50, min_value=200)
h = sb.number_input("Depth h (Axis-3 direction, mm)", value=800, step=50, min_value=200)
fc = sb.number_input("Concrete f'c (MPa)", value=40, step=5, min_value=20)
fy = sb.number_input("Steel fy (MPa)", value=500, step=10, min_value=300)

sb.header("2 · Transverse ties")
tie_map = {'RB9': 9, 'DB10': 10, 'DB12': 12}
sel_tie = sb.selectbox("Tie bar size", list(tie_map.keys()), index=1)
tie_d = tie_map[sel_tie]
cover = sb.number_input("Clear cover to tie face (mm)", value=40, step=5, min_value=20)

sb.header("3 · Longitudinal bars")
auto_opt = sb.checkbox("🚀 Auto-Design Optimizer", value=False)

bar_map = {'DB16': 16, 'DB20': 20, 'DB25': 25, 'DB28': 28, 'DB32': 32}

if not auto_opt:
    nw = sb.number_input("Bars on width face (3-dir)", value=4, step=1, min_value=2)
    nd = sb.number_input("Bars on depth face (2-dir)", value=4, step=1, min_value=2)
    sel_bar = sb.selectbox("Bar size", list(bar_map.keys()), index=2)
    bar_dia = bar_map[sel_bar]
    total_bars = 2 * nw + 2 * (nd - 2)
    Ast = total_bars * math.pi * bar_dia ** 2 / 4
    rho_pct, above_min, below_max = check_rho_g(b, h, Ast)
    fits, req_w, req_h = check_bar_fit(b, h, nw, nd, bar_dia, cover, tie_d)

    sb.markdown(f"**Bars:** {total_bars} · **Ast:** {round(Ast, 0):.0f} mm²")
    sb.markdown(f"**ρg:** {rho_pct} %  {'✅' if above_min and below_max else '❌'}")
    if not above_min:
        sb.error("ρg < 1 % — ACI §10.6.1.1")
    if not below_max:
        sb.error("ρg > 8 % — ACI §10.6.1.1")
    if not fits:
        sb.error(f"Bars don't fit! Needs {req_w} mm (width) / {req_h} mm (depth) clear — ACI §25.2.3")

sb.header("4 · Slenderness / creep")
lu = sb.number_input("Unsupported length lu (mm)", value=3_000, step=100, min_value=500)
k  = sb.number_input("Effective length factor k", value=1.0, step=0.05, min_value=0.5)
sb.caption("Cm — ACI §6.6.4.5.3a: Cm = 0.6 − 0.4·(M1/M2). Use 1.0 for transverse loads or if M1/M2 unknown.")
Cm = sb.slider("Cm factor", 0.20, 1.00, 1.00, 0.05)
beta_dns = sb.slider("βdns (sustained-load ratio)", 0.0, 1.0, 0.0, 0.05)

sb.header("5 · Shear check (optional)")
check_shear = sb.checkbox("Check column shear capacity")
fyt_shear = sb.number_input("Tie fy for shear (MPa)", value=400, step=10, min_value=250) if check_shear else 400

# ---- File upload ----
uploaded = st.file_uploader("Upload SAP2000 frame-forces CSV", type=["csv"])

if uploaded is None:
    st.info("Upload a SAP2000 frame-forces CSV to begin. Expected columns: Frame, OutputCase, P, M2, M3.")
    st.stop()

# ---- Parse CSV ----
df_raw = pd.read_csv(uploaded)

# Drop SAP2000 units-header row if present
if 'Frame' in df_raw.columns:
    first = str(df_raw['Frame'].iloc[0]).strip().lower()
    if first in ('text', 'unitless'):
        df_raw = df_raw.drop(0).reset_index(drop=True)

for col in ['P', 'M2', 'M3', 'V2', 'V3']:
    if col in df_raw.columns:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

if 'Frame' not in df_raw.columns:
    st.error("CSV must contain a 'Frame' column.  Check your SAP2000 export.")
    st.stop()

sb.header("6 · Select column")
frame_id = sb.selectbox("Frame (column)", df_raw['Frame'].unique())

df = df_raw[df_raw['Frame'] == frame_id].copy()
df['Load_Combo']     = df['OutputCase']
# SAP2000 default: compression is negative in P — flip to positive for ACI convention
df['P_Demand_kN']   = df['P'] * -1
df['M2_Demand_kNm'] = df['M2']
df['M3_Demand_kNm'] = df['M3']

st.caption(
    "**SAP2000 sign convention:** P is negated (compression positive). "
    "Verify this matches your export settings before using results."
)

# ---- Auto-optimizer ----
if auto_opt:
    with st.spinner("Optimising constructable rebar grid…"):
        best = run_optimizer(df, b, h, fc, fy, cover, lu, k, Cm, beta_dns, tie_d)
    if best is None:
        st.error("Optimisation failed — section cannot satisfy demands within 1–8 % ρg. Increase section size.")
        st.stop()
    nw, nd, bar_dia = best['nw'], best['nd'], best['dia']
    Ast = best['Ast']
    total_bars = best['total_bars']
    rho_pct = round(Ast / (b * h) * 100, 2)
    fits = True
    st.success(f"Optimised layout: **{best['label']}**  |  ρg = {rho_pct} %")
else:
    # Manual mode — halt if geometry is invalid
    if not fits:
        st.error("Fix the bar-fit error in the sidebar before running the design.")
        st.stop()
    if not (above_min and below_max):
        st.error("Fix the ρg error in the sidebar before running the design.")
        st.stop()

# ---- Tie spacing ----
tie_spacing = calculate_tie_spacing(b, h, bar_dia, tie_d)
layout_text = f"{total_bars}-DB{bar_dia}  ({nw}×{nd} grid)"
tie_text    = f"{sel_tie} @ {tie_spacing} mm (ACI §25.7.2)"

# ---- Slenderness parameters ----
r2 = 0.3 * b   # radius of gyration about axis 2 (bending in plane of h)
r3 = 0.3 * h   # radius of gyration about axis 3
klu_r_2 = (k * lu) / r2
klu_r_3 = (k * lu) / r3

# ---- Moment magnification ----
df[['M2_Mag', 'Delta_2']] = df.apply(
    lambda r: pd.Series(magnify_moment(r['P_Demand_kN'], r['M2_Demand_kNm'],
                                        h, b, fc, lu, k, Cm, beta_dns)), axis=1)
df[['M3_Mag', 'Delta_3']] = df.apply(
    lambda r: pd.Series(magnify_moment(r['P_Demand_kN'], r['M3_Demand_kNm'],
                                        b, h, fc, lu, k, Cm, beta_dns)), axis=1)

# ---- PM curves ----
# Axis-2 bending: neutral axis || axis 2 → use (width=h, depth=b) with nd, nw bars
pm2 = generate_pm_curve(h, b, fc, fy, cover, nd, nw, bar_dia, tie_d)
# Axis-3 bending: neutral axis || axis 3 → use (width=b, depth=h) with nw, nd bars
pm3 = generate_pm_curve(b, h, fc, fy, cover, nw, nd, bar_dia, tie_d)
Po_kN = get_Po(b, h, fc, fy, Ast) / 1_000

# ---- D/C ratios ----
df['DC_2']  = df.apply(lambda r: get_dc_ratio(pm2, r['P_Demand_kN'], r['M2_Mag']), axis=1)
df['DC_3']  = df.apply(lambda r: get_dc_ratio(pm3, r['P_Demand_kN'], r['M3_Mag']), axis=1)
df['Alpha'] = df.apply(lambda r: dynamic_alpha(r['P_Demand_kN'], Po_kN), axis=1)
df['PMM']   = df.apply(lambda r: biaxial_pmm(r['DC_2'], r['DC_3'], r['Alpha']), axis=1)

max_idx   = df['PMM'].idxmax()
max_ratio = df.loc[max_idx, 'PMM']
max_combo = df.loc[max_idx, 'Load_Combo']

# ---- Shear check ----
shear_info = {}
if check_shear and 'V2' in df.columns and 'V3' in df.columns:
    Ag = b * h
    d2 = b - cover - tie_d - bar_dia / 2   # effective depth for V2
    d3 = h - cover - tie_d - bar_dia / 2   # effective depth for V3
    Av_tie = 2 * math.pi * tie_d ** 2 / 4  # two legs per tie set
    Pu_avg = df[df['P_Demand_kN'] > 0]['P_Demand_kN'].mean() if df['P_Demand_kN'].gt(0).any() else 0

    sv2 = column_shear_capacity(h, d2, fc, Pu_avg, Ag, fyt_shear, Av_tie, tie_spacing)
    sv3 = column_shear_capacity(b, d3, fc, Pu_avg, Ag, fyt_shear, Av_tie, tie_spacing)
    V2_max = df['V2'].abs().max()
    V3_max = df['V3'].abs().max()
    shear_info = {
        'V2_max': round(V2_max, 1), 'phi_Vn2': sv2['phi_Vn_kN'],
        'V3_max': round(V3_max, 1), 'phi_Vn3': sv3['phi_Vn_kN'],
        'pass2': V2_max <= sv2['phi_Vn_kN'],
        'pass3': V3_max <= sv3['phi_Vn_kN'],
    }

# ============================================================
# DISPLAY RESULTS
# ============================================================

st.markdown("---")

# -- Key metric row --
c1, c2, c3, c4 = st.columns(4)
c1.metric("Max PMM ratio", max_ratio, help="PCA biaxial load contour — must be ≤ 1.0")
c2.metric("ρg", f"{rho_pct} %", help="ACI §10.6.1.1: 1 – 8 %")
c3.metric("klu/r (axis 2)", round(klu_r_2, 1))
c4.metric("klu/r (axis 3)", round(klu_r_3, 1))

# -- Overall status banner --
if max_ratio > 1.0:
    st.error(f"### 🚨 FAIL — PMM Ratio: {max_ratio}  |  Combo: {max_combo}")
elif max_ratio > 0.90:
    st.warning(f"### ⚠️ MARGINAL — PMM Ratio: {max_ratio}  |  Combo: {max_combo}")
else:
    st.success(f"### ✅ PASS — PMM Ratio: {max_ratio}  |  Combo: {max_combo}")

st.markdown(
    f"**Layout:** {layout_text} &nbsp;|&nbsp; **Ties:** {tie_text} &nbsp;|&nbsp; "
    f"**ρg:** {rho_pct} %"
)
st.info(
    "**Biaxial method:** PCA Load Contour (Bresler-Parme) — "
    "$(DC_2)^\\alpha + (DC_3)^\\alpha = 1$  with dynamic $\\alpha$ from 1.15 (low axial) "
    "to 1.50 (high axial). D/C ratios use radial interpolation on the PM curve.",
    icon="ℹ️"
)

# -- Slenderness flags --
for axis, klu_r in [("Axis 2", klu_r_2), ("Axis 3", klu_r_3)]:
    if klu_r > 100:
        st.error(f"klu/r = {round(klu_r, 1)} ({axis}) — extremely slender. Consider second-order analysis.")
    elif klu_r > 34:
        st.warning(f"klu/r = {round(klu_r, 1)} ({axis}) — slender; moment magnification applied.")

max_delta = max(df['Delta_2'].max(), df['Delta_3'].max())
if max_delta >= 990:
    st.error("🚨 **BUCKLING FAILURE:** Pu ≥ 0.75·Pc. Increase section or reduce unsupported length.")
elif max_delta > 1.4:
    st.warning(f"⚠️ Max moment magnifier δ = {round(max_delta, 2)} > 1.4 — large slenderness effect.")

# -- Shear results --
if shear_info:
    st.subheader("Column Shear Check (ACI §22.5.6.1)")
    sc1, sc2 = st.columns(2)
    icon2 = "✅" if shear_info['pass2'] else "❌"
    icon3 = "✅" if shear_info['pass3'] else "❌"
    sc1.metric(f"{icon2} φVn (axis 2)", f"{shear_info['phi_Vn2']} kN",
               f"Demand V2 = {shear_info['V2_max']} kN")
    sc2.metric(f"{icon3} φVn (axis 3)", f"{shear_info['phi_Vn3']} kN",
               f"Demand V3 = {shear_info['V3_max']} kN")
    if not shear_info['pass2'] or not shear_info['pass3']:
        st.error("Shear demand exceeds capacity — increase tie size, reduce spacing, or enlarge section.")

# -- Interaction diagram --
st.subheader("P-M Interaction Diagram")
st.caption(
    "The diagram shows one uniaxial PM curve (governing axis). "
    "Points inside the curve may still fail the **biaxial** check — "
    "refer to the PMM column in the table below."
)

fig, ax = plt.subplots(figsize=(8, 6))
gov_pm = pm2 if df['DC_2'].max() > df['DC_3'].max() else pm3
gov_label = "Axis-2 (M2 governs)" if df['DC_2'].max() > df['DC_3'].max() else "Axis-3 (M3 governs)"

# Sort curve for a clean plot
gov_sorted = gov_pm.sort_values('Axial_kN').drop_duplicates()
ax.plot(gov_sorted['Moment_kNm'], gov_sorted['Axial_kN'],
        color='steelblue', linewidth=2, label=f'φPn-φMn ({gov_label})')

ax.scatter(df['M2_Mag'], df['P_Demand_kN'],
           c=df['DC_2'], cmap='RdYlGn_r', vmin=0, vmax=1.2,
           s=60, marker='s', zorder=5, label='M2 demands (colour = DC_2)')
ax.scatter(df['M3_Mag'], df['P_Demand_kN'],
           c=df['DC_3'], cmap='RdYlGn_r', vmin=0, vmax=1.2,
           s=60, marker='^', zorder=5, label='M3 demands (colour = DC_3)')

ax.axhline(0, color='black', linewidth=0.8)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel("φMn (kNm)")
ax.set_ylabel("φPn (kN)")
ax.set_title(f"Frame {frame_id}  —  {layout_text}")
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(fontsize=9)
plt.colorbar(plt.cm.ScalarMappable(
    cmap='RdYlGn_r',
    norm=plt.Normalize(0, 1.2)), ax=ax, label='DC ratio')
plt.tight_layout()
st.pyplot(fig)

# -- Results table --
st.subheader("Load-combination table")
disp_cols = ['Load_Combo', 'P_Demand_kN',
             'M2_Mag', 'Delta_2', 'DC_2',
             'M3_Mag', 'Delta_3', 'DC_3',
             'Alpha', 'PMM']

def _hl_delta(v):
    if isinstance(v, (int, float)):
        if v >= 990: return 'background-color:#8B0000;color:white'
        if v > 1.4:  return 'background-color:#ffcccc;color:#c00'
    return ''

def _hl_pmm(v):
    if isinstance(v, (int, float)):
        if v > 1.0: return 'background-color:#8B0000;color:white;font-weight:bold'
        if v > 0.9: return 'background-color:#ffcccc;color:#c00'
    return ''

styled = (df[disp_cols]
          .style
          .map(_hl_delta, subset=['Delta_2', 'Delta_3'])
          .map(_hl_pmm,   subset=['PMM'])
          .format({'P_Demand_kN': '{:.1f}', 'M2_Mag': '{:.1f}', 'M3_Mag': '{:.1f}',
                   'Delta_2': '{:.3f}', 'Delta_3': '{:.3f}',
                   'DC_2': '{:.3f}', 'DC_3': '{:.3f}',
                   'Alpha': '{:.2f}', 'PMM': '{:.3f}'}))
st.dataframe(styled, use_container_width=True)

# -- PDF export --
st.markdown("---")
st.subheader("📄 Export report")
pdf_bytes = create_pdf(
    frame_id, b, h, fc, fy,
    layout_text, tie_text, rho_pct,
    max_ratio, max_combo,
    klu_r_2, klu_r_3, fig
)
st.download_button(
    "📥 Download PDF",
    data=pdf_bytes,
    file_name=f"Column_Design_{frame_id}.pdf",
    mime="application/pdf"
)
