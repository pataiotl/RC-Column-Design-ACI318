import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np


# ==========================================
# 1. THE ENGINEERING ENGINES
# ==========================================

def generate_pm_curve(b, h, fc, fy, ast):
    """Generates an ACI 318 compliant DESIGN P-M interaction curve (phi*Pn, phi*Mn)."""
    cover = 50
    d = h - cover
    d_prime = cover
    as_top = ast / 2
    as_bot = ast / 2
    Es = 200000
    ecu = 0.003
    eps_y = fy / Es

    beta1 = 0.85 if fc <= 28 else max(0.65, 0.85 - 0.05 * ((fc - 28) / 7))
    Ag = b * h
    Po = 0.85 * fc * (Ag - ast) + fy * ast
    Pn_max = 0.80 * Po

    curve_points = []
    c = h * 2.0

    while c > 5:
        a = min(beta1 * c, h)
        Cc = 0.85 * fc * a * b
        eps_top = ecu * (c - d_prime) / c
        eps_bot = ecu * (d - c) / c
        fs_top = min(fy, max(-fy, eps_top * Es))
        fs_bot = min(fy, max(-fy, -eps_bot * Es))
        Cs = as_top * fs_top
        Ts = as_bot * fs_bot
        Pn_newtons = Cc + Cs + Ts

        Mc = Cc * (h / 2 - a / 2)
        Ms_top = Cs * (h / 2 - d_prime)
        Ms_bot = abs(Ts) * (d - h / 2)
        Mn_Nmm = Mc + Ms_top + Ms_bot

        eps_t = eps_bot
        if eps_t <= eps_y:
            phi = 0.65
        elif eps_t >= (eps_y + 0.003):
            phi = 0.90
        else:
            phi = 0.65 + 0.25 * ((eps_t - eps_y) / 0.003)

        design_Pn = min(phi * Pn_newtons, 0.65 * Pn_max) / 1000
        design_Mn = (phi * Mn_Nmm) / 1000000

        if design_Pn > - (ast * fy * 0.9) / 1000:
            curve_points.append({'Moment_kNm': round(design_Mn, 1), 'Axial_kN': round(design_Pn, 1)})
        c -= 5

    return pd.DataFrame(curve_points)


def calculate_magnified_moment(Pu, M_demand, width, depth, fc, lu_mm, k, cm_factor, beta_dns):
    """Calculates the magnified moment (Mc) and Delta factor."""
    if Pu <= 0: return pd.Series([abs(M_demand), 1.0])
    M_min = Pu * (15 + 0.03 * depth) / 1000
    if M_demand == 0: M_demand = 0.001

    Ec = 4700 * math.sqrt(fc)
    Ig = (width * (depth ** 3)) / 12
    EI = (0.4 * Ec * Ig) / (1 + beta_dns)
    Pc = ((math.pi ** 2 * EI) / (k * lu_mm) ** 2) / 1000

    if Pu >= 0.75 * Pc: return pd.Series([9999.9, 999.9])

    delta = cm_factor / (1 - (Pu / (0.75 * Pc)))
    delta = max(delta, 1.0)
    Mc = delta * abs(M_demand)
    return pd.Series([round(max(Mc, M_min), 1), round(delta, 3)])


def check_capacity(pm_data, P_demand, M_demand):
    """Computationally checks if a specific P, M point is inside the curve."""
    pm_asc = pm_data.sort_values(by='Axial_kN', ascending=True)
    if P_demand > pm_asc['Axial_kN'].max() or P_demand < pm_asc['Axial_kN'].min():
        return False
    M_cap = np.interp(P_demand, pm_asc['Axial_kN'], pm_asc['Moment_kNm'])
    return abs(M_demand) <= M_cap


def get_dc_ratio(pm_data, P_demand, M_demand):
    """Calculates the true Demand/Capacity (PMM) Ratio for a given axis."""
    pm_asc = pm_data.sort_values(by='Axial_kN', ascending=True)
    P_max = pm_asc['Axial_kN'].max()
    P_min = pm_asc['Axial_kN'].min()

    if P_demand > P_max: return round(P_demand / P_max, 3)  # Fails in pure axial
    if P_demand < P_min: return 9.99  # Fails in pure tension

    M_cap = np.interp(P_demand, pm_asc['Axial_kN'], pm_asc['Moment_kNm'])
    if M_cap <= 0: return 9.99
    return round(abs(M_demand) / M_cap, 3)


def run_optimizer(df, b, h, fc, fy, lu, k, cm, beta_dns):
    """Iterates through standard DB rebar to find the cheapest layout that passes."""
    Ag = b * h
    min_ast = 0.01 * Ag
    max_ast = 0.08 * Ag
    bars = {'DB20': 314.16, 'DB25': 490.87, 'DB28': 615.75, 'DB32': 804.25}

    configs = []
    for name, area in bars.items():
        for count in range(4, 42, 2):
            total_area = count * area
            if min_ast <= total_area <= max_ast:
                configs.append({'Name': f"{count}-{name}", 'Ast': total_area})

    configs = sorted(configs, key=lambda x: x['Ast'])

    for config in configs:
        pm_test = generate_pm_curve(b, h, fc, fy, config['Ast'])
        all_pass = True
        for index, row in df.iterrows():
            P = row['P_Demand_kN']
            m2_res = calculate_magnified_moment(P, row['M2_Demand_kNm'], h, b, fc, lu, k, cm, beta_dns)
            m3_res = calculate_magnified_moment(P, row['M3_Demand_kNm'], b, h, fc, lu, k, cm, beta_dns)
            if not check_capacity(pm_test, P, m2_res.iloc[0]) or not check_capacity(pm_test, P, m3_res.iloc[0]):
                all_pass = False
                break
        if all_pass: return config
    return None


# ==========================================
# 2. THE WEB INTERFACE
# ==========================================

st.title("🏗️ Advanced RC Column App")
st.write("Upload raw ETABS loads. Toggle the Auto-Optimizer to find the most efficient rebar layout.")

st.sidebar.header("1. Section Properties")
b = st.sidebar.number_input("Width (b, Axis 2) in mm", value=800, step=50)
h = st.sidebar.number_input("Depth (h, Axis 3) in mm", value=800, step=50)
fc = st.sidebar.number_input("f'c (MPa)", value=40, step=5)
fy = st.sidebar.number_input("fy (MPa)", value=500, step=10)

auto_optimize = st.sidebar.checkbox("🚀 Enable Auto-Design Optimizer")
optimized_name = ""

uploaded_file = st.file_uploader("Upload RAW ETABS loads (CSV)", type=["csv"])

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)

    st.sidebar.header("2. Filter ETABS Data")
    if 'Story' in df_raw.columns and 'Column' in df_raw.columns:
        selected_story = st.sidebar.selectbox("Select Story", df_raw['Story'].unique())
        available_columns = df_raw[df_raw['Story'] == selected_story]['Column'].unique()
        selected_column = st.sidebar.selectbox("Select Column Pier", available_columns)

        df = df_raw[(df_raw['Story'] == selected_story) & (df_raw['Column'] == selected_column)].copy()
        df['Load_Combo'] = df['Output Case']
        df['P_Demand_kN'] = df['P'] * -1
        df['M2_Demand_kNm'] = df['M2']
        df['M3_Demand_kNm'] = df['M3']
        st.write(f"### Designing: {selected_column} at {selected_story}")
    else:
        df = df_raw.copy()
        st.warning("Standard ETABS 'Story' or 'Column' headers not found. Using raw table.")

    st.sidebar.header("3. Slenderness & Creep")
    lu = st.sidebar.number_input("Unsupported Length (lu) in mm", value=3000, step=100)
    k_factor = st.sidebar.number_input("Effective Length Factor (k)", value=1.0, step=0.1)
    cm_val = st.sidebar.slider("Cm Factor", min_value=0.2, max_value=1.0, value=1.0, step=0.05)
    beta_dns = st.sidebar.slider("Beta_dns", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

    # --- Rebar Optimization ---
    if auto_optimize:
        with st.spinner("Optimizing rebar layout..."):
            best_layout = run_optimizer(df, b, h, fc, fy, lu, k_factor, cm_val, beta_dns)
        if best_layout:
            ast = best_layout['Ast']
            optimized_name = best_layout['Name']
        else:
            ast = b * h * 0.08
            st.error("❌ **Optimization Failed:** Section fails even at 8% limit.")
    else:
        ast = st.sidebar.number_input("Rebar Area (Ast) mm^2", value=5890, step=100)

    # --- Run Slenderness Engine ---
    df[['M2_Mag_kNm', 'Delta_2']] = df.apply(lambda row: calculate_magnified_moment(
        row['P_Demand_kN'], row['M2_Demand_kNm'], width=h, depth=b, fc=fc, lu_mm=lu, k=k_factor, cm_factor=cm_val,
        beta_dns=beta_dns
    ), axis=1)

    df[['M3_Mag_kNm', 'Delta_3']] = df.apply(lambda row: calculate_magnified_moment(
        row['P_Demand_kN'], row['M3_Demand_kNm'], width=b, depth=h, fc=fc, lu_mm=lu, k=k_factor, cm_factor=cm_val,
        beta_dns=beta_dns
    ), axis=1)

    # --- Run PMM Ratio Engine ---
    pm_data = generate_pm_curve(b, h, fc, fy, ast)
    df['DC_2'] = df.apply(lambda row: get_dc_ratio(pm_data, row['P_Demand_kN'], row['M2_Mag_kNm']), axis=1)
    df['DC_3'] = df.apply(lambda row: get_dc_ratio(pm_data, row['P_Demand_kN'], row['M3_Mag_kNm']), axis=1)
    df['PMM_Ratio'] = df[['DC_2', 'DC_3']].max(axis=1)  # Governing ratio

    # --- TOP DASHBOARD BANNER ---
    max_idx = df['PMM_Ratio'].idxmax()
    max_ratio = df.loc[max_idx, 'PMM_Ratio']
    max_combo = df.loc[max_idx, 'Load_Combo']
    layout_text = f"Layout: {optimized_name}" if optimized_name else f"Ast: {ast} mm²"

    st.markdown("---")
    if max_ratio > 1.0:
        st.error(f"### 🚨 MAX PMM RATIO: {max_ratio} \n**Governing Combo:** `{max_combo}` | **{layout_text}**")
    elif max_ratio > 0.9:
        st.warning(f"### ⚠️ MAX PMM RATIO: {max_ratio} \n**Governing Combo:** `{max_combo}` | **{layout_text}**")
    else:
        st.success(f"### ✅ MAX PMM RATIO: {max_ratio} \n**Governing Combo:** `{max_combo}` | **{layout_text}**")
    st.markdown("---")

    # --- Global Slenderness Warnings ---
    max_delta = max(df['Delta_2'].max(), df['Delta_3'].max())
    if max_delta >= 990:
        st.error("🚨 **CRITICAL FAILURE:** Column will buckle!")
    elif max_delta > 1.4:
        st.warning("⚠️ **SLENDERNESS WARNING:** Moment magnifier (δ) > 1.4.")

    # --- Draw Interaction Diagram ---
    st.subheader("Interaction Diagram: PMM")
    fig, ax = plt.subplots(figsize=(8, 6))
    curve_label = f'Design Capacity ({optimized_name})' if auto_optimize and optimized_name else 'Design Capacity (\u03c6Pn, \u03c6Mn)'

    ax.plot(pm_data['Moment_kNm'], pm_data['Axial_kN'], label=curve_label, color='blue', linewidth=2)
    ax.scatter(df['M2_Mag_kNm'], df['P_Demand_kN'], color='mediumseagreen', label='M2 Demands', zorder=5, s=50,
               marker='s')
    ax.scatter(df['M3_Mag_kNm'], df['P_Demand_kN'], color='darkorange', label='M3 Demands', zorder=5, s=50, marker='^')
    ax.set_title(f"Column {b}x{h} mm")
    ax.set_xlabel("Bending Moment (kNm)")
    ax.set_ylabel("Axial Load (kN)")
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    st.pyplot(fig)


    # --- Draw Output Table ---
    def highlight_table(val):
        if isinstance(val, (int, float)):
            if val >= 990:
                return 'background-color: darkred; color: white;'
            elif val > 1.4:
                return 'background-color: #ffcccc; color: red;'
        return ''


    def highlight_pmm(val):
        if isinstance(val, (int, float)):
            if val > 1.0:
                return 'background-color: darkred; color: white; font-weight: bold;'
            elif val > 0.9:
                return 'background-color: #ffcccc; color: red;'
        return ''


    display_columns = ['Load_Combo', 'P_Demand_kN', 'M2_Mag_kNm', 'Delta_2', 'M3_Mag_kNm', 'Delta_3', 'PMM_Ratio']
    styled_df = df[display_columns].style.map(highlight_table, subset=['Delta_2', 'Delta_3'])
    styled_df = styled_df.map(highlight_pmm, subset=['PMM_Ratio'])

    st.dataframe(styled_df, use_container_width=True)
