import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from fpdf import FPDF
import tempfile


# ==========================================
# 1. THE ENGINEERING ENGINES
# ==========================================

def create_pdf(frame_name, b, h, fc, fy, layout_text, max_ratio, max_combo, fig):
    """Generates a formatted A4 PDF calculation report."""
    import os  # Ensure os is imported for cleanup

    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"RC Column Design Report - Frame: {frame_name}", ln=True, align='C')
    pdf.ln(5)

    # Section Properties
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "1. Section Properties", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 6, f"Dimensions: {b} mm x {h} mm", ln=True)
    pdf.cell(0, 6, f"Concrete Compressive Strength (f'c): {fc} MPa", ln=True)
    pdf.cell(0, 6, f"Steel Yield Strength (fy): {fy} MPa", ln=True)
    pdf.ln(5)

    # Rebar Layout
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "2. Reinforcement Layout", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 6, f"Design Layout: {layout_text}", ln=True)
    pdf.ln(5)

    # Results Summary
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "3. Design Summary", ln=True)
    pdf.set_font("Arial", '', 11)
    if max_ratio > 1.0:
        pdf.set_text_color(220, 53, 69)  # Red text for failure
        status = "FAIL"
    else:
        pdf.set_text_color(40, 167, 69)  # Green text for pass
        status = "PASS"

    pdf.cell(0, 6, f"Status: {status}", ln=True)
    pdf.cell(0, 6, f"Maximum PMM Ratio: {max_ratio}", ln=True)
    pdf.cell(0, 6, f"Governing Load Combination: {max_combo}", ln=True)
    pdf.set_text_color(0, 0, 0)  # Reset to black
    pdf.ln(5)

    # Interaction Diagram
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "4. PMM Interaction Diagram", ln=True)

    # Save the matplotlib plot to a temporary file and insert it into the PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        fig.savefig(tmpfile.name, format="png", bbox_inches="tight", dpi=300)
        pdf.image(tmpfile.name, x=15, w=180)

    # Safely extract PDF as bytes using a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        pdf.output(tmp_pdf.name)
        with open(tmp_pdf.name, "rb") as f:
            pdf_bytes = f.read()

    # Clean up both temporary files to prevent server memory leaks
    try:
        os.remove(tmpfile.name)
        os.remove(tmp_pdf.name)
    except Exception:
        pass

    return pdf_bytes


def generate_pm_curve(width, depth, fc, fy, cover_mm, n_width, n_depth, bar_dia):
    """Generates an ACI 318 DESIGN P-M curve using a true 4-face discrete rebar array."""
    tie_dia = 10
    d_prime = cover_mm + tie_dia + (bar_dia / 2)
    bar_area = math.pi * (bar_dia ** 2) / 4

    layers = []
    layers.append({'area': n_width * bar_area, 'd': d_prime})
    if n_depth > 2:
        spacing = (depth - 2 * d_prime) / (n_depth - 1)
        for i in range(1, n_depth - 1):
            layers.append({'area': 2 * bar_area, 'd': d_prime + i * spacing})
    layers.append({'area': n_width * bar_area, 'd': depth - d_prime})

    total_ast = sum(layer['area'] for layer in layers)
    Es = 200000
    ecu = 0.003
    eps_y = fy / Es
    beta1 = 0.85 if fc <= 28 else max(0.65, 0.85 - 0.05 * ((fc - 28) / 7))
    Ag = width * depth
    Po = 0.85 * fc * (Ag - total_ast) + fy * total_ast
    Pn_max = 0.80 * Po

    curve_points = []
    c = depth * 2.0

    while c > 5:
        a = min(beta1 * c, depth)
        Cc = 0.85 * fc * a * width
        Mc_concrete = Cc * (depth / 2 - a / 2)
        Fs_total = Ms_total = 0

        for layer in layers:
            eps_s = ecu * (c - layer['d']) / c
            f_s = min(fy, max(-fy, eps_s * Es))
            force = layer['area'] * f_s
            if layer['d'] <= a and eps_s > 0:
                force -= layer['area'] * 0.85 * fc
            Fs_total += force
            Ms_total += force * (depth / 2 - layer['d'])

        Pn_newtons = Cc + Fs_total
        Mn_Nmm = Mc_concrete + Ms_total

        eps_t = ecu * (c - layers[-1]['d']) / c
        if eps_t <= eps_y:
            phi = 0.65
        elif eps_t >= (eps_y + 0.003):
            phi = 0.90
        else:
            phi = 0.65 + 0.25 * ((eps_t - eps_y) / 0.003)

        design_Pn = min(phi * Pn_newtons, 0.65 * Pn_max) / 1000
        design_Mn = (phi * Mn_Nmm) / 1000000

        if design_Pn > - (total_ast * fy * 0.9) / 1000:
            curve_points.append({'Moment_kNm': round(design_Mn, 1), 'Axial_kN': round(design_Pn, 1)})
        c -= 5

    return pd.DataFrame(curve_points)


def calculate_magnified_moment(Pu, M_demand, width, depth, fc, lu_mm, k, cm_factor, beta_dns):
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


def get_dc_ratio(pm_data, P_demand, M_demand):
    pm_asc = pm_data.sort_values(by='Axial_kN', ascending=True)
    P_max = pm_asc['Axial_kN'].max()
    P_min = pm_asc['Axial_kN'].min()
    if P_demand > P_max: return round(P_demand / P_max, 3)
    if P_demand < P_min: return 9.99
    M_cap = np.interp(P_demand, pm_asc['Axial_kN'], pm_asc['Moment_kNm'])
    if M_cap <= 0: return 9.99
    return round(abs(M_demand) / M_cap, 3)


def run_optimizer(df, b, h, fc, fy, cover, lu, k, cm, beta_dns):
    Ag = b * h
    min_ast = 0.01 * Ag
    max_ast = 0.08 * Ag
    bars = {'DB16': 16, 'DB20': 20, 'DB25': 25, 'DB28': 28, 'DB32': 32}

    configs = []
    for name, dia in bars.items():
        area = math.pi * (dia ** 2) / 4
        for nw in range(3, 13):
            for nd in range(3, 13):
                total_bars = 2 * nw + 2 * (nd - 2)
                ast = total_bars * area
                if min_ast <= ast <= max_ast:
                    configs.append({
                        'Name': f"{total_bars}-{name} ({nw}x{nd} Layout)",
                        'Ast': ast, 'nw': nw, 'nd': nd, 'dia': dia
                    })

    configs = sorted(configs, key=lambda x: x['Ast'])

    for config in configs:
        pm_test_2 = generate_pm_curve(h, b, fc, fy, cover, config['nd'], config['nw'], config['dia'])
        pm_test_3 = generate_pm_curve(b, h, fc, fy, cover, config['nw'], config['nd'], config['dia'])

        all_pass = True
        for index, row in df.iterrows():
            P = row['P_Demand_kN']
            m2_res = calculate_magnified_moment(P, row['M2_Demand_kNm'], h, b, fc, lu, k, cm, beta_dns)
            m3_res = calculate_magnified_moment(P, row['M3_Demand_kNm'], b, h, fc, lu, k, cm, beta_dns)

            if get_dc_ratio(pm_test_2, P, m2_res.iloc[0]) > 1.0 or get_dc_ratio(pm_test_3, P, m3_res.iloc[0]) > 1.0:
                all_pass = False
                break
        if all_pass: return config
    return None


# ==========================================
# 2. THE WEB INTERFACE
# ==========================================

st.title("🏗️ Advanced RC Column App")
st.write("Upload raw SAP2000 loads. Discrete 4-face rebar analysis is active.")

st.sidebar.header("1. Section Properties")
b = st.sidebar.number_input("Width (b, Axis 2) in mm", value=800, step=50)
h = st.sidebar.number_input("Depth (h, Axis 3) in mm", value=800, step=50)
fc = st.sidebar.number_input("f'c (MPa)", value=40, step=5)
fy = st.sidebar.number_input("fy (MPa)", value=500, step=10)

st.sidebar.header("2. Longitudinal Bars")
auto_optimize = st.sidebar.checkbox("🚀 Enable Auto-Design Optimizer")
optimized_name = ""

if not auto_optimize:
    cover_mm = st.sidebar.number_input("Clear Cover for Confinement Bars (mm)", value=40, step=5)
    n3_bars = st.sidebar.number_input("Number of Longit Bars Along 3-dir Face (Width)", value=4, step=1)
    n2_bars = st.sidebar.number_input("Number of Longit Bars Along 2-dir Face (Depth)", value=4, step=1)

    bar_options = {'16d': 16, '20d': 20, '25d': 25, '28d': 28, '32d': 32}
    selected_bar = st.sidebar.selectbox("Longitudinal Bar Size", list(bar_options.keys()), index=2)
    bar_dia = bar_options[selected_bar]

    total_bars = 2 * n3_bars + 2 * (n2_bars - 2)
    total_ast = total_bars * (math.pi * bar_dia ** 2 / 4)
    st.sidebar.markdown(f"**Total Area Provided:** {round(total_ast, 1)} mm² ({total_bars} bars)")
else:
    cover_mm = st.sidebar.number_input("Clear Cover for Confinement Bars (mm)", value=40, step=5)

uploaded_file = st.file_uploader("Upload RAW SAP2000 loads (CSV)", type=["csv"])

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)

    if 'Frame' in df_raw.columns and (
            str(df_raw['Frame'].iloc[0]).strip().lower() == 'text' or str(df_raw['P'].iloc[0]).strip().lower() == 'kn'):
        df_raw = df_raw.drop(0).reset_index(drop=True)
    for col in ['P', 'M2', 'M3']:
        if col in df_raw.columns: df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

    st.sidebar.header("3. Filter SAP2000 Data")
    if 'Frame' in df_raw.columns:
        selected_frame = st.sidebar.selectbox("Select Frame (Column)", df_raw['Frame'].unique())
        df = df_raw[df_raw['Frame'] == selected_frame].copy()
        df['Load_Combo'] = df['OutputCase']
        df['P_Demand_kN'] = df['P'] * -1
        df['M2_Demand_kNm'] = df['M2']
        df['M3_Demand_kNm'] = df['M3']
    else:
        df = df_raw.copy()
        selected_frame = "Custom"

    st.sidebar.header("4. Slenderness & Creep")
    lu = st.sidebar.number_input("Unsupported Length (lu) in mm", value=3000, step=100)
    k_factor = st.sidebar.number_input("Effective Length Factor (k)", value=1.0, step=0.1)
    cm_val = st.sidebar.slider("Cm Factor", min_value=0.2, max_value=1.0, value=1.0, step=0.05)
    beta_dns = st.sidebar.slider("Beta_dns", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

    if auto_optimize:
        with st.spinner("Optimizing precise rebar grid..."):
            best_layout = run_optimizer(df, b, h, fc, fy, cover_mm, lu, k_factor, cm_val, beta_dns)
        if best_layout:
            n3_bars, n2_bars, bar_dia = best_layout['nw'], best_layout['nd'], best_layout['dia']
            optimized_name = best_layout['Name']
        else:
            st.error("❌ **Optimization Failed:** Section fails even at 8% limit.")
            n3_bars, n2_bars, bar_dia = 10, 10, 32

    df[['M2_Mag_kNm', 'Delta_2']] = df.apply(lambda row: calculate_magnified_moment(
        row['P_Demand_kN'], row['M2_Demand_kNm'], h, b, fc, lu, k_factor, cm_val, beta_dns), axis=1)

    df[['M3_Mag_kNm', 'Delta_3']] = df.apply(lambda row: calculate_magnified_moment(
        row['P_Demand_kN'], row['M3_Demand_kNm'], b, h, fc, lu, k_factor, cm_val, beta_dns), axis=1)

    pm_data_2 = generate_pm_curve(h, b, fc, fy, cover_mm, n2_bars, n3_bars, bar_dia)
    pm_data_3 = generate_pm_curve(b, h, fc, fy, cover_mm, n3_bars, n2_bars, bar_dia)

    df['DC_2'] = df.apply(lambda row: get_dc_ratio(pm_data_2, row['P_Demand_kN'], row['M2_Mag_kNm']), axis=1)
    df['DC_3'] = df.apply(lambda row: get_dc_ratio(pm_data_3, row['P_Demand_kN'], row['M3_Mag_kNm']), axis=1)
    df['PMM_Ratio'] = df[['DC_2', 'DC_3']].max(axis=1)

    max_idx = df['PMM_Ratio'].idxmax()
    max_ratio = df.loc[max_idx, 'PMM_Ratio']
    max_combo = df.loc[max_idx, 'Load_Combo']
    layout_text = f"Layout: {optimized_name}" if optimized_name else f"{total_bars}-{selected_bar} ({n3_bars}x{n2_bars})"

    st.markdown("---")
    if max_ratio > 1.0:
        st.error(f"### 🚨 MAX PMM RATIO: {max_ratio} \n**Governing Combo:** `{max_combo}` | **{layout_text}**")
    elif max_ratio > 0.9:
        st.warning(f"### ⚠️ MAX PMM RATIO: {max_ratio} \n**Governing Combo:** `{max_combo}` | **{layout_text}**")
    else:
        st.success(f"### ✅ MAX PMM RATIO: {max_ratio} \n**Governing Combo:** `{max_combo}` | **{layout_text}**")
    st.markdown("---")

    max_delta = max(df['Delta_2'].max(), df['Delta_3'].max())
    if max_delta >= 990:
        st.error("🚨 **CRITICAL FAILURE:** Column will buckle!")
    elif max_delta > 1.4:
        st.warning("⚠️ **SLENDERNESS WARNING:** Moment magnifier (δ) > 1.4.")

    st.subheader("Interaction Diagram: PMM")
    fig, ax = plt.subplots(figsize=(8, 6))
    curve_label = f'Design Capacity ({layout_text})'

    governing_pm = pm_data_2 if df['DC_2'].max() > df['DC_3'].max() else pm_data_3

    ax.plot(governing_pm['Moment_kNm'], governing_pm['Axial_kN'], label=curve_label, color='blue', linewidth=2)
    ax.scatter(df['M2_Mag_kNm'], df['P_Demand_kN'], color='mediumseagreen', label='M2 Demands', zorder=5, s=50,
               marker='s')
    ax.scatter(df['M3_Mag_kNm'], df['P_Demand_kN'], color='darkorange', label='M3 Demands', zorder=5, s=50, marker='^')
    ax.set_title(f"Frame {b}x{h} mm")
    ax.set_xlabel("Bending Moment (kNm)")
    ax.set_ylabel("Axial Load (kN)")
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    st.pyplot(fig)


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


    display_columns = ['Load_Combo', 'P_Demand_kN', 'M2_Mag_kNm', 'Delta_2', 'DC_2', 'M3_Mag_kNm', 'Delta_3', 'DC_3',
                       'PMM_Ratio']
    styled_df = df[display_columns].style.map(highlight_table, subset=['Delta_2', 'Delta_3'])
    styled_df = styled_df.map(highlight_pmm, subset=['PMM_Ratio'])

    st.dataframe(styled_df, use_container_width=True)

    # --- PDF EXPORT FEATURE ---
    st.markdown("---")
    st.subheader("📄 Export Calculation Report")

    # Run the PDF generator engine
    pdf_bytes = create_pdf(selected_frame, b, h, fc, fy, layout_text, max_ratio, max_combo, fig)

    # Streamlit Download Button
    st.download_button(
        label="📥 Download PDF Report",
        data=pdf_bytes,
        file_name=f"Design_Report_Frame_{selected_frame}.pdf",
        mime="application/pdf"
    )
