# km_viewer.py
import io
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# -------------------------------------------------
# App chrome
# -------------------------------------------------
st.set_page_config(page_title="Novak's TriNetX KM Viewer", layout="wide")
st.title("Novak's TriNetX Kaplan–Meier Survival Curve Viewer")
st.markdown(
    "Upload the TriNetX Kaplan–Meier CSV. Customize the plot, optionally attach Number-at-Risk, "
    "and download publication-ready figures (PNG + SVG)."
)

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def find_header_row(lines, keywords=("Time", "Survival")):
    for i, line in enumerate(lines[:200]):
        low = line.lower()
        if all(k.lower() in low for k in keywords):
            return i
    return 0

def soft_col(df, patterns):
    for pat in patterns:
        for c in df.columns:
            if re.search(pat, c, flags=re.I):
                return c
    return None

def forward_fill(df, cols):
    present = [c for c in cols if c in df.columns]
    if present:
        df[present] = df[present].ffill()
    return df

def trim_tail(df, surv_cols):
    mask = np.zeros(len(df), dtype=bool)
    for c in surv_cols:
        if c and c in df.columns:
            mask |= df[c].notna().values
    if not mask.any():
        return df
    last_idx = np.where(mask)[0].max()
    return df.iloc[: last_idx + 1].copy()

def step_plot(ax, x, y, **kwargs):
    ax.plot(x, y, drawstyle="steps-post", **kwargs)

def median_survival(time, surv):
    if surv.isna().all():
        return None
    s = surv.astype(float)
    idx = np.where(s <= 0.5)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    return float(time.iloc[i]), float(s.iloc[i])

def export_figure(fig, base="kaplan_meier_curve", dpi=300):
    png = io.BytesIO()
    fig.savefig(png, format="png", dpi=dpi, bbox_inches="tight")
    png.seek(0)
    svg = io.BytesIO()
    fig.savefig(svg, format="svg", bbox_inches="tight")
    svg.seek(0)
    return png, svg, f"{base}.png", f"{base}.svg"

def export_csv(df, base="km_plotted_subset"):
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf, f"{base}.csv"

# -------------------------------------------------
# Upload(s)
# -------------------------------------------------
km_file = st.file_uploader("Upload KM CSV (from TriNetX)", type=["csv"])
risk_file = st.file_uploader(
    "Optional: Upload Number-at-Risk CSV (must include columns for Cohort 1/2 risk at the same time grid)",
    type=["csv"],
    key="risk_uploader"
)

if not km_file:
    st.stop()

# Read and detect header
raw_lines = km_file.getvalue().decode("utf-8", errors="ignore").splitlines()
header_row = find_header_row(raw_lines)
df = pd.read_csv(io.StringIO("\n".join(raw_lines[header_row:])))
df.columns = df.columns.str.strip()

# Column mapping (robust to minor variants)
time_col = soft_col(df, [r"^time *(\(days\))?$", r"^time$"]) or ("Time (Days)" if "Time (Days)" in df.columns else None)
c1_surv = soft_col(df, [r"^cohort *1: *survival probability", r"^c.*1.*survival"])
c2_surv = soft_col(df, [r"^cohort *2: *survival probability", r"^c.*2.*survival"])
c1_lo   = soft_col(df, [r"cohort *1.*95.*lower"])
c1_hi   = soft_col(df, [r"cohort *1.*95.*upper"])
c2_lo   = soft_col(df, [r"cohort *2.*95.*lower"])
c2_hi   = soft_col(df, [r"cohort *2.*95.*upper"])

# Basic validation
if time_col is None or all(v is None for v in [c1_surv, c2_surv]):
    st.error("Couldn't find required columns. Need a Time column and at least one cohort survival column.")
    st.stop()

# Clean & trim
df = df.sort_values(time_col).reset_index(drop=True)
df = forward_fill(df, [c1_surv, c2_surv, c1_lo, c1_hi, c2_lo, c2_hi])
df = trim_tail(df, [c1_surv, c2_surv])

# Optional: parse Number-at-Risk from second CSV (if present)
risk_df = None
c1_risk = c2_risk = None
if risk_file:
    raw_risk = risk_file.getvalue().decode("utf-8", errors="ignore").splitlines()
    risk_header = find_header_row(raw_risk, keywords=("Time", "Risk"))
    risk_df = pd.read_csv(io.StringIO("\n".join(raw_risk[risk_header:])))
    risk_df.columns = risk_df.columns.str.strip()
    # map columns
    time_risk = soft_col(risk_df, [r"^time *(\(days\))?$", r"^time$"]) or ("Time (Days)" if "Time (Days)" in risk_df.columns else None)
    c1_risk = soft_col(risk_df, [r"cohort *1: *number at risk", r"c.*1.*risk"])
    c2_risk = soft_col(risk_df, [r"cohort *2: *number at risk", r"c.*2.*risk"])
    # merge on nearest time if exact grid differs
    if time_risk:
        # nearest merge
        base_times = df[time_col].to_numpy()
        risk_times = risk_df[time_risk].to_numpy()
        idx_map = np.searchsorted(base_times, risk_times, side="left")
        idx_map = np.clip(idx_map, 0, len(base_times) - 1)
        merged = pd.DataFrame({time_col: base_times})
        if c1_risk:
            arr = np.full_like(base_times, np.nan, dtype=float)
            arr[idx_map] = risk_df[c1_risk].to_numpy(dtype=float, copy=True)
            merged["C1_at_risk"] = arr
        if c2_risk:
            arr = np.full_like(base_times, np.nan, dtype=float)
            arr[idx_map] = risk_df[c2_risk].to_numpy(dtype=float, copy=True)
            merged["C2_at_risk"] = arr
        df = df.merge(merged, on=time_col, how="left")
        c1_risk = "C1_at_risk" if "C1_at_risk" in df.columns else None
        c2_risk = "C2_at_risk" if "C2_at_risk" in df.columns else None

# -------------------------------------------------
# Sidebar Controls
# -------------------------------------------------
st.sidebar.header("Customize Plot")

plot_title = st.sidebar.text_input("Plot Title", "Kaplan–Meier Survival Curve")
subtitle   = st.sidebar.text_input("Subtitle (optional)", "")

label1 = st.sidebar.text_input("Label for Cohort 1", "Cohort 1" if c1_surv else "")
label2 = st.sidebar.text_input("Label for Cohort 2", "Cohort 2" if c2_surv else "")

y_mode = st.sidebar.radio("Y-axis Mode", ["Survival Probability", "Cumulative Incidence (1 − S)"])
theme  = st.sidebar.selectbox("Theme", ["Color", "Black & White", "Journal"])
color1 = st.sidebar.color_picker("Cohort 1 Color", "#1f77b4")
color2 = st.sidebar.color_picker("Cohort 2 Color", "#ff7f0e")
line_width = st.sidebar.slider("Line Width", 1.0, 5.0, 2.0)

show_ci  = st.sidebar.checkbox("Show Confidence Intervals", True)
ci_alpha = st.sidebar.slider("CI Transparency", 0.0, 1.0, 0.15)

show_grid  = st.sidebar.checkbox("Show Grid", True)
fig_width  = st.sidebar.slider("Figure Width (inches)", 6, 16, 10)
fig_height = st.sidebar.slider("Figure Height (inches)", 5, 12, 7)

y_min = st.sidebar.slider("Y-axis Min", 0.0, 1.0, 0.0)
y_max = st.sidebar.slider("Y-axis Max", 0.0, 1.5, 1.05)
x_max = int(df[time_col].max())
max_days = st.sidebar.number_input("Maximum Days to Display", min_value=0, max_value=x_max, value=x_max, step=1)

landmarks  = st.sidebar.text_input("Landmark Lines (comma-separated days)", "")
show_median = st.sidebar.checkbox("Annotate Median Survival", True)
footnote   = st.sidebar.text_input("Footnote (e.g., data source)", "")

# Theme styles
if theme == "Black & White":
    c1_line, c2_line = "black", "gray"
    c1_style, c2_style = "-", "--"
elif theme == "Journal":
    c1_line, c2_line = color1, color2
    c1_style = c2_style = "-"
    line_width = max(line_width, 2.5)
else:
    c1_line, c2_line = color1, color2
    c1_style = c2_style = "-"

# Restrict data
dfp = df[df[time_col] <= max_days].copy()
if dfp.empty:
    st.warning("No rows within the selected maximum days. Adjust the control to include data.")
    st.stop()

# -------------------------------------------------
# Tabs
# -------------------------------------------------
tabs = ["Plot", "Data & Export"]
show_risk_tab = (c1_risk in dfp.columns) or (c2_risk in dfp.columns)
if show_risk_tab:
    tabs.insert(1, "Number-at-Risk")

tab_objs = st.tabs(tabs)

# --- Plot tab ---
with tab_objs[0]:
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    def plot_one(surv_col, lo_col, hi_col, label, line_color, line_style):
        if not surv_col or surv_col not in dfp.columns:
            return False
        y = dfp[surv_col].astype(float)
        if y_mode.startswith("Cumulative"):
            y = 1.0 - y
        step_plot(ax, dfp[time_col], y, label=label, color=line_color, linewidth=line_width, linestyle=line_style)
        if show_ci and lo_col in dfp.columns and hi_col in dfp.columns:
            lo = dfp[lo_col].astype(float)
            hi = dfp[hi_col].astype(float)
            if y_mode.startswith("Cumulative"):
                lo, hi = 1.0 - lo, 1.0 - hi
            ax.fill_between(dfp[time_col], lo, hi, alpha=ci_alpha, color=line_color)

        if show_median:
            ms = median_survival(dfp[time_col], dfp[surv_col])
            if ms:
                t_star, s_star = ms
                s_plot = (1.0 - s_star) if y_mode.startswith("Cumulative") else s_star
                ax.plot([t_star], [s_plot], marker="o", ms=6, color=line_color)
                ax.axvline(t_star, linestyle=":", linewidth=1)
                ax.axhline(s_plot, linestyle=":", linewidth=1)
                ax.annotate(f"Median ~ {t_star:.0f} d", (t_star, s_plot), xytext=(5, -10),
                            textcoords="offset points", fontsize=9)
        return True

    plotted = False
    if c1_surv: plotted |= plot_one(c1_surv, c1_lo, c1_hi, label1, c1_line, c1_style)
    if c2_surv: plotted |= plot_one(c2_surv, c2_lo, c2_hi, label2, c2_line, c2_style)
    if not plotted:
        st.error("No survival series could be plotted from the uploaded file.")
        st.stop()

    ax.set_title(plot_title, fontsize=18, loc="left")
    if subtitle:
        ax.set_title(subtitle, fontsize=12, loc="left", pad=26)
    ax.set_xlabel("Time (Days)")
    ax.set_ylabel("Cumulative Incidence" if y_mode.startswith("Cumulative") else "Survival Probability")
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(left=0, right=max_days)
    if show_grid:
        ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    # Landmarks
    if landmarks.strip():
        try:
            for t in [float(x.strip()) for x in landmarks.split(",") if x.strip()]:
                if 0 <= t <= max_days:
                    ax.axvline(t, linestyle="--", linewidth=1)
        except Exception:
            st.warning("Could not parse landmark lines. Use comma-separated numbers (e.g., 30, 90, 180).")

    # Footnote
    if footnote:
        ax.annotate(footnote, (0, 0), xytext=(0, -30), textcoords="offset points", fontsize=9, va="top")

    st.pyplot(fig, use_container_width=True)

    # Exports
    safe_title = re.sub(r"[^\w\-_. ]", "", plot_title).strip().replace(" ", "_") or "kaplan_meier_curve"
    png, svg, png_name, svg_name = export_figure(fig, base=safe_title)
    c = st.columns(2)
    with c[0]:
        st.download_button("Download PNG (300 DPI)", data=png, file_name=png_name, mime="image/png")
    with c[1]:
        st.download_button("Download SVG (vector)", data=svg, file_name=svg_name, mime="image/svg+xml")

# --- Risk tab (optional) ---
if show_risk_tab:
    with tab_objs[1]:
        default_ticks = [0]
        mx = int(dfp[time_col].max())
        for t in [30, 90, 180, 365, 730, 1095]:
            if t <= mx: default_ticks.append(t)
        default_ticks = sorted(set(default_ticks))
        tick_str = st.text_input("Display at-risk times (days, comma-separated)", ",".join(map(str, default_ticks)))
        try:
            show_times = [int(x.strip()) for x in tick_str.split(",") if x.strip()]
        except Exception:
            show_times = default_ticks
            st.warning("Could not parse; using defaults.")

        times = dfp[time_col].to_numpy()
        def nearest_idx(t): return int(np.argmin(np.abs(times - t)))
        table = {"Time (Days)": [times[nearest_idx(t)] for t in show_times]}
        if c1_risk in dfp.columns:
            table[label1 or "Cohort 1"] = [
                int(dfp[c1_risk].iloc[nearest_idx(t)]) if pd.notna(dfp[c1_risk].iloc[nearest_idx(t)]) else np.nan
                for t in show_times
            ]
        if c2_risk in dfp.columns:
            table[label2 or "Cohort 2"] = [
                int(dfp[c2_risk].iloc[nearest_idx(t)]) if pd.notna(dfp[c2_risk].iloc[nearest_idx(t)]) else np.nan
                for t in show_times
            ]
        risk_table = pd.DataFrame(table)
        st.dataframe(risk_table, use_container_width=True)
        buf, name = export_csv(risk_table, base="number_at_risk")
        st.download_button("Download At-Risk CSV", data=buf, file_name=name, mime="text/csv")

# --- Data tab ---
with tab_objs[-1]:
    st.markdown("**Plotted subset (respecting max-days and trimming trailing missing values):**")
    st.dataframe(dfp, use_container_width=True)
    sub_csv, sub_name = export_csv(dfp, base="km_plotted_subset")
    st.download_button("Download Plotted Subset CSV", data=sub_csv, file_name=sub_name, mime="text/csv")

# Nudge about limitations
if "event" not in " ".join(df.columns).lower():
    st.caption("Note: HRs/log-rank tests require event/censor data; KM CSV alone typically isn’t sufficient.")
