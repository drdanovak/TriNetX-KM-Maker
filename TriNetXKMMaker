# Streamlit Kaplan-Meier Curve Web App
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Kaplan-Meier Curve Generator", layout="centered")
st.title("Kaplan-Meier Curve Generator")

# Step 1: File upload
uploaded_file = st.file_uploader("Upload your Kaplan-Meier CSV file")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    df.fillna(method='ffill', inplace=True)

    # Step 2: User Parameters
    st.subheader("Display Parameters")
    style = st.radio("Select style", ['Color', 'Black & White'])
    cohort1_color = st.color_picker("Cohort 1 Color", '#1f77b4')
    cohort2_color = st.color_picker("Cohort 2 Color", '#ff7f0e')
    label1 = st.text_input("Cohort 1 Label", "Cohort 1")
    label2 = st.text_input("Cohort 2 Label", "Cohort 2")
    max_days = st.number_input("Max Days", 0, int(df['Time (Days)'].max()), value=int(df['Time (Days)'].max()))

    # Step 3: Plotting
    df = df[df['Time (Days)'] <= max_days]
    time = df['Time (Days)']

    fig, ax = plt.subplots(figsize=(10, 6))

    if style == 'Black & White':
        color1, color2, alpha = 'black', 'gray', 0.1
    else:
        color1, color2, alpha = cohort1_color, cohort2_color, 0.2

    ax.plot(time, df['Cohort 1: Survival Probability'], label=label1, color=color1, linewidth=2)
    if 'Cohort 1: Survival Probability 95 % CI Lower' in df.columns:
        ax.fill_between(time,
                        df['Cohort 1: Survival Probability 95 % CI Lower'],
                        df['Cohort 1: Survival Probability 95 % CI Upper'],
                        alpha=alpha, color=color1)

    ax.plot(time, df['Cohort 2: Survival Probability'], label=label2, color=color2, linewidth=2)
    if 'Cohort 2: Survival Probability 95 % CI Lower' in df.columns:
        ax.fill_between(time,
                        df['Cohort 2: Survival Probability 95 % CI Lower'],
                        df['Cohort 2: Survival Probability 95 % CI Upper'],
                        alpha=alpha, color=color2)

    ax.set_title("Kaplan-Meier Survival Curve")
    ax.set_xlabel("Time (Days)")
    ax.set_ylabel("Survival Probability")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Step 4: Download option
    st.download_button("Download Figure as PNG", data=fig_to_bytes(fig), file_name="kaplan_meier_curve.png")

# Helper function to export figure
from io import BytesIO
def fig_to_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return buf.getvalue()
