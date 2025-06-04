import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import io
import re
from base64 import b64encode

# Title and Instructions
st.title("Kaplan-Meier Survival Curve Maker")
st.markdown("Upload your Kaplan-Meier CSV output from TriNetX. The tool will clean the file, plot the survival curve, and provide you with a PNG file for your manuscript or poster.")

# Step 1: File Upload
uploaded_file = st.file_uploader("Upload CSV file that you downloaded from TriNetX", type=["csv"])
if uploaded_file:
    lines = uploaded_file.getvalue().decode("utf-8").splitlines()
    header_keywords = ["Time (Days)", "Cohort 1: Survival Probability"]
    header_row_idx = next(i for i, line in enumerate(lines) if all(k in line for k in header_keywords))
    df = pd.read_csv(uploaded_file, skiprows=header_row_idx)

    # Clean data
    df.columns = df.columns.str.strip()
    df.fillna(method='ffill', inplace=True)

    # Step 2: User Parameters
    st.subheader("Customize Your Graph")
    style = st.radio("Style", ['Color', 'Black & White'])
    color1 = st.color_picker("Cohort 1 Color", '#1f77b4')
    color2 = st.color_picker("Cohort 2 Color", '#ff7f0e')
    label1 = st.text_input("Label for Cohort 1", "Cohort 1")
    label2 = st.text_input("Label for Cohort 2", "Cohort 2")
    plot_title = st.text_input("Plot Title", "Kaplan-Meier Survival Curve")
    max_days = st.number_input("Maximum Days to Display", min_value=0, max_value=int(df['Time (Days)'].max()), value=int(df['Time (Days)'].max()))

    # Step 3: Generate Plot
    if st.button("Generate Plot"):
        df_limited = df[df['Time (Days)'] <= max_days]
        time = df_limited['Time (Days)']

        fig, ax = plt.subplots(figsize=(10, 6))

        # Style and transparency
        if style == 'Black & White':
            color1_use, color2_use = 'black', 'gray'
            ci_alpha = 0.1
        else:
            color1_use, color2_use = color1, color2
            ci_alpha = 0.2

        # Plot cohort 1
        ax.plot(time, df_limited['Cohort 1: Survival Probability'], label=label1, color=color1_use, linewidth=2)
        if 'Cohort 1: Survival Probability 95 % CI Lower' in df.columns:
            ax.fill_between(time,
                            df_limited['Cohort 1: Survival Probability 95 % CI Lower'],
                            df_limited['Cohort 1: Survival Probability 95 % CI Upper'],
                            color=color1_use, alpha=ci_alpha)

        # Plot cohort 2
        ax.plot(time, df_limited['Cohort 2: Survival Probability'], label=label2, color=color2_use, linewidth=2)
        if 'Cohort 2: Survival Probability 95 % CI Lower' in df.columns:
            ax.fill_between(time,
                            df_limited['Cohort 2: Survival Probability 95 % CI Lower'],
                            df_limited['Cohort 2: Survival Probability 95 % CI Upper'],
                            color=color2_use, alpha=ci_alpha)

        ax.set_title(plot_title)
        ax.set_xlabel('Time (Days)')
        ax.set_ylabel('Survival Probability')
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Step 4: PNG Download with Button
        st.subheader("Download Plot")

        # Create filename from cleaned title
        cleaned_title = re.sub(r'[^\w\-_. ]', '', plot_title).strip().replace(" ", "_")
        filename = f"{cleaned_title or 'kaplan_meier_curve'}.png"

        img_bytes = io.BytesIO()
        fig.savefig(img_bytes, format='png', dpi=300, bbox_inches='tight')
        img_bytes.seek(0)
        b64 = b64encode(img_bytes.read()).decode()

        download_button = st.download_button(
            label="Download Plot",
            data=b64,
            file_name=filename,
            mime="image/png",
            key="download_button"
        )
