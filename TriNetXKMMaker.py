# Kaplan-Meier Plot Script for Google Colab
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import ipywidgets as widgets
from google.colab import files
import io

# Step 1: File upload widget
print("Step 1: Upload your Kaplan-Meier CSV file")
uploaded = files.upload()
filename = next(iter(uploaded))

# Automatically detect and skip irrelevant rows
lines = io.StringIO(uploaded[filename].decode('utf-8')).readlines()
header_keywords = ["Time (Days)", "Cohort 1: Survival Probability"]
header_row_idx = next(i for i, line in enumerate(lines) if all(keyword in line for keyword in header_keywords))
df = pd.read_csv(io.BytesIO(uploaded[filename]), skiprows=header_row_idx)

# Clean column names and fill missing data
df.columns = df.columns.str.strip()
df.fillna(method='ffill', inplace=True)

# Step 2: Parameter selection
print("\nStep 2: Set your display parameters")
style_selector = widgets.ToggleButtons(
    options=['Color', 'Black & White'],
    description='Style:',
    disabled=False
)
color1_picker = widgets.ColorPicker(description="Cohort 1 Color", value='blue')
color2_picker = widgets.ColorPicker(description="Cohort 2 Color", value='orange')
cohort1_name = widgets.Text(value='Cohort 1', description='Label 1:')
cohort2_name = widgets.Text(value='Cohort 2', description='Label 2:')
max_days = widgets.IntText(value=int(df['Time (Days)'].max()), description='Max Days:')

param_widgets = widgets.VBox([style_selector, color1_picker, color2_picker, cohort1_name, cohort2_name, max_days])
display(param_widgets)

# Step 3: Run analysis button
run_button = widgets.Button(description="Run Analysis")
output_area = widgets.Output()

def on_run_button_clicked(b):
    with output_area:
        output_area.clear_output()
        df_limited = df[df['Time (Days)'] <= max_days.value]
        time = df_limited['Time (Days)']

        fig, ax = plt.subplots(figsize=(10, 6))

        if style_selector.value == 'Black & White':
            color1, color2 = 'black', 'gray'
            ci_alpha = 0.1
        else:
            color1, color2 = color1_picker.value, color2_picker.value
            ci_alpha = 0.2

        ax.plot(time, df_limited['Cohort 1: Survival Probability'], label=cohort1_name.value, color=color1, linewidth=2)
        if 'Cohort 1: Survival Probability 95 % CI Lower' in df.columns:
            ax.fill_between(time,
                            df_limited['Cohort 1: Survival Probability 95 % CI Lower'],
                            df_limited['Cohort 1: Survival Probability 95 % CI Upper'],
                            alpha=ci_alpha, color=color1)

        ax.plot(time, df_limited['Cohort 2: Survival Probability'], label=cohort2_name.value, color=color2, linewidth=2)
        if 'Cohort 2: Survival Probability 95 % CI Lower' in df.columns:
            ax.fill_between(time,
                            df_limited['Cohort 2: Survival Probability 95 % CI Lower'],
                            df_limited['Cohort 2: Survival Probability 95 % CI Upper'],
                            alpha=ci_alpha, color=color2)

        ax.set_title('Kaplan-Meier Survival Curve')
        ax.set_xlabel('Time (Days)')
        ax.set_ylabel('Survival Probability')
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()

        # Save format selector and save button
        format_selector = widgets.Dropdown(
            options=['png', 'jpg'],
            value='png',
            description='Format:',
        )

        save_button = widgets.Button(description="Download Figure")

        def save_fig_callback(button):
            fmt = format_selector.value
            file_name = f"kaplan_meier_curve.{fmt}"
            fig.savefig(file_name, format=fmt, dpi=300)
            files.download(file_name)

        save_button.on_click(save_fig_callback)
        display(widgets.HBox([format_selector, save_button]))

run_button.on_click(on_run_button_clicked)
display(run_button, output_area)
