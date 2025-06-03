# Python Script for Kaplan-Meier Plot (Standalone Version)
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Generate a Kaplan-Meier curve from a CSV file.')
parser.add_argument('--file', required=True, help='Path to the Kaplan-Meier CSV file')
parser.add_argument('--label1', default='Cohort 1', help='Label for Cohort 1')
parser.add_argument('--label2', default='Cohort 2', help='Label for Cohort 2')
parser.add_argument('--color1', default='blue', help='Color for Cohort 1')
parser.add_argument('--color2', default='orange', help='Color for Cohort 2')
parser.add_argument('--style', choices=['color', 'bw'], default='color', help='Plot style')
parser.add_argument('--max_days', type=int, default=None, help='Maximum number of days to include')
parser.add_argument('--output', default='kaplan_meier_curve.png', help='Output filename for the plot')
args = parser.parse_args()

# Read data
df = pd.read_csv(args.file)
df.columns = df.columns.str.strip()
df.fillna(method='ffill', inplace=True)

if args.max_days:
    df = df[df['Time (Days)'] <= args.max_days]

# Prepare figure
fig, ax = plt.subplots(figsize=(10, 6))
time = df['Time (Days)']

# Select colors
if args.style == 'bw':
    color1, color2 = 'black', 'gray'
    alpha = 0.1
else:
    color1, color2 = args.color1, args.color2
    alpha = 0.2

# Plot Cohort 1
ax.plot(time, df['Cohort 1: Survival Probability'], label=args.label1, color=color1, linewidth=2)
if 'Cohort 1: Survival Probability 95 % CI Lower' in df.columns:
    ax.fill_between(time,
                    df['Cohort 1: Survival Probability 95 % CI Lower'],
                    df['Cohort 1: Survival Probability 95 % CI Upper'],
                    color=color1, alpha=alpha)

# Plot Cohort 2
ax.plot(time, df['Cohort 2: Survival Probability'], label=args.label2, color=color2, linewidth=2)
if 'Cohort 2: Survival Probability 95 % CI Lower' in df.columns:
    ax.fill_between(time,
                    df['Cohort 2: Survival Probability 95 % CI Lower'],
                    df['Cohort 2: Survival Probability 95 % CI Upper'],
                    color=color2, alpha=alpha)

# Final formatting
ax.set_title('Kaplan-Meier Survival Curve')
ax.set_xlabel('Time (Days)')
ax.set_ylabel('Survival Probability')
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig(args.output)
print(f"Plot saved as {args.output}")
