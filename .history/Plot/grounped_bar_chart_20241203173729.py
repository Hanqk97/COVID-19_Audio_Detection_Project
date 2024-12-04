import json
import numpy as np
import matplotlib.pyplot as plt

output_dir = "Plot"

# Load the JSON data
with open("result/summary.json", "r") as f:
    data = json.load(f)

# Function to extract values for raw and augmented data
def extract_values(model_name, metric_name):
    raw_values = [data[model_name][f"raw_{i+1}"][metric_name] for i in range(5)]
    aug_values = [data[model_name][f"aug_{i+1}"][metric_name] for i in range(5)]
    return raw_values, aug_values

# Function to plot grouped bar chart
def plot_grouped_bar_chart(model_name, raw_f1, aug_f1, raw_auc, aug_auc, filename):
    x_labels = [f"Fold {i+1}" for i in range(5)]
    x = np.arange(len(x_labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot bars for F1 Score and AUC
    ax.bar(x - 1.5 * width, raw_f1, width, label='F1 Score (Raw)', color='#4c72b0')
    ax.bar(x - 0.5 * width, aug_f1, width, label='F1 Score (Aug)', color='#92c6ff')
    ax.bar(x + 0.5 * width, raw_auc, width, label='AUC (Raw)', color='#f59542')
    ax.bar(x + 1.5 * width, aug_auc, width, label='AUC (Aug)', color='#ffbc79')

    # Add labels, title, and legend
    ax.set_ylabel('Metric Value')
    ax.set_title(f'Comparison of F1 Score and AUC for {model_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend(loc='upper left', fontsize='small')  # Place legend in top-left

    # Save the plot
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Generate and save grouped bar charts for each model
models = ["CNN", "XGBoost"]

for model in models:
    raw_f1, aug_f1 = extract_values(model, "f1_score")
    raw_auc, aug_auc = extract_values(model, "auc")
    filename = f"{output_dir}/{model.lower()}_f1_auc_comparison_bar_chart.png"
    plot_grouped_bar_chart(model, raw_f1, aug_f1, raw_auc, aug_auc, filename)
