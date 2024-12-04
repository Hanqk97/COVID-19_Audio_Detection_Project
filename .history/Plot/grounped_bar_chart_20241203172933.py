import json
import numpy as np
import matplotlib.pyplot as plt

output_dir = "Plot"

# Load the JSON data
with open("result/summary.json", "r") as f:
    data = json.load(f)

# Function to calculate differences for each fold
def calculate_differences(model_name, metric_name):
    raw_values = [data[model_name][f"raw_{i+1}"][metric_name] for i in range(5)]
    aug_values = [data[model_name][f"aug_{i+1}"][metric_name] for i in range(5)]
    differences = [aug - raw for raw, aug in zip(raw_values, aug_values)]
    return differences

# Function to plot grouped bar chart
def plot_grouped_bar_chart(model_name, metric_differences, metric_name, filename):
    x_labels = [f"Fold {i+1}" for i in range(5)]
    x = np.arange(len(x_labels))  # the label locations
    width = 0.4  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, metric_differences, width, label=f'{metric_name.capitalize()} Difference')

    # Add labels, title, and legend
    ax.set_ylabel('Difference (Aug - Raw)')
    ax.set_title(f'{metric_name.capitalize()} Differences (Augmented - Raw) for {model_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()

    # Save the plot
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Generate and save grouped bar charts for each model
models = ["CNN", "XGBoost"]
metrics = ["f1_score", "auc"]

for model in models:
    for metric in metrics:
        differences = calculate_differences(model, metric)
        filename = f"{output_dir}/{model.lower()}_{metric}_differences_bar_chart.png"
        plot_grouped_bar_chart(model, differences, metric, filename)
