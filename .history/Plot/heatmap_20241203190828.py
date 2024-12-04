import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

output_dir = "Plot"

# Load the JSON data
with open("result/summary.json", "r") as f:
    data = json.load(f)

# Extract relevant metrics and organize into a DataFrame
def create_dataframe(metric_name):
    models = ["CNN", "XGBoost", "RandomForest", "SVM"]
    tests = ["RTRV", "ATAV", "ATRV"]
    values = []

    for model in models:
        raw_avg = sum(data[model][f"raw_{i+1}"][metric_name] for i in range(5)) / 5
        aug_avg = sum(data[model][f"aug_{i+1}"][metric_name] for i in range(5)) / 5
        train_avg = data[model]["train"][metric_name]
        values.append([raw_avg, train_avg, aug_avg])

    # Create a DataFrame
    df = pd.DataFrame(values, columns=tests, index=models)

    # Sort rows based on "ATRV" (Augmented Test) values
    df = df.sort_values(by="ATRV", ascending=False)

    return df

# Generate heatmaps and save them to files
def plot_heatmap(df, metric_name, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="coolwarm", cbar_kws={"label": metric_name.capitalize()})
    plt.title(f"Heatmap of {metric_name.capitalize()} across Models and Tests")
    plt.ylabel("Model")
    plt.xlabel("Experiment Type")
    plt.tight_layout()
    plt.savefig(filename)  # Save the heatmap to a file
    plt.close()

# Generate combined heatmap figure
# Generate combined heatmap with a taller aspect ratio
def plot_combined_heatmaps(f1_df, auc_df, filename):
    fig, axes = plt.subplots(2, 1, figsize=(8, 16), sharex=True)  # Two heatmaps stacked vertically

    # F1 Score Heatmap
    sns.heatmap(f1_df, annot=True, fmt=".3f", cmap="coolwarm", cbar=False, ax=axes[0])
    axes[0].set_title("F1 Score")
    axes[0].set_ylabel("Model")
    axes[0].set_xlabel("")

    # AUC Heatmap
    sns.heatmap(auc_df, annot=True, fmt=".3f", cmap="coolwarm", cbar_kws={"label": "Metric Value"}, ax=axes[1])
    axes[1].set_title("AUC")
    axes[1].set_ylabel("Model")
    axes[1].set_xlabel("Experiment Type")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Generate and save heatmaps for individual metrics
f1_df = create_dataframe("f1_score")
plot_heatmap(f1_df, "f1_score", f"{output_dir}/f1_score_heatmap.png")

auc_df = create_dataframe("auc")
plot_heatmap(auc_df, "auc", f"{output_dir}/auc_heatmap.png")

# Generate and save combined heatmap
plot_combined_heatmaps(f1_df, auc_df, f"{output_dir}/combined_heatmaps.png")
