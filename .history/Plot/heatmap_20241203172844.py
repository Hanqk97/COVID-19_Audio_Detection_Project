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
    models = ["CNN", "XGBoost"]
    tests = ["RTRV", "ATAV", "ATRV"]
    values = []

    for model in models:
        raw_avg = sum(data[model][f"raw_{i+1}"][metric_name] for i in range(5)) / 5
        aug_avg = sum(data[model][f"aug_{i+1}"][metric_name] for i in range(5)) / 5
        train_avg = data[model]["train"][metric_name]
        values.append([raw_avg, train_avg, aug_avg])

    # Create a DataFrame
    df = pd.DataFrame(values, columns=tests, index=models)

    # Sort rows based on "Augmented Data" values
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

# Generate and save heatmap for F1 Score
f1_df = create_dataframe("f1_score")
plot_heatmap(f1_df, "f1_score", f"{output_dir}/f1_score_heatmap.png")

# Generate and save heatmap for AUC
auc_df = create_dataframe("auc")
plot_heatmap(auc_df, "auc", f"{output_dir}/auc_heatmap.png")
