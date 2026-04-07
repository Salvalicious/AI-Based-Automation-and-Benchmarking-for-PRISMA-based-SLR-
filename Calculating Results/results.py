import pandas as pd
from pathlib import Path
from metrics import *

golden_dataset = "" # for example "SLR_PREG_W_Full_Text_Human.csv"
ai_screened_dataset = "" # for example "SLR_PREG_W_FULL_TEXT_AI_RESULT_0.50.csv"
output_text_file = "" # for example "SLR_PREG_W_FULL_TEXT_0.50_METRICS_V2.txt"
output_false_negatives_csv = "" #for example "false_negatives_SLR_PREG_W_FULL_TEXT_V2.csv"

HUMAN_CSV = golden_dataset
AI_CSV = ai_screened_dataset

df_human = pd.read_csv(HUMAN_CSV, sep=',', quotechar='"', encoding='utf-8')
df_ai = pd.read_csv(AI_CSV, sep=',', quotechar='"', encoding='utf-8')
print(df_human["Relevance"].unique())

y_true = df_human["Relevance"].astype(int).to_numpy()   # Golden_Datasets = ground truth
y_pred = df_ai["Relevance"].astype(int).to_numpy()      # AI_Dataset = prediction

if len(y_true) != len(y_pred):
    raise ValueError("Golden_Datasets and AI_Dataset CSVs do not have the same number of rows")

if not set(y_true).issubset({0, 1}):
    raise ValueError("Golden_Datasets relevance column contains values other than 0/1")

if not set(y_pred).issubset({0, 1}):
    raise ValueError("AI_Dataset relevance column contains values other than 0/1")


metrics = screening_metrics(y_true, y_pred)
metrics["WSS@95"] = wss_at_recall(metrics["Recall"], metrics["WorkSaved"])

TP, FP, TN, FN = confusion_counts(y_true, y_pred)

print("\n=== Title & Abstract Screening Evaluation ===\n")

# Confusion matrix
print("Confusion Matrix:")
print(f"               Predicted Relevant    Predicted Not Relevant")
print(f"Actual Relevant         {TP:5d}                  {FN:5d}")
print(f"Actual Not Relevant     {FP:5d}                  {TN:5d}\n")

# Other metrics
print("Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

# Optional: FN check
FN_indices = (y_true == 1) & (y_pred == 0)
num_fn = FN_indices.sum()
print(f"\nFalse Negatives (missed relevant studies): {num_fn}")

# Identify False Negatives
FN_mask = (y_true == 1) & (y_pred == 0)

# Extract the corresponding rows from the human dataset
#fn_df = df_human[FN_mask].copy()
fn_df = df_human[FN_mask]

# Add AI_Dataset probability column for reference
#fn_df["Probability"] = df_ai.loc[FN_mask, "Probability"].values

# Optional: reset index for clean printing
fn_df = fn_df.reset_index(drop=True)

# Print the dataframe
print("\n=== False Negatives (Missed Relevant Papers) ===\n")
print(fn_df)

fn_df.to_csv(output_false_negatives_csv, index=False, sep=';', encoding='utf-8')

with open(output_text_file, "w", encoding="utf-8") as f:
    f.write("=== Title & Abstract Screening Evaluation ===\n\n")

    f.write("Confusion Matrix:\n")
    f.write(f"               Predicted Relevant    Predicted Not Relevant\n")
    f.write(f"Actual Relevant         {TP:5d}                  {FN:5d}\n")
    f.write(f"Actual Not Relevant     {FP:5d}                  {TN:5d}\n\n")

    f.write("Metrics:\n")
    for k, v in metrics.items():
        f.write(f"{k}: {v:.4f}\n" if isinstance(v, float) else f"{k}: {v}\n")

    f.write(f"\nFalse Negatives (missed relevant studies): {num_fn}\n")


