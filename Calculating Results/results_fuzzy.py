import pandas as pd
import numpy as np
import re
from rapidfuzz import process, fuzz
from metrics import *

golden_dataset = "" # for example "SLR_CO_BEN_Full_Text_Human.csv"
ai_screened_dataset = "" # for example "SLR_CO_BEN_FULL_TEXT_AI_RESULT_0.50.csv"

output_text_file = "" # for example "SLR_CO_BEN_FULL_TEXT_0.50_METRICS_V2.txt"
output_false_negatives_csv = "" #for example "false_negatives_SLR_CO_BEN_FULL_TEXT_V2.csv"

FUZZY_THRESHOLD = 90 #fuzzy threshold level

def normalize_doi(doi):
    if pd.isna(doi):
        return None
    return str(doi).lower().replace(".", "_").replace("/", "_").strip()


def normalize_title(title):
    if pd.isna(title):
        return ""
    return str(title).lower().strip()


# =====================
# LOAD DATA
# =====================

df_human = pd.read_csv(golden_dataset)
df_ai = pd.read_csv(ai_screened_dataset)

print("Golden_Datasets relevant papers:", len(df_human))
print("AI_Dataset screened papers:", len(df_ai))


# =====================
# NORMALIZE HUMAN DATA
# =====================

df_human["doi_norm"] = df_human["doi_adjusted"].apply(normalize_doi)
df_human["title_norm"] = df_human["Title"].apply(normalize_title)

human_doi_set = set(df_human["doi_norm"].dropna())
human_title_list = df_human["title_norm"].tolist()
human_title_set = set(human_title_list)


# =====================
# NORMALIZE AI_Dataset DATA
# =====================

df_ai["doi_norm"] = df_ai["Title"].apply(normalize_doi)
df_ai["title_norm"] = df_ai["Title"].apply(normalize_title)


# =====================
# RECONSTRUCT TRUE LABELS
# =====================

y_true = []
y_pred = []

matched_dois = set()

for _, row in df_ai.iterrows():

    ai_doi = row["doi_norm"]
    ai_title = row["title_norm"]
    ai_pred = int(row["Relevance"])

    relevant = False

    # DOI match
    if ai_doi and ai_doi in human_doi_set:
        relevant = True
        matched_dois.add(ai_doi)

    # exact title match
    elif ai_title in human_title_set:
        relevant = True

    # fuzzy match
    else:
        match = process.extractOne(ai_title, human_title_list, scorer=fuzz.token_sort_ratio)
        if match and match[1] >= FUZZY_THRESHOLD:
            relevant = True

    y_true.append(1 if relevant else 0)
    y_pred.append(ai_pred)


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# =====================
# HANDLE HUMAN PAPERS NOT IN AI_Dataset DATASET
# =====================

matched_relevant = sum(y_true)
missing_human = len(df_human) - matched_relevant

print("Matched relevant papers:", matched_relevant)
print("Unmatched human relevant papers:", missing_human)

if missing_human > 0:
    y_true = np.concatenate([y_true, np.ones(missing_human)])
    y_pred = np.concatenate([y_pred, np.zeros(missing_human)])


# =====================
# METRICS
# =====================

TP, FP, TN, FN = confusion_counts(y_true, y_pred)

metrics = screening_metrics(y_true, y_pred)
metrics["WSS@95"] = wss_at_recall(metrics["Recall"], metrics["WorkSaved"])


# =====================
# PRINT RESULTS
# =====================

print("\n=== Full Text Screening Evaluation ===\n")

print("Confusion Matrix:")
print(f"               Predicted Relevant    Predicted Not Relevant")
print(f"Actual Relevant         {TP:5d}                  {FN:5d}")
print(f"Actual Not Relevant     {FP:5d}                  {TN:5d}\n")

print("Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

print(f"\nFalse Negatives (missed relevant studies): {FN}")


# =====================
# EXPORT FALSE NEGATIVES
# =====================

FN_mask = (y_true[:len(df_ai)] == 1) & (y_pred[:len(df_ai)] == 0)

fn_df = df_ai[FN_mask].copy()
fn_df = fn_df.reset_index(drop=True)

fn_df.to_csv(output_false_negatives_csv, index=False, sep=';', encoding='utf-8')


# =====================
# SAVE TEXT RESULTS
# =====================

with open(output_text_file, "w", encoding="utf-8") as f:

    f.write("=== Full Text Screening Evaluation ===\n\n")

    f.write("Confusion Matrix:\n")
    f.write(f"               Predicted Relevant    Predicted Not Relevant\n")
    f.write(f"Actual Relevant         {TP:5d}                  {FN:5d}\n")
    f.write(f"Actual Not Relevant     {FP:5d}                  {TN:5d}\n\n")

    f.write("Metrics:\n")

    for k, v in metrics.items():
        if isinstance(v, float):
            f.write(f"{k}: {v:.4f}\n")
        else:
            f.write(f"{k}: {v}\n")

    f.write(f"\nFalse Negatives (missed relevant studies): {FN}\n")

print("\nResults exported successfully.")