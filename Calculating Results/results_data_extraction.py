import pandas as pd
import numpy as np
import json
import re
import unicodedata
from metrics import *

golden_dataset = "" #for example "baseline_json_co_ben.json"
ai_screened_dataset = "" #for example "SLR_CO_BEN_AI_Data_Extraction.json"

output_text_file = "" #for example "SLR_CO_BEN_EXTRACTION_METRICS.txt"
output_errors_csv = "" #for example "extraction_errors.csv"


# =====================
# LOAD JSON
# =====================

with open(golden_dataset, encoding="utf-8") as f:
    human_data = json.load(f)

with open(ai_screened_dataset, encoding="utf-8") as f:
    ai_data = json.load(f)

print("Golden_Datasets records:", len(human_data))
print("AI_Dataset records:", len(ai_data))


# =====================
# NORMALIZATION
# =====================

def normalize(x):

    if x is None:
        return ""

    def clean(v):

        v = unicodedata.normalize("NFKD", str(v))
        v = v.encode("ascii", "ignore").decode("ascii")
        v = v.lower().strip()

        v = re.sub(r"[_\-]", " ", v)
        v = re.sub(r"\s+", " ", v)

        return v

    if isinstance(x, list):
        return sorted([clean(i) for i in x])

    return clean(x)


# =====================
# RECORD IDENTIFIER
# =====================

def record_id(r):

    return normalize(r.get("ref", ""))


# =====================
# INDEX DATA
# =====================

human_dict = {record_id(r): r for r in human_data}
ai_dict = {record_id(r): r for r in ai_data}

all_records = sorted(set(human_dict.keys()) | set(ai_dict.keys()))

print("Golden_Datasets entries:", len(human_dict))
print("AI_Dataset entries:", len(ai_dict))
print("Union entries:", len(all_records))


# =====================
# EXTRACTION COVERAGE
# =====================

matched_records = set(human_dict.keys()) & set(ai_dict.keys())
coverage = len(matched_records) / len(human_dict)

print("\nExtraction Coverage:", round(coverage,4))


# =====================
# BUILD DATAFRAME
# =====================

records = []

for record in all_records:

    h = human_dict.get(record, {})
    a = ai_dict.get(record, {})

    fields = set(h.keys()) | set(a.keys())

    for field in fields:

        if field == "ref":
            continue

        human_value = normalize(h.get(field))
        ai_value = normalize(a.get(field))

        correct = human_value == ai_value

        records.append({
            "record": record,
            "field": field,
            "human": human_value,
            "ai": ai_value,
            "correct": correct
        })


df = pd.DataFrame(records)

print("Total field predictions:", len(df))


# =====================
# CREATE LABELS
# =====================

y_true = np.ones(len(df))
y_pred = df["correct"].astype(int).to_numpy()


# =====================
# METRICS
# =====================

metrics = screening_metrics(y_true, y_pred)
metrics["WSS@95"] = wss_at_recall(metrics["Recall"], metrics["WorkSaved"])

TP, FP, TN, FN = confusion_counts(y_true, y_pred)


# =====================
# PRINT RESULTS
# =====================

print("\n=== Data Extraction Evaluation ===\n")

print("Confusion Matrix:")
print(f"               Predicted Correct    Predicted Incorrect")
print(f"Actual Correct         {TP:5d}                  {FN:5d}")
print(f"Actual Incorrect       {FP:5d}                  {TN:5d}\n")

print("Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


# =====================
# FIELD ACCURACY
# =====================

field_accuracy = (
    df.groupby("field")["correct"]
    .mean()
    .sort_values(ascending=False)
)

print("\nField Accuracy:")
print(field_accuracy)


# =====================
# STUDY-LEVEL ACCURACY
# =====================

study_accuracy = (
    df.groupby("record")["correct"]
    .all()
    .mean()
)

print("\nStudy-level Accuracy:")
print(round(study_accuracy, 4))


# =====================
# ERROR ANALYSIS
# =====================

errors_df = df[df["correct"] == False]

print("\nExtraction Errors:", len(errors_df))

print("\n=== Example Errors ===\n")
print(errors_df.head())

errors_df.to_csv(output_errors_csv, index=False, sep=';', encoding='utf-8')


# =====================
# SAVE RESULTS
# =====================

with open(output_text_file, "w", encoding="utf-8") as f:

    f.write("=== Data Extraction Evaluation ===\n\n")

    f.write("Confusion Matrix:\n")
    f.write(f"               Predicted Correct    Predicted Incorrect\n")
    f.write(f"Actual Correct         {TP:5d}                  {FN:5d}\n")
    f.write(f"Actual Incorrect       {FP:5d}                  {TN:5d}\n\n")

    f.write("Metrics:\n")

    for k, v in metrics.items():
        f.write(f"{k}: {v:.4f}\n" if isinstance(v, float) else f"{k}: {v}\n")

    f.write("\nField Accuracy:\n")
    f.write(field_accuracy.to_string())

    f.write("\n\nStudy-level Accuracy:\n")
    f.write(f"{study_accuracy:.4f}\n")

    f.write(f"\n\nExtraction Errors: {len(errors_df)}\n")

print("\nResults exported successfully.")