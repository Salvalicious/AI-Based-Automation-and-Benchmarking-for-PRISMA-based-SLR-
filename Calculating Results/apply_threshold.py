import pandas as pd
from pathlib import Path

INPUT_FILE = "" # for example "SLR_CO_BEN_Full_Text_with_probability_20260217_000906.csv"
THRESHOLD = 0.5
PROB_COL = "Probability"
RELEVANCE_COL = "Relevance"
OUTPUT_PREFIX = "" #output_prefix for example SLR_CO_BEN_FULL_TEXT_AI_RESULT

def threshold_csv(input_csv=INPUT_FILE, threshold=THRESHOLD,
                  prob_col=PROB_COL, relevance_col=RELEVANCE_COL,
                  output_prefix=OUTPUT_PREFIX, output_csv=None):
    """
    Read CSV with probabilities and convert to 0/1 relevance based on threshold.

    Args:
        input_csv: path to CSV with probability column
        threshold: probability threshold to classify as relevant
        prob_col: column name with AI_Dataset probability (0.0-1.0)
        relevance_col: column name to write 0/1 predictions
        output_prefix: string prefix for output CSV filename
        output_csv: optional output CSV path. If None, filename is generated dynamically

    Returns:
        pandas DataFrame with Relevance column filled
    """
    df = pd.read_csv(input_csv)

    # Create 0/1 relevance column based on threshold
    df[relevance_col] = (df[prob_col] >= threshold).astype(int)

    # Generate dynamic output CSV filename if not provided
    if output_csv is None:
        p = Path(input_csv).parent  # same folder as input
        output_csv = str(p / f"{output_prefix}_{threshold:.2f}.csv")

    # Save CSV
    df.to_csv(output_csv, index=False)
    print(f"Thresholded CSV saved as: {output_csv}")

    return df

# Run with current config
threshold_csv()
