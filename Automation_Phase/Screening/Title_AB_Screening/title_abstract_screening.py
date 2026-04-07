
from openai import OpenAI
import pandas as pd
import time
import re
from datetime import datetime

# Path to the input CSV (Title + Abstract)
INPUT_FILE = ""  # for example "SLR_CO_BEN_Baseline_Clean_Dataset.csv"

# Output folder or file path (timestamp will be added automatically)

OUTPUT_FOLDER = ""  # for example "outputs"
OUTPUT_FILE = f"{OUTPUT_FOLDER}/SLR_RT_with_relevance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# Partial output checkpoint
PARTIAL_OUTPUT_FILE = "" #e.g. "SLR_RT_partial_putput.csv"

# OpenAI API key (set in environment variable or here)
OPENAI_API_KEY = ""  # <-- DO NOT hardcode your real key on GitHub!

df = pd.read_csv(
    INPUT_FILE,
    sep=';',          # semicolon delimiter
    usecols=[0, 1],   # only Title + Abstract
    quotechar='"',    # handle commas inside text
    encoding='utf-8'
)

df.columns = ["Title", "Abstract"]
def get_average_length():
  df['abstract_length_chars'] = df['Abstract'].str.len()
  avg_chars_abstract = df['abstract_length_chars'].mean()
  print("Average characters per abstract:", avg_chars_abstract)
  df['title_length_chars'] = df['Title'].str.len()
  avg_chars_title = df['title_length_chars'].mean()
  print("Average characters per title:", avg_chars_title)

#df = df.iloc[:5].copy()

df['Relevance'] = None

client = OpenAI(
  api_key="OPENAI_API_KEY",
)


def build_prompt(row):
  """
  Build a prompt for GPT-5-nano to classify a study as relevant (1) or not (0)
  based on the inclusion/exclusion criteria for AI_Dataset-based tools for SLRs/Meta-Analyses.
  """
  return f"""
  You are screening a study for a systematic literature review (SLR) or Meta-Analysis.
  On the diagnostic accuracy of periodontitis criteria in pregnant women.
  Task: Decide if this study is relevant based on the title and abstract together, using the inclusion/exclusion criteria derived from Conceição et al., 2024.

  Inclusion criteria (relevant studies should meet one or more):
  - Studies related to periodontitis in pregnant women
  - Studies reporting primary data on diagnostic criteria or tests for periodontitis.
  - Studies that provide information on diagnostic outcomes, such as sensitivity, specificity, predictive values, or likelihood ratios.
  - Studies using human participants only, with animal models explicitly excluded

  Exclusion criteria (not relevant):
  - Studies that do not focus on pregnant women.
  - Studies that use animal models instead of human participants.
  - Reviews, editorials, letters, or opinion pieces without primary diagnostic data.
  - Studies with insufficient data to extract diagnostic outcomes.
  
  Instructions:
  - Respond with **ONLY a single digit**:
      - 1 → study is relevant
      - 0 → study is NOT relevant
  - Do NOT classify the title and abstract separately.
  - Consider **both the title and abstract together** for your decision.
  - If a study matches any exclusion criterion, it is automatically NOT relevant

  Title:
  {row['Title']}

  Abstract:
  {row['Abstract']}
  """

# response = client.responses.create(
#  model="gpt-5-nano",
#  input="write a haiku about ai",
#  store=True,
# )

start_time = time.perf_counter()
total_input = 0
total_output = 0
total_tokens = 0

for i, row in df.iterrows():
    prompt = build_prompt(row)
    try:
        response = client.responses.create(
            model="gpt-5-nano",
            input=prompt,
            #max_output_tokens=250,
            #temperature=0
        )

        output_text = response.output_text.strip()
        match = re.search(r'\b([01])\b', output_text)

        if match:
            df.at[i, 'Relevance'] = int(match.group(1))
        else:
            print(f"⚠ Unexpected or empty output on row {i}: {repr(output_text)}")
            df.at[i, 'Relevance'] = None

        # ✅ Progress message
        print(f"Study {i+1}/{len(df)} processed. GPT response: {output_text}")
        #print(f"Study {i + 1}/{len(df)} raw GPT DOI_MATCH_FINAL: {repr(response.output_text)}")

        if hasattr(response, "usage") and response.usage is not None:
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            total_row_tokens = response.usage.total_tokens

            total_input += input_tokens
            total_output += output_tokens
            total_tokens += total_row_tokens

            #print("Tokens used this row:")
            #print(f"  Input tokens:  {input_tokens}")
            #print(f"  DOI_MATCH_FINAL tokens: {output_tokens}")
            #print(f"  Total tokens:  {total_row_tokens}")
            #print(f"Cumulative total tokens so far: {total_tokens}")

        else:
            print("No usage info available")


    except Exception as e:
        print(f"Error on row {i}: {e}")
        df.at[i, 'Relevance'] = None
    #time.sleep(0.5)


    if (i + 1) % 100 == 0:  # every 100 rows
        df.to_csv("SLR_RT_partial.csv", index=False)
        print(f"Checkpoint saved at row {i + 1}")
        print(f"  Cumulative tokens so far: {total_tokens}")

df.to_csv(OUTPUT_FILE, index=False)

end_time = time.perf_counter()
elapsed = end_time - start_time

print(f"Finished! CSV saved as {OUTPUT_FILE}")
print(f"Took {elapsed:.2f} seconds ({elapsed/60:.2f} minutes).")


#print(response.output_text);
