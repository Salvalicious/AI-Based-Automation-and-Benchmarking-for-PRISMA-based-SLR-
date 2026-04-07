
from openai import OpenAI, AsyncOpenAI
import pandas as pd
import time
import re
from datetime import datetime

# Path to input CSV (Title + Abstract)
INPUT_FILE = ""  # for example, "SLR_CO_BEN_Baseline_Clean_Dataset.csv"

# Output folder and file path (timestamp appended automatically)
OUTPUT_FOLDER = ""  # for example, "outputs"
OUTPUT_FILE = f"{OUTPUT_FOLDER}/SLR_CO_BEN_with_probability_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# Partial output checkpoint
PARTIAL_OUTPUT_FILE = "" #Adjust based on file naming you want to use, for example SLR_CO_BEN_partial_prob.csv

# OpenAI API key
OPENAI_API_KEY = ""


df = pd.read_csv(
    INPUT_FILE,
    sep=',',          # semicolon delimiter
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

df = df.iloc[:5].copy()

df['Relevance'] = None
df['Probability'] = None

client = OpenAI(
  api_key= OPENAI_API_KEY
)


def build_prompt(row):
  """
  Build a prompt for GPT-5-nano to classify a study as relevant (1) or not (0)
  based on the inclusion/exclusion criteria for AI_Dataset-based tools for SLRs/Meta-Analyses.
  """
  return f"""
You are screening a study for a systematic literature review (SLR) or meta-analysis on interventions that aim to change active travel behavior (e.g., walking, cycling, or physically active public transport).
    Task: Decide if this study is potentially relevant based on the title and abstract together, using the inclusion/exclusion criteria below.

Inclusion criteria (relevant studies generally meet one or more):
    - Empirical studies that appear to evaluate an intervention using experimental, quasi-, or natural-experimental approaches (including non-randomised).
    - Interventions targeting individuals, settings, built environment, or policies to change active travel; recreational/fitness-only interventions are excluded.
    - Outcomes include transport-related physical activity or potential co-benefits of active travel (e.g., safety, health, economic, environmental, transport quality, social). Abstracts mentioning potential benefits should be considered at TL/AB.
    - Comparisons with no/minimal intervention, attention control, or controlled variation in exposure, if mentioned.

Exclusion criteria (not relevant):
    - Modelling/simulation not evaluating real-life interventions.
    - Studies on overall activity unrelated to transport.
    - Narrative reviews, commentaries, or policy documents that do not report or evaluate outcomes of a real-world intervention.

Instructions:
- Respond with ONLY a single decimal number between 0.0 and 1.0 representing the confidence/probability that the study is relevant.
    - If any exclusion criterion is clearly met, assign 0.0.
    - Confidence should reflect how clearly the study meets one or more inclusion criteria; the more criteria and the clearer they appear, the higher the score.
    - Do NOT classify title and abstract separately.
    - Any mention of a comparison, before–after assessment, or evaluation of change strengthens confidence, but its absence does not automatically exclude the study.
 


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
        match = re.search(r'\b0\.\d+|1\.0|0|1\b', output_text)

        if match:
            df.at[i, 'Probability'] = float(match.group(0))
        else:
            print(f"⚠ Unexpected or empty output on row {i}: {repr(output_text)}")
            df.at[i, 'Probability'] = None

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

            print("Tokens used this row:")
            print(f"  Input tokens:  {input_tokens}")
            print(f"  Output tokens: {output_tokens}")
            print(f"  Total tokens:  {total_row_tokens}")
            print(f"Cumulative total tokens so far: {total_tokens}")

        else:
            print("No usage info available")


    except Exception as e:
        print(f"Error on row {i}: {e}")
        df.at[i, 'Probability'] = None
    #time.sleep(0.5)


    if (i + 1) % 100 == 0:  # every 100 rows
        df.to_csv(partial_output, index=False)
        print(f"Checkpoint saved at row {i + 1}")
        print(f"  Cumulative tokens so far: {total_tokens}")

df.to_csv(OUTPUT_FILE, index=False)

end_time = time.perf_counter()
elapsed = end_time - start_time

print(f"Finished! CSV saved as {OUTPUT_FILE}")
print(f"Took {elapsed:.2f} seconds ({elapsed/60:.2f} minutes).")


#print(response.output_text);
