import asyncio
import re
from datetime import datetime
import pandas as pd
from openai import AsyncOpenAI

# Path to input CSV (Title + Abstract)
INPUT_FILE = ""  # for example, "SLR_CO_BEN_Baseline_Clean_Dataset.csv"

# Output folder and file path (timestamp appended automatically)
OUTPUT_FOLDER = ""  # for example, "outputs"
OUTPUT_FILE = f"{OUTPUT_FOLDER}/SLR_CO_BEN_with_probability_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# Partial output checkpoint
PARTIAL_OUTPUT_FILE = "" #Adjust based on file naming you want to use, for example SLR_CO_BEN_partial_prob.csv

# OpenAI API keys, you can use just 1 or multiple API Keys.
API_KEYS = [
    "",  #  "api_key"
    "",  # "api_key_2"
    "",  #  "api_key_3" etc.
]

CONCURRENT_PER_KEY = 8 #number of concurrent requests per key, adjust accordingly.

# =====================
# LOAD DATA
# =====================
df = pd.read_csv(
    INPUT_FILE,
    sep=',',
    usecols=[0, 1],
    quotechar='"',
    encoding='utf-8'
)
#df = df.iloc[:5].copy()

df.columns = ["Title", "Abstract"]
df['Relevance'] = None
df['Probability'] = None

# =====================
# PROMPT BUILDER
# =====================
def build_prompt(row):
    return f"""
You are screening a study for a systematic literature review (SLR) or meta-analysis on interventions that aim to change active travel behavior (e.g., walking, cycling, or physically active public transport).
    Task: Decide if this study is potentially relevant based on the title and abstract together, using the inclusion/exclusion criteria below. 

Inclusion criteria (relevant studies generally meet one or more):
    - Studies that appear to evaluate an intervention using experimental, quasi-, or natural-experimental designs (including non-randomised), or any clear assessment of intervention effects.
    - Interventions targeting individuals, settings, built environment, or policies to change active travel; recreational/fitness-only interventions excluded.
    - Outcomes include transport-related physical activity or clearly stated co-benefits plausibly resulting from changes in active travel behavior (e.g., safety, health, economic, environmental, transport quality).
    - Comparisons with no/minimal intervention, attention control, or variation in exposure, if mentioned.

Exclusion criteria (not relevant):
    - Studies that clearly lack any evaluative component (e.g., descriptive reports without assessment of change).
    - Modelling/simulation not evaluating real-life interventions.
    - Studies on overall activity unrelated to transport.
    - Narrative reviews, commentaries, or policy documents that do not report or evaluate outcomes of a real-world intervention.

Instructions:
- Respond with ONLY a single decimal number between 0.0 and 1.0 representing the confidence/probability that the study is relevant.
    - If any exclusion criterion is clearly met, assign 0.0.
    - Confidence should reflect how clearly the study meets one or more inclusion criteria; the more criteria and the clearer they appear, the higher the score.
    - Any mention of a comparison, before–after assessment, or variation in exposure strengthens confidence, but its absence does not automatically exclude the study.
    - Do NOT classify title and abstract separately.
 

Title:
{row['Title']}

Abstract:
{row['Abstract']}
"""

# =====================
# ASYNC SCREENING FUNCTIONS
# =====================
async def screen_paper(client, row, i):
    try:
        response = await client.responses.create(
            model="gpt-5-nano",
            input=build_prompt(row),
        )

        output_text = response.output_text.strip()
        match = re.search(r'\b0\.\d+|1\.0|0|1\b', output_text)
        probability = float(match.group(0)) if match else None

        print(f"Study {i+1}/{len(df)} processed. GPT response: {output_text}")

        usage = response.usage
        input_tokens = getattr(usage, "input_tokens", 0)
        output_tokens = getattr(usage, "output_tokens", 0)
        total_tokens = getattr(usage, "total_tokens", 0)

        return i, probability, input_tokens, output_tokens, total_tokens

    except Exception as e:
        print(f"Error on row {i}: {e}")
        return i, None, 0, 0, 0

# Semaphore per key
semaphores = [asyncio.Semaphore(CONCURRENT_PER_KEY) for _ in API_KEYS]

async def screen_with_key(key_index, row, i):
    async with semaphores[key_index]:
        client = AsyncOpenAI(api_key=API_KEYS[key_index])
        return await screen_paper(client, row, i)

# =====================
# MAIN ASYNC LOOP
# =====================
async def main():
    tasks = []
    key_index = 0

    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    completed = 0

    for i, row in df.iterrows():
        tasks.append(screen_with_key(key_index % len(API_KEYS), row, i))
        key_index += 1

    results = await asyncio.gather(*tasks)

    for i, prob, in_tok, out_tok, tot_tok in results:
        df.at[i, 'Probability'] = prob

        total_input_tokens += in_tok
        total_output_tokens += out_tok
        total_tokens += tot_tok
        completed += 1

        # PARTIAL SAVE EVERY 100 COMPLETED
        if completed % 100 == 0:
            df.to_csv(partial_output, index=False)
            print(f"Checkpoint saved at {completed} completed papers")
            print(
                f"Cumulative tokens — "
                f"Input: {total_input_tokens}, "
                f"Output: {total_output_tokens}, "
                f"Total: {total_tokens}"
            )

    # Final save
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Finished! CSV saved as {OUTPUT_FILE}")
    print(
        f"FINAL TOKEN COUNT — "
        f"Input: {total_input_tokens}, "
        f"Output: {total_output_tokens}, "
        f"Total: {total_tokens}"
    )

# =====================
# RUN SCRIPT
# =====================
if __name__ == "__main__":
    asyncio.run(main())
