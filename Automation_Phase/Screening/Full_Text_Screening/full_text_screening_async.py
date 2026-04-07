import asyncio
import random
import re
import os
from datetime import datetime
import pandas as pd
import sys
from openai import AsyncOpenAI

# Input folder containing markdown files of full text papers
INPUT_FOLDER = ""  # for example, "PDFs_and_Markdown_Files\SLR_CO_BEN_PDFs\Human\Markdown"

# Output CSV folder and file path (timestamp appended automatically)
OUTPUT_FOLDER = ""  # for example, "Full_Text_Screening\outputs"
OUTPUT_FILE = f"{OUTPUT_FOLDER}/SLR_CO_BEN_Full_Text_with_probability_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# Partial file (checkpoint in case token limit is reached or 404 error appears)
PARTIAL_OUTPUT = "FullText_CO_BEN_partial_prob.csv"

# Log file (optional, mostly for debugging)
LOG_FILE = f"{OUTPUT_FOLDER}/SLR_CO_BEN_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

class Tee:
    def __init__(self, filename):
        self.file = open(filename, "w", encoding="utf-8")
        self.console = sys.stdout

    def write(self, message):
        self.console.write(message)   # show in PyCharm console
        self.file.write(message)      # save to file

    def flush(self):
        self.console.flush()
        self.file.flush()

sys.stdout = Tee(LOG_FILE)
sys.stderr = sys.stdout


API_KEYS = [
    "",  #  "api_key"
    "",  # "api_key_2"
    "",  #  "api_key_3" etc.
]
CONCURRENT_PER_KEY = 1

# =====================
# LOAD MARKDOWN FILES
# =====================
records = []

for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith(".md"):
        path = os.path.join(INPUT_FOLDER, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except:
            with open(path, "r", encoding="latin-1") as f:
                text = f.read()

        records.append({
            "Title": os.path.splitext(filename)[0],
            "FullText": text
        })

df = pd.DataFrame(records)
df['Probability'] = None

print(f"Loaded {len(df)} markdown files")

# =====================
# PROMPT BUILDER
# =====================
def build_prompt(row):
    return f"""
You are screening a study for the **full-text stage** of a systematic literature review (SLR) on the **co-benefits of active travel interventions beyond physical activity**.

Task: Decide if this study is eligible based on the full text, using the inclusion/exclusion criteria below.

Inclusion criteria:
- The study evaluates a **real-world active travel intervention** intended to change **transport-related physical activity** (e.g., walking, cycling, active commuting, or public transport-related activity).
- Active travel interventions may include **infrastructure changes, environmental modifications, programs, campaigns, incentives, or policies**, as long as their purpose is to influence transport-related physical activity.
- The study reports **at least one physical activity or active travel outcome** (e.g., walking, cycling, active travel behaviour, mode share, or transport-related physical activity).
- The study reports **at least one non-physical activity outcome (co-benefit)**, such as:
  - Health outcomes beyond physical activity  
  - Safety outcomes  
  - Environmental outcomes (e.g., emissions, air quality, noise)  
  - Economic outcomes  
  - Transport outcomes (e.g., congestion, travel time, accessibility)  
  - Social outcomes
- The study uses an **intervention evaluation design**, such as:
  - Randomised controlled trial (RCT)
  - Cluster RCT
  - Quasi-experimental study
  - Natural experiment
  - Controlled before–after, interrupted time series, or comparison across groups, areas, or time.
- The study provides **empirical evaluation results** of the intervention (not only description or theory).

Exclusion criteria (exclude if any apply):
- The study does **not evaluate an active travel intervention** (e.g., observational association studies, modelling/simulation only, or descriptive policy analysis without evaluation).
- The intervention is a **general physical activity or exercise program** not related to transport.
- The study reports **no physical activity or active travel outcome**.
- The study reports **no co-benefits beyond physical activity**.
- The study uses an **ineligible design**, such as cross-sectional only, qualitative-only, descriptive report without effect evaluation.
- The paper is a **review, protocol, editorial, commentary, narrative review, or conceptual discussion** without primary empirical evaluation.
- Conference abstracts, posters, or short papers **without sufficient methodological detail**.
- The study does **not evaluate the effects of an intervention**.

Instructions:
- Respond with ONLY a single decimal number between 0.0 and 1.0 representing the confidence/probability that the study is eligible.
    - If any exclusion criterion is clearly met, assign 0.0.
    - Confidence should reflect how well the study meets the inclusion criteria.


Full text:
{row['FullText'][:120000]}
"""

async def call_with_retry(client, prompt, row_index, max_retries=6):
    delay = 2  # initial wait

    for attempt in range(max_retries):
        try:
            response = await client.responses.create(
                model="gpt-5-nano",
                input=prompt,
            )
            return response

        except Exception as e:
            error_text = str(e)

            # Handle rate limit (429 / TPM)
            if "429" in error_text or "rate_limit" in error_text.lower():
                wait_time = delay * (2 ** attempt) + random.uniform(0, 1)
                print(
                    f"Row {row_index}: Rate limited. "
                    f"Retrying in {wait_time:.1f}s (attempt {attempt+1}/{max_retries})"
                )
                await asyncio.sleep(wait_time)
                continue

            # Temporary network issues
            if "timeout" in error_text.lower() or "connection" in error_text.lower():
                wait_time = delay * (2 ** attempt)
                print(
                    f"Row {row_index}: Network issue. "
                    f"Retrying in {wait_time:.1f}s"
                )
                await asyncio.sleep(wait_time)
                continue

            # Other errors → don't retry
            print(f"Row {row_index}: Unhandled error: {e}")
            return None

    print(f"Row {row_index}: Failed after {max_retries} retries.")
    return None


# =====================
# ASYNC SCREENING FUNCTIONS
# =====================
async def screen_paper(client, row, i):
    max_retries = 6
    base_delay = 2  # seconds

    for attempt in range(max_retries):
        try:
            response = await client.responses.create(
                model="gpt-5-nano",
                input=build_prompt(row),   # <-- kept exactly as requested
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
            error_text = str(e)

            # ----- RATE LIMIT / TPM -----
            if "429" in error_text or "rate_limit" in error_text.lower():
                wait_time = base_delay * (2 ** attempt)
                print(
                    f"Row {i}: Rate limit hit. "
                    f"Waiting {wait_time:.1f}s before retry "
                    f"(attempt {attempt+1}/{max_retries})"
                )
                await asyncio.sleep(wait_time)
                continue

            # ----- NETWORK / TEMP -----
            if "timeout" in error_text.lower() or "connection" in error_text.lower():
                wait_time = base_delay * (2 ** attempt)
                print(f"Row {i}: Network issue. Retrying in {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                continue

            # ----- OTHER ERROR -----
            print(f"Error on row {i}: {e}")
            return i, None, 0, 0, 0

    print(f"Row {i}: Failed after {max_retries} retries.")
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

        # PARTIAL SAVE EVERY 20 (recommended for full text)
        if completed % 20 == 0:
            df.to_csv(PARTIAL_OUTPUT, index=False)
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
