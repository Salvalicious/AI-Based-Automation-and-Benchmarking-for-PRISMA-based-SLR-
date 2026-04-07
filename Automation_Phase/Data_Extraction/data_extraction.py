import asyncio
import random
import os
import json
from datetime import datetime
import pandas as pd
import sys
from openai import AsyncOpenAI

# Folder with markdown files for full text (data extraction)
INPUT_FOLDER = ""  # e.g., "data/markdown_files"

# Output CSV folder and file path (timestamp appended automatically)
OUTPUT_FOLDER = ""  # for example, "Data_Extraction\SLR_CO_BEN\outputs"
OUTPUT_FILE = f"{OUTPUT_FOLDER}/SLR_Preg_W_data_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# Partial save checkpoint (optional)
PARTIAL_OUTPUT = "" # for example "Data_Extraction_SLR_CO_BEN_partial.csv"

# Log file (optional, mostly for debugging)
LOG_FILE = f"{OUTPUT_FOLDER}/Extraction_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"


class Tee:
    def __init__(self, filename):
        self.file = open(filename, "w", encoding="utf-8")
        self.console = sys.stdout

    def write(self, message):
        self.console.write(message)
        self.file.write(message)

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

CONCURRENT_PER_KEY = 1 #number of concurrent requests per key, adjust accordingly.

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
print(f"Loaded {len(df)} markdown files")

# =====================
# PROMPT BUILDER
# (Change schema here per dataset)
# =====================
def build_prompt(row):
    return f"""
You are a researcher performing a systematic review about diagnostic criteria for periodontitis in pregnant women.

Task: Extract information from the paper about each diagnostic criterion evaluated in the included primary studies.

Rules:
- Extract only information explicitly reported.
- Do not guess or infer.
- If a value is not reported, omit it from the object.
- Information may appear in text, tables, or supplementary material.

Instructions:
Scan the entire document. Identify all six included primary studies. For each index test evaluated, create a JSON object with:

- `study_reference`: citation including authors and year (e.g., "Conceição et al., 2021")
- `ref_num`: reference number in square brackets (e.g., 32)
- `criterion_name`: name/description of the diagnostic criterion
- `gold_standard`: reference standard used for comparison
- `test_type`: one of ["clinical_examination", "self_reported_questionnaire", "salivary_enzymatic_test", "microbiological_test", "other"]
- `parameters`: list of clinical parameters used in gold standard diagnosis (from Table 2's "Clinical parameters" column)
- `sensitivity`: sensitivity value with confidence interval (without % sign) (e.g., "86.5 (82.0-90.0)")
- `specificity`: specificity value with confidence interval (without % sign) (e.g., "72.50 (69.5-75.3)")
- `ppv`: positive predictive value if reported (without % sign)
- `npv`: negative predictive value if reported (without % sign)
- `lr_plus`: positive likelihood ratio if reported
- `lr_minus`: negative likelihood ratio if reported

Important:
- Create a separate object for EACH index test.
- Table 2 shows multiple index tests per study. Match each criterion with its specific values.
- Only include fields where values are explicitly reported.

Return a JSON array of objects. Return ONLY the JSON array.

Example for 1 entry:
[{{"study_reference": "Conceição et al., 2023", "ref_num": 32, "criterion_name": "CDC/AAP [9,10]", "gold_standard": "Gomes-Filho et al. [13]", "test_type": "clinical_examination", "parameters": ["probing depth", "clinical attachment level", "bleeding on probing"], "sensitivity": "86.5% (82.0-90.0)", "specificity": "72.50% (69.5-75.3)"}}]

Extract ALL diagnostic criteria. Do not stop early.

Full text:
{row['FullText']}
"""

# =====================
# JSON SAFE PARSER
# =====================
def safe_json_parse(text, row_index):
    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except Exception as e:
        print(f"Row {row_index}: JSON parse error")
        return None


# =====================
# ASYNC EXTRACTION FUNCTION
# =====================
async def extract_paper(client, row, i):
    max_retries = 6
    base_delay = 2

    for attempt in range(max_retries):
        try:
            response = await client.responses.create(
                model="gpt-5-nano",
                input=build_prompt(row),
            )

            output_text = response.output_text

            # SAVE RAW RESPONSE FIRST
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            raw_file = f"raw_response_{i}_{timestamp}.json"
            with open(raw_file, "w", encoding="utf-8") as f:
                f.write(output_text)
            print(f"Row {i}: Saved raw response to {raw_file}")

            # THEN TRY TO PARSE
            data = safe_json_parse(output_text, i)

            print(f"Study {i + 1}/{len(df)} processed")

            usage = response.usage
            input_tokens = getattr(usage, "input_tokens", 0)
            output_tokens = getattr(usage, "output_tokens", 0)
            total_tokens = getattr(usage, "total_tokens", 0)

            return i, data, output_text, input_tokens, output_tokens, total_tokens  # Also return raw text

        except Exception as e:
            error_text = str(e)
            print(f"Row {i}: Error {e}")
            return i, None, None, 0, 0, 0

    print(f"Row {i}: Failed after retries")
    return i, None, None, 0, 0, 0

# Semaphore per key
semaphores = [asyncio.Semaphore(CONCURRENT_PER_KEY) for _ in API_KEYS]

async def extract_with_key(key_index, row, i):
    async with semaphores[key_index]:
        client = AsyncOpenAI(api_key=API_KEYS[key_index])
        return await extract_paper(client, row, i)


# =====================
# MAIN LOOP
# =====================
async def main():
    tasks = []
    key_index = 0

    extracted_records = []
    all_raw_responses = []  # Store raw responses metadata

    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    completed = 0

    for i, row in df.iterrows():
        tasks.append(extract_with_key(key_index % len(API_KEYS), row, i))
        key_index += 1

    results = await asyncio.gather(*tasks)

    for i, data, raw_text, in_tok, out_tok, tot_tok in results:

        if raw_text:
            all_raw_responses.append({
                "index": i,
                "raw_length": len(raw_text),
                "preview": raw_text[:200]
            })

        if data:
            if isinstance(data, list):
                extracted_records.extend(data)
            elif isinstance(data, dict):
                extracted_records.append(data)

        total_input_tokens += in_tok
        total_output_tokens += out_tok
        total_tokens += tot_tok
        completed += 1

    # Save raw responses summary
    if all_raw_responses:
        pd.DataFrame(all_raw_responses).to_csv("raw_responses_summary.csv", index=False)

    # Final save
    if extracted_records:
        extracted_df = pd.DataFrame(extracted_records)
        extracted_df.to_csv(OUTPUT_FILE, index=False)

        # Also save as JSON
        json_output = OUTPUT_FILE.replace('.csv', '.json')
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(extracted_records, f, indent=2)
    else:
        print("WARNING: No records extracted. Check raw response files.")
        # Create empty file with note
        with open(OUTPUT_FILE.replace('.csv', '_ERROR.txt'), 'w') as f:
            f.write("No records extracted. Check raw_response_*.json files")

    print(f"Finished! Saved to {OUTPUT_FILE}")
    print(f"FINAL TOKENS — Input: {total_input_tokens}, Output: {total_output_tokens}, Total: {total_tokens}")

# =====================
# RUN
# =====================
if __name__ == "__main__":
    asyncio.run(main())