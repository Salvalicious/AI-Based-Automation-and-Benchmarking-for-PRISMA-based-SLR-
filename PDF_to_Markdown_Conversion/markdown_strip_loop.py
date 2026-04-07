import os
import pymupdf.layout
import pymupdf4llm

INPUT_DIR = "" # for example /PDFs_and_Markdown_Files\Data_Extraction"
OUTPUT_DIR = "" #for example "/Data_Extraction\Markdown"

# Make sure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_SECTIONS = {
    "introduction": ["introduction", "background"],
    "methods": ["methods", "methodology", "materials and methods"],
    "results": ["results", "findings"],
    "discussion": ["discussion"],
    "conclusion": ["conclusion", "summary"]
}

ALL_KEYWORDS = [k for v in TARGET_SECTIONS.values() for k in v]

def is_header(line):
    s = line.strip()
    if s.startswith("#"):
        return True
    s = s.strip("*").strip()
    if len(s.split()) > 5 or any(p in s for p in ".:,;"):
        return False
    if s.isupper():
        return True
    return any(s.lower() == k for k in ALL_KEYWORDS)

def process_pdf(pdf_path, output_file):
    # Convert PDF to Markdown
    md = pymupdf4llm.to_markdown(pdf_path)

    lines = md.splitlines()
    sections = []
    current_header = None
    buffer = []

    for line in lines:
        if is_header(line):
            if current_header:
                sections.append((current_header, "\n".join(buffer).strip()))
            current_header = line.strip("#* ").strip()
            buffer = []
        else:
            if current_header:
                buffer.append(line)

    if current_header and buffer:
        sections.append((current_header, "\n".join(buffer).strip()))

    filtered = []
    for h, t in sections:
        if any(k in h.lower() for k in ALL_KEYWORDS):
            print(f"✅ Keeping section: {h} from {os.path.basename(pdf_path)}")
            filtered.append(f"## {h}\n{t}")

    result = "\n\n".join(filtered)

    # Save filtered markdown immediately
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result)

    print(f"💾 Saved filtered markdown: {output_file}\n")
    return result

# Process all PDFs_and_Markdown_Files in folder
for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith(".pdf"):
        continue

    pdf_path = os.path.join(INPUT_DIR, filename)
    output_file = os.path.join(OUTPUT_DIR, filename.replace(".pdf", ".md"))

    # Skip if markdown already exists
    if os.path.exists(output_file):
        print(f"⏩ Skipping {filename}, markdown already exists.")
        continue

    try:
        process_pdf(pdf_path, output_file)
    except Exception as e:
        print(f"❌ Failed to process {filename}: {e}")
        with open("failed_pdfs.txt", "a", encoding="utf-8") as f_fail:
            f_fail.write(filename + "\n")
        continue
