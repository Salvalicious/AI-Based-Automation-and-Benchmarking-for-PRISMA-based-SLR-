import os

# Tell Python where to find Tesseract
os.environ["PATH"] += "" #for example ;C:\Program Files\Tesseract-OCR"
os.environ["TESSDATA_PREFIX"] = "" #for example r"C:\Program Files\Tesseract-OCR\tessdata"

import pymupdf.layout
import pymupdf4llm

INPUT_DIR = "" # for example /PDFs_and_Markdown_Files\Data_Extraction"
OUTPUT_DIR = "" # for example "/Data_Extraction\Markdown"

# Make sure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def convert_pdf_to_md(pdf_path, output_file):
    # Convert full PDF to Markdown (no filtering)
    md = pymupdf4llm.to_markdown(pdf_path, use_ocr=True)

    # Save markdown
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"💾 Saved full markdown: {output_file}")

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
        convert_pdf_to_md(pdf_path, output_file)
    except Exception as e:
        print(f"❌ Failed to process {filename}: {e}")
        with open("failed_pdfs.txt", "a", encoding="utf-8") as f_fail:
            f_fail.write(filename + "\n")