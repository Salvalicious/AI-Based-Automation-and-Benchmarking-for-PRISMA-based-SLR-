import pymupdf4llm
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor


# ---------------- CONFIG ----------------
INPUT_DIR = "" # for example PDF_Markdown\SLR_CO_BEN_PDFs\Human\PDF")
OUTPUT_DIR = "" # for example PDF_Markdown\SLR_CO_BEN_PDFs\Human\Markdown")

MAX_WORKERS = 3  # Adjust depending on CPU cores you want to allocate (if ran on local machine)

# Headers (or partial header names) you want to REMOVE
REMOVE_SECTIONS = [
    "references",
    "acknowledgements",
    "funding",
    "appendix",
    "supplementary",
    "conflict of interest",
    "funding",
    "publication",
    "authors"
]
# ---------------------------------------


def pdf_to_markdown(pdf_path: Path) -> str:
    """Convert PDF to Markdown using pymupdf4llm."""
    return pymupdf4llm.to_markdown(str(pdf_path))

def is_section_header(line: str) -> str | None:
    """Detect headers in Markdown, bold, or standalone lines."""
    line_clean = line.strip()

    # Markdown headers (#, ##, ###)
    m = re.match(r"^(#+)\s+(.*)", line_clean)
    if m:
        return m.group(2).lower()

    # Bold headers (**Header**)
    m = re.match(r"^\*\*(.+?)\*\*$", line_clean)
    if m:
        return m.group(1).lower()

    # Standalone single-line headers
    if line_clean.isalpha() and len(line_clean) < 40:
        return line_clean.lower()

    return None

def remove_sections(markdown: str, headers_to_remove: list[str]) -> str:
    """Remove unwanted sections from Markdown."""
    lines = markdown.splitlines()
    output = []
    skip = False

    for line in lines:
        header = is_section_header(line)

        if header:
            # Optional speed-up: stop completely after References
            if "references" in header:
                break

            if any(h in header for h in headers_to_remove):
                skip = True
                continue
            else:
                skip = False

        if not skip:
            output.append(line)

    return "\n".join(output)

def process_pdf(pdf_path: Path):
    """Process a single PDF: convert, clean, save."""
    output_file = OUTPUT_DIR / f"{pdf_path.stem}.md"

    if output_file.exists():
        print(f"⏭ Skipping {pdf_path.name}, output already exists.")
        return

    print(f"📄 Processing: {pdf_path.name}")
    try:
        md = pdf_to_markdown(pdf_path)
        cleaned_md = remove_sections(md, REMOVE_SECTIONS)
        output_file.write_text(cleaned_md, encoding="utf-8")
        print(f"✅ Saved: {output_file.name}")
    except Exception as e:
        print(f"❌ Failed on {pdf_path.name}: {e}")

def main():
    pdf_files = sorted(INPUT_DIR.glob("*.pdf"))

    if not pdf_files:
        print("⚠️ No PDF files found in input folder.")
        return

    # --- Use multiprocessing instead of threads for safety ---
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(process_pdf, pdf_files)

    print("✅ All PDFs_and_Markdown_Files processed.")

if __name__ == "__main__":
    main()
