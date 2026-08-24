"""Extract text from a PDF file for reading by the agent.

Usage: python pdf_extract.py <path.pdf> [--pages 1-5] [--out txt]

Reads the PDF with PyMuPDF (fitz), prints page text to stdout (or --out file).
If a page has no extractable text (scanned image), it is flagged so the caller
knows OCR (pytesseract) is needed.
"""
import sys, re, argparse

try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("PyMuPDF not installed: pip install pymupdf")


def extract(path, pages=None):
    doc = fitz.open(path)
    n = doc.page_count
    if pages:
        m = re.match(r"(\d+)(?:-(\d+))?$", pages.strip())
        lo = int(m.group(1)) - 1
        hi = (int(m.group(2)) if m.group(2) else lo + 1)
        idx = range(lo, min(hi, n))
    else:
        idx = range(n)
    out = []
    for i in idx:
        page = doc[i]
        txt = page.get_text("text")
        if not txt.strip():
            txt = f"[PAGE {i+1}: NO TEXT LAYER — scanned image, OCR required]"
        out.append(f"\n----- PAGE {i+1}/{n} -----\n{txt}")
    doc.close()
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--pages", help="e.g. 1-5 or 3")
    ap.add_argument("--out", help="write to file instead of stdout")
    a = ap.parse_args()
    text = extract(a.path, a.pages)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Wrote {len(text)} chars -> {a.out}")
    else:
        print(text)
