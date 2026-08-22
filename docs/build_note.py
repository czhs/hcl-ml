#!/usr/bin/env python3
"""Render TECHNICAL_NOTE.md to docs/TECHNICAL_NOTE.pdf (Markdown -> HTML -> PDF).

Uses the `markdown` package and a headless Chrome/Chromium for printing;
pass --chrome to point at the binary. Prints the resulting page count.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
CSS = """
@page { size: letter; margin: 0.55in 0.65in; }
body { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 9.4pt; line-height: 1.27; color: #111; }
h1 { font-size: 15pt; margin: 0 0 4pt 0; }
h2 { font-size: 11pt; margin: 9pt 0 3pt 0; border-bottom: 1px solid #999; padding-bottom: 1pt; }
p { margin: 0 0 5pt 0; text-align: justify; }
h1 + p { text-align: left; color: #444; }
ul { margin: 0 0 5pt 0; padding-left: 14pt; }
li { margin-bottom: 2pt; }
table { border-collapse: collapse; margin: 3pt 0 6pt 0; font-size: 9pt; }
th, td { border-bottom: 1px solid #bbb; padding: 1.5pt 8pt; text-align: left; }
th { border-bottom: 1.5px solid #444; }
td:not(:first-child), th:not(:first-child) { text-align: right; }
code { font-family: Menlo, Consolas, monospace; font-size: 8.6pt; }
a { color: #1a4d8f; text-decoration: none; }
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=str(ROOT / "TECHNICAL_NOTE.md"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "TECHNICAL_NOTE.pdf"))
    ap.add_argument("--chrome", default=None)
    args = ap.parse_args()

    html = markdown.markdown(Path(args.md).read_text(), extensions=["tables"])
    html_path = Path(args.out).with_suffix(".html")
    html_path.write_text(f"<!doctype html><meta charset='utf-8'><style>{CSS}</style><body>{html}</body>")

    chrome = args.chrome or next((c for c in [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", shutil.which("google-chrome"),
        shutil.which("chromium"), shutil.which("chromium-browser"), shutil.which("chrome")] if c and Path(c).exists()), None)
    if not chrome:
        sys.exit(f"wrote {html_path}; no Chrome found to print it — pass --chrome")
    subprocess.run([chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={args.out}", html_path.as_uri()], check=True, capture_output=True)
    try:
        import pypdf
        n = len(pypdf.PdfReader(args.out).pages)
        print(f"wrote {args.out}: {n} page(s)")
    except ImportError:
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
