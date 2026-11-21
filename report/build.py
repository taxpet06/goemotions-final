"""Render report.md -> report.pdf (python-markdown + headless Chromium). Run: python report/build.py"""
import re
import subprocess
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
src = (HERE / "report.md").read_text()
meta, body = re.match(r"---\n(.*?)\n---\n(.*)", src, re.S).groups()
title = re.search(r'title: "(.*)"', meta).group(1)
author = re.search(r'author: "(.*)"', meta).group(1)
html = markdown.markdown(body, extensions=["tables"])
css = """
@page { size: Letter; margin: 1in; }
body { font-family: "Times New Roman", "DejaVu Serif", serif; font-size: 11pt; line-height: 1.2; color: #111; }
h1 { font-size: 16pt; margin: 0 0 4pt; line-height: 1.2; } .author { color: #444; margin-bottom: 14pt; }
h2 { font-size: 12.5pt; margin: 16pt 0 5pt; } h3 { font-size: 11pt; margin: 12pt 0 4pt; }
p, li { margin: 0 0 6pt; } ul, ol { padding-left: 18pt; }
table { border-collapse: collapse; width: 100%; font-size: 9pt; margin: 6pt 0 10pt; page-break-inside: avoid; }
th, td { border-bottom: 0.5pt solid #bbb; padding: 3pt 5pt; text-align: left; vertical-align: top; }
th { border-bottom: 1pt solid #333; }
img { max-width: 100%; max-height: 2.1in; display: block; margin: 6pt auto 2pt; page-break-inside: avoid; }
img[src*="gap_vs"] { max-height: 2.1in; }
p:has(> img) { font-size: 9pt; color: #333; text-align: center; page-break-inside: avoid; }
code { font-size: 9pt; }
.supp { page-break-before: always; }
p:has(+ table) { break-after: avoid; page-break-after: avoid; }
table { break-inside: avoid; }
"""
# Figure captions: markdown puts the alt text only in alt=""; repeat it visibly under the image.
html = re.sub(r'<p><img alt="([^"]*)" src="([^"]*)" /></p>', r'<p><img alt="\1" src="\2" /><br>\1</p>', html)
out = HERE / "report.html"
out.write_text(f"<!doctype html><meta charset='utf-8'><title>{title}</title><style>{css}</style>"
               f"<h1>{title}</h1><div class='author'>{author}</div>{html}")
subprocess.run(["chromium", "--headless", "--disable-gpu", "--no-sandbox", "--no-pdf-header-footer",
                f"--print-to-pdf={HERE / 'report.pdf'}", out.as_uri()], check=True, capture_output=True)
print("wrote", HERE / "report.pdf")
