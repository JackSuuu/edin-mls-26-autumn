from pathlib import Path
import sys
from pypdf import PdfReader

sys.stdout.reconfigure(encoding='utf-8')
root = Path(__file__).resolve().parent
reader = PdfReader(root / 'github-report.pdf')
pages = [f'\n=== PAGE {i+1} ===\n{page.extract_text()}' for i,page in enumerate(reader.pages)]
(root / 'github-report.txt').write_text('\n'.join(pages),encoding='utf-8')
print('\n'.join(pages))
