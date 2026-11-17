from pathlib import Path
import sys
from pypdf import PdfReader
sys.stdout.reconfigure(encoding='utf-8')
path=Path.cwd().parent/'02_group_report_rubric.pdf'
for i,page in enumerate(PdfReader(path).pages,1):
    print('PAGE',i,page.extract_text())
