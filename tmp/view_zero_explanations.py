import json,sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
rows=json.loads(Path('tmp/zero-diagnosis/all-zero-cases.json').read_text(encoding='utf-8'))
for r in rows[int(sys.argv[1]):int(sys.argv[2])]:
    print(r['query_id'],r['coverage'],'Q:',r['question'],'J:',' '.join(r['judge_explanations']))
