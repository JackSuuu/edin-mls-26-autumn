import json, zipfile
from pathlib import Path

root = Path.cwd()
run = root / 'rag/results/qwen4-answers-3724290'
with zipfile.ZipFile(root / 'rag-assets.zip') as z:
    names = [n for n in z.namelist() if n.endswith(('.jsonl', '.tsv', '.csv', '.json')) and 'models/' not in n]
    print('ASSET_FILES', names[:35])
    for n in names:
        if 'quer' in n.lower() or 'question' in n.lower():
            print('QUERY_SCHEMA', n, z.read(n).decode('utf-8-sig').splitlines()[0][:1800])
for p in ['judge_json_qwen4_matched5/answer_judge_per_query.jsonl', 'answers/qwen4_matched5.jsonl']:
    row = json.loads((run / p).read_text(encoding='utf-8').splitlines()[0])
    print(p, list(row))
rank = json.loads((root / 'rag/results/qwen4-retrieval-3724174/full/rankings.jsonl').read_text(encoding='utf-8').splitlines()[0])
print('RANKING_KEYS', list(rank), 'DENSE_KEYS', list(rank['dense']))
