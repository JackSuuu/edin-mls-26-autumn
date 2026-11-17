from pathlib import Path
import hashlib,json,sys,zipfile
sys.path.insert(0,str(Path.cwd()/'rag'))
from extensions.passage_selection import make_windows
root=Path.cwd()
p=json.loads((root/'rag/experiments/passage-pilot-v1.json').read_text())
rank={r['query_id']:r for r in map(json.loads,(root/p['rankings_path']).read_text().splitlines())}
with zipfile.ZipFile(root/'rag-assets.zip') as z:
    records={r['record_id']:r['text'] for r in map(json.loads,z.read('data/rag_corpus/corpus.jsonl').decode().splitlines())}
counts=[]
for qid in p['query_ids']:
    ids=rank[qid]['qwen_matched']['record_ids'][:5]
    windows=make_windows(ids,records)
    assert windows and set(w['record_id'] for w in windows)==set(ids)
    for w in windows:
        assert records[w['record_id']][w['start']:w['end']]==w['text']
    counts.append(len(windows))
files=['rag/extensions/run_passage_pilot.py','rag/extensions/passage_selection.py','rag/tests/test_passage_selection.py',
       'rag/experiments/passage-pilot-v1.json','rag/scripts/icf_passage_prepare.sbatch','rag/scripts/icf_passage_answers.sbatch']
for name in files:
    assert b'\r' not in (root/name).read_bytes(),name
print('PASSAGE_PREFLIGHT',len(counts),'queries',sum(counts),'windows','range',min(counts),max(counts),'all 120 parents covered; exact source offsets; LF scripts')
manifest={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}
(root/'tmp/passage-upload-sha256.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
