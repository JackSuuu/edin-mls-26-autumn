import json,re,sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
rows={r['query_id']:r for r in map(json.loads,Path('rag/results/qwen4-answers-3724290/zero-diagnosis-v1/cases.jsonl').read_text(encoding='utf-8').splitlines())}
targets={
    'q_371c6ca10bf1bf5e':['carry','future profits','mental'],
    'q_9cd791a970ab648f':['Authorization','Capture'],
    'q_45ce8f0db1865a56':['Illinois','W-4','12-13'],
    'q_8706e2925ba6d565':['high','personal loan'],
    'q_2803366d84526278':['Help to Buy'],
}
for qid,terms in targets.items():
    print('\nCASE',qid)
    for d in rows[qid]['selected_documents']:
        for term in terms:
            match=re.search(re.escape(term),d['text'],re.I)
            if match:
                print('SOURCE',d['record_id'],'GOLD',d['is_positive_qrel'],'TERM',term,'EXCERPT',d['text'][max(0,match.start()-90):match.end()+240])
