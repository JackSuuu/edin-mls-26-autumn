import json, sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
rows = json.loads(Path('tmp/zero-diagnosis/all-zero-cases.json').read_text(encoding='utf-8'))
state, start, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rows = [r for r in rows if state == 'all' or r['coverage'] == state]
for r in rows[start:start+count]:
    print('\nCASE',r['query_id'],r['coverage'],'gold ranks',r['gold_reranked_positions'],'dense',r['dense_score'])
    print('Q',r['question'])
    print('A',r['answer'].replace('\n',' '))
    print('J',' '.join(r['judge_explanations']))
    for d in r['gold_documents']:
        print('GOLD',d['record_id'],'selected',d['selected_rank'],d['text'])
