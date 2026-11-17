import collections,json,statistics,sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
run=Path('rag/results/qwen4-answers-3724290')
cases=json.loads(Path('tmp/zero-diagnosis/all-zero-cases.json').read_text(encoding='utf-8'))
judges=[json.loads(s) for s in (run/'judge_json_qwen4_matched5/answer_judge_per_query.jsonl').read_text(encoding='utf-8').splitlines()]
print('ZERO_DIMENSIONS',{d:sum(r[d]==0 for r in judges) for d in ('correctness','grounding','completeness')})
print('MOST_COMMON_TRIPLES',collections.Counter(tuple(r[d] for d in ('correctness','grounding','completeness')) for r in judges).most_common(10))
for qid in ['q_039f48f42921b2be','q_15481dc07d08e3b7','q_41e3c12e4d573cb5','q_c5a5320bcfd3a373','q_dd94af61414916d5','q_33e70afa6a74be26','q_8de6d5607824f3c0']:
    r=next(c for c in cases if c['query_id']==qid)
    print('\nCASE',qid,r['coverage'],'Q',r['question'],'A',r['answer'],'J',r['judge_explanations'])
    for d in r['gold_documents']:
        print('GOLD',d['record_id'],d['text'][:1500])
    if qid in ['q_c5a5320bcfd3a373','q_41e3c12e4d573cb5']:
        for d in r['selected_documents']:
            print('SELECTED',d['record_id'],d['text'][:650])
