import json,re,sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
root=Path.cwd()
base=root/'rag/results/qwen4-answers-3724290/zero-diagnosis-v1'
cases=[json.loads(s) for s in (base/'cases.jsonl').read_text(encoding='utf-8').splitlines()]
summary=json.loads((base/'summary.json').read_text(encoding='utf-8'))
scratch=json.loads((root/'tmp/zero-diagnosis/summary.json').read_text(encoding='utf-8'))
reviews=json.loads((base/'source-review.json').read_text(encoding='utf-8'))['cases']
assert len(cases)==len({r['query_id'] for r in cases})==121
assert summary['zero_pool_status']==scratch['zero_pool_status']
assert summary['zero_first_gold_reranked_position']==scratch['zero_first_gold_position']
assert all(c['answer'].strip() and c['finish_reason']=='stop' and not c['selected_text_truncated'] for c in cases)
caseids={c['query_id'] for c in cases}
reviewids={r['query_id'] for r in reviews}
assert len(reviews)==len(reviewids)==38 and reviewids<=caseids
assert {c['query_id'] for c in cases if c['labelled_top5_coverage']=='full'}<=reviewids
for p in [root/'rag/experiments/qwen4-zero-score-diagnosis.md',root/'rag/experiments/qwen4-answer-evaluation.md',root/'rag/extensions/README.md']:
    for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
        if '://' not in target and not target.startswith('#'):
            assert (p.parent/target.split('#')[0]).exists(),(p,target)
print('DIAGNOSIS_VERIFIED: 121 complete cases; independent coverage/position counts match; 38 unique source reviews include all 30 full-coverage zeros; local links resolve')
print('JUDGE_INPUT_MISREAD_FLAGS_IN_REVIEWED_CASES',sum('judge_input_misread' in r['flags'] for r in reviews))
