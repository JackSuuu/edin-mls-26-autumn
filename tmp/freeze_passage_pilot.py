from pathlib import Path
import hashlib,json,statistics,sys
root=Path.cwd()
key=root/'rag/results/qwen4-answers-3724290/zero-review-v1/review-key.json'
qids=list(dict.fromkeys(r['query_id'] for r in json.loads(key.read_text(encoding='utf-8'))))
assert len(qids)==24
protocol={
 'schema':'passage-pilot-v1','split':'practice','query_ids':qids,
 'sample_policy':'Reuse all 24 questions from the previously frozen zero-review-v1 diagnostic sample; score-conditioned exploratory sample, not representative validation',
 'source_sample_sha256':hashlib.sha256(key.read_bytes()).hexdigest(),
 'rankings_path':'rag/results/qwen4-retrieval-3724174/full/rankings.jsonl',
 'rankings_sha256':'0eebc96ee37e5a9786d5e9d68d8cd7ac15e2286253630e999574fad9c4579146',
 'parent_records':5,'window_sentences':3,'window_overlap':1,'input_budget':768,
 'arms':['full5','prefix768','passages768'],
 'passage_scorer':'Qwen/Qwen3-Reranker-4B','passage_scorer_revision':'22e683669bc0f0bd69640a1354a6d0aebcfeede5',
 'passage_score_max_tokens':2048,'forbid_scoring_truncation':True,'batch_size':8,
 'generator_model':'utter-project/EuroLLM-22B-Instruct-2512','generator_temperature':0.1,
 'judge_model':'utter-project/EuroLLM-22B-Instruct-2512','judge_json_mode':True,'judge_repeats':1,
 'arm_order_seed':'passage-pilot-20261117-v1','bootstrap_seed':20261117,'bootstrap_replicates':2000,
 'stop_after_pilot':True,'dev_not_used':True,
 'budget_rationale':'Fixed compression control below the observed approximately 1267-token full-context mean; no budget sweep or tuning on new scores',
 'limitations':['All practice questions and earlier scores have been inspected; this is a mechanism pilot.','Three-sentence overlap retains neighbouring text but does not guarantee preserving every applicability condition.','Equal maximum budgets do not imply equal actual token counts.','No full-practice or dev expansion is automatic.']}
out=root/'rag/experiments/passage-pilot-v1.json'
if out.exists(): raise SystemExit('Protocol already exists; do not overwrite')
out.write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
answers={r['query_id']:r for r in map(json.loads,(root/'rag/results/qwen4-answers-3724290/answers/qwen4_matched5.jsonl').read_text(encoding='utf-8').splitlines())}
tokens=[answers[q]['local_input_tokens'] for q in qids]
print('FROZEN_SAMPLE',len(qids),'source tokens min/median/max',min(tokens),statistics.median(tokens),max(tokens),'above 768',sum(t>768 for t in tokens))
