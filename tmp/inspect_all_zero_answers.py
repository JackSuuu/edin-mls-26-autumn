import collections, csv, hashlib, io, json, re, statistics, zipfile
from pathlib import Path

ROOT = Path.cwd()
RUN = ROOT / 'rag/results/qwen4-answers-3724290'
def read(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
answers = {arm: {r['query_id']: r for r in read(RUN / 'answers' / (arm + '.jsonl'))} for arm in ('dense5', 'qwen4_matched5')}
judges = {arm: {r['query_id']: r for r in read(RUN / ('judge_json_' + arm) / 'answer_judge_per_query.jsonl')} for arm in answers}
ranks = {r['query_id']: r for r in read(ROOT / 'rag/results/qwen4-retrieval-3724174/full/rankings.jsonl')}
with zipfile.ZipFile(ROOT / 'rag-assets.zip') as archive:
    corpus = {r['record_id']: r for r in map(json.loads, archive.read('data/rag_corpus/corpus.jsonl').decode('utf-8-sig').splitlines())}
    manifest = list(map(json.loads, archive.read('data/rag_corpus/student_audio_manifest.jsonl').decode('utf-8-sig').splitlines()))
    qrels = collections.defaultdict(dict)
    for r in csv.DictReader(io.StringIO(archive.read('data/rag_corpus/qrels_practice.tsv').decode('utf-8-sig')), delimiter='\t'):
        if float(r['score']) > 0:
            qrels[r['query_id']][r['record_id']] = float(r['score'])
print('MANIFEST_SAMPLE', manifest[0])
print('CORPUS_FIELDS', list(next(iter(corpus.values()))))
inventory, coverage, pools = [], collections.defaultdict(collections.Counter), collections.Counter()
for qid, answer in answers['qwen4_matched5'].items():
    gold = qrels[qid]
    selected = answer['retrieved_record_ids']
    matched = ranks[qid]['qwen_matched']['record_ids']
    present = set(gold) & set(selected)
    pool = set(gold) & set(ranks[qid]['dense']['record_ids'])
    state = 'none' if not present else 'full' if len(present) == len(gold) else 'partial'
    zero = judges['qwen4_matched5'][qid]['normalised_answer_score'] == 0
    coverage[state]['questions'] += 1
    coverage[state]['zeros'] += zero
    if not zero:
        continue
    pool_status = 'no_gold_in_pool100' if not pool else 'gold_in_pool_but_none_selected' if not present else 'some_gold_selected'
    pools[pool_status] += 1
    a = answer['answer']
    inventory.append({
        'query_id': qid, 'question': answer['transcript'], 'coverage': state,
        'positive_qrels': len(gold), 'selected_positive_qrels': len(present), 'pool100_positive_qrels': len(pool),
        'pool_status': pool_status,
        'gold_reranked_positions': {rid: matched.index(rid) + 1 if rid in matched else None for rid in gold},
        'dense_score': judges['dense5'][qid]['normalised_answer_score'],
        'dense_recall5': ranks[qid]['dense']['metrics']['Recall@5'],
        'qwen_recall5': ranks[qid]['qwen_matched']['metrics']['Recall@5'],
        'native_recall5': ranks[qid]['qwen_native2048']['metrics']['Recall@5'],
        'answer_words': len(a.split()), 'input_tokens': answer['local_input_tokens'],
        'finish_reason': answer['finish_reason'],
        'selected_text_truncated': any(r['truncated'] for r in answer['selected_documents']),
        'refusal_phrase_flag': bool(re.search(r'insufficient (?:information|evidence)|not enough information|cannot (?:determine|provide)|unable to|evidence (?:provided )?does not (?:provide|contain|address)|no (?:specific )?information', a, re.I)),
        'answer': a, 'judge_explanations': judges['qwen4_matched5'][qid]['judge_explanations'],
        'dense_answer': answers['dense5'][qid]['answer'],
        'dense_judge_explanations': judges['dense5'][qid]['judge_explanations'],
        'gold_documents': [{'record_id': rid, 'relevance': rel, 'selected_rank': selected.index(rid)+1 if rid in selected else None, 'text': corpus[rid]['text']} for rid, rel in gold.items()],
        'selected_documents': [{'record_id': rid, 'is_positive_qrel': rid in gold, 'text': corpus[rid]['text']} for rid in selected]
    })
summary = {
    'questions': 350, 'zeros': len(inventory), 'coverage': dict(coverage), 'zero_pool_status': dict(pools),
    'zeros_with_refusal_phrase_flag_not_manual_diagnosis': sum(r['refusal_phrase_flag'] for r in inventory),
    'zero_input_tokens': {'min': min(r['input_tokens'] for r in inventory), 'max': max(r['input_tokens'] for r in inventory), 'mean': statistics.mean(r['input_tokens'] for r in inventory)},
    'zero_finish_reasons': dict(collections.Counter(r['finish_reason'] for r in inventory)),
    'zero_selected_text_truncated': sum(r['selected_text_truncated'] for r in inventory),
    'zero_first_gold_position': dict(collections.Counter('absent' if not (positions := [p for p in r['gold_reranked_positions'].values() if p]) else '1-5' if min(positions) <= 5 else '6-10' if min(positions) <= 10 else '11-20' if min(positions) <= 20 else '21-100' for r in inventory)),
    'regressions_by_coverage': dict(collections.Counter(r['coverage'] for r in inventory if r['dense_score'] > 0)),
    'native_recall_changes_on_zero_questions': {'improved': sum(r['native_recall5'] > r['qwen_recall5'] for r in inventory), 'worse': sum(r['native_recall5'] < r['qwen_recall5'] for r in inventory)},
}
assert len(inventory) == 121
out = ROOT / 'tmp/zero-diagnosis'
out.mkdir(exist_ok=True)
(out / 'all-zero-cases.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
(out / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
for state in ('none', 'partial', 'full'):
    lines = []
    for r in inventory:
        if r['coverage'] != state:
            continue
        lines.extend([f"CASE {r['query_id']} pool={r['pool100_positive_qrels']}/{r['positive_qrels']} pos={list(r['gold_reranked_positions'].values())} dense={r['dense_score']:.3f}",
                      'QUESTION '+r['question'], 'ANSWER '+r['answer'], 'JUDGE '+' '.join(r['judge_explanations']),
                      'GOLD '+json.dumps(r['gold_documents'], ensure_ascii=False), ''])
    (out / (state+'-dossier.txt')).write_text('\n'.join(lines), encoding='utf-8')
print(json.dumps(summary, indent=2))
for i, r in enumerate(inventory):
    print(f"{i+1:03} {r['query_id']} {r['coverage']} pool={r['pool100_positive_qrels']}/{r['positive_qrels']} dense={r['dense_score']:.3f} Q={r['question']} J={' '.join(r['judge_explanations'])}")
