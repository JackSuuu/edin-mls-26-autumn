"""Three-arm practice diagnostic: full Qwen top-5, prefix, scored passages.

prepare: GPU sentence-window scoring and immutable context plans.
generate/smoke/summarise: CPU orchestration of the unchanged ELM interfaces.
No qrels are loaded while constructing or scoring contexts.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'rag'))
from extensions.evidence_selection import chat_input_tokens, select_context
from extensions.passage_selection import make_windows, pack_windows
from extensions.evaluate_qwen_reranker import qwen_input

ARMS = ('full5', 'prefix768', 'passages768')
MODEL = 'utter-project/EuroLLM-22B-Instruct-2512'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')


def load(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def index(rows):
    result = {r['query_id']: r for r in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate query IDs')
    return result


def sources():
    return [Path(__file__), ROOT/'rag/extensions/passage_selection.py', ROOT/'rag/extensions/evidence_selection.py',
            ROOT/'rag/extensions/evaluate_qwen_reranker.py', ROOT/'rag/src/generator.py',
            ROOT/'rag/scripts/evaluate_answers.py', ROOT/'rag/src/evaluation/answer_judge.py']


def read_generator_system_prompt(source=None):
    """Read the literal prompt without importing API-only dotenv dependencies."""
    source = ROOT/'rag/src/generator.py' if source is None else Path(source)
    module = ast.parse(source.read_text(encoding='utf-8'))
    for node in module.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SYSTEM_PROMPT' for t in node.targets):
            value = ast.literal_eval(node.value)
            if isinstance(value, str) and value.strip():
                return value
    raise ValueError('Generator SYSTEM_PROMPT must be a nonempty literal string')


def prepare(args):
    SYSTEM_PROMPT = read_generator_system_prompt()
    print('PASSAGE_PREPARE_STAGE importing GPU runtime', flush=True)
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    if protocol['arms'] != list(ARMS) or protocol['input_budget'] != 768 or len(set(protocol['query_ids'])) != 24:
        raise ValueError('Unexpected frozen protocol')
    rankings_path = ROOT / protocol['rankings_path']
    if digest(rankings_path) != protocol['rankings_sha256']:
        raise ValueError('Frozen ranking hash mismatch')
    source_config = json.loads((rankings_path.parent/'config.json').read_text())
    corpus = ROOT/'rag/data/rag_corpus/corpus.jsonl'
    if digest(corpus) != source_config['input_sha256']['corpus']:
        raise ValueError('Corpus changed')
    if args.run.exists():
        raise ValueError('Choose a fresh run directory; do not overwrite partial scoring')
    if not torch.cuda.is_available():
        raise ValueError('Submit prepare through a GPU Slurm job')
    torch.set_num_threads(2)
    torch.manual_seed(20261117)
    records = {r['record_id']:r['text'] for r in load(corpus)}
    rankings = index(load(rankings_path))
    if not set(protocol['query_ids']) <= set(rankings):
        raise ValueError('Unknown sampled queries')
    gen_assets = ROOT/'rag/models/selection'
    token_manifest = json.loads((gen_assets/'manifest.json').read_text())['tokenizer']
    if token_manifest['repo'] != MODEL:
        raise ValueError('Generator tokenizer mismatch')
    for name, expected in token_manifest['files_sha256'].items():
        if digest(gen_assets/'tokenizer'/name) != expected:
            raise ValueError('Generator tokenizer hash mismatch')
    gen_tok = AutoTokenizer.from_pretrained(gen_assets/'tokenizer', local_files_only=True)
    def count_tokens(prompt):
        return chat_input_tokens(gen_tok, SYSTEM_PROMPT, prompt)
    model_dir = ROOT/'rag/models/qwen3-reranker-4b'
    manifest = json.loads((model_dir/'experiment_manifest.json').read_text())
    if manifest['repo'] != protocol['passage_scorer'] or manifest['revision'] != protocol['passage_scorer_revision']:
        raise ValueError('Passage scorer revision mismatch')
    print('PASSAGE_PREPARE_STAGE verifying cached scorer files', flush=True)
    for name, expected in manifest['files_sha256'].items():
        if digest(model_dir/name) != expected:
            raise ValueError('Passage scorer asset hash mismatch: '+name)
    print('PASSAGE_PREPARE_STAGE loading Qwen-4B', flush=True)
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, dtype=torch.bfloat16,
                                               attn_implementation='sdpa').eval().cuda()
    yes, no = [tok.convert_tokens_to_ids(w) for w in ('yes','no')]
    if tok.encode('yes',add_special_tokens=False) != [yes] or tok.encode('no',add_special_tokens=False) != [no]:
        raise ValueError('Decision token mismatch')
    @torch.inference_mode()
    def score(question, texts):
        scores, lengths = [], []
        for start in range(0,len(texts),protocol['batch_size']):
            encoded = [qwen_input(tok, question, text, max_tokens=protocol['passage_score_max_tokens'],
                                  forbid_truncation=True)[0] for text in texts[start:start+protocol['batch_size']]]
            lengths.extend(len(v['input_ids']) for v in encoded)
            inputs = tok.pad(encoded,padding=True,return_tensors='pt').to('cuda')
            logits = model(**inputs,use_cache=False,logits_to_keep=1).logits[:,-1,:].float()
            scores.extend((logits[:,yes]-logits[:,no]).cpu().tolist())
        return scores, lengths
    toy_q, toy_docs = 'What is the capital of France?', ['Paris is the capital of France.','Bananas are yellow fruit.']
    for _ in range(3):
        toy_scores, _ = score(toy_q,toy_docs)
    single = [score(toy_q,[d])[0][0] for d in toy_docs]
    if toy_scores[0] <= toy_scores[1] or any(abs(a-b)>0.25+0.02*abs(b) for a,b in zip(toy_scores,single)):
        raise ValueError('Scoring direction/padding sanity failed')
    args.run.mkdir(parents=True)
    plans, totals = [], []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with (args.run/'window_scores.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
        for i,qid in enumerate(protocol['query_ids'],1):
            row = rankings[qid]
            parent_ids = row['qwen_matched']['record_ids'][:5]
            question = row['transcript']
            built = {}
            for arm,budget in (('full5',None),('prefix768',protocol['input_budget'])):
                t0=time.perf_counter()
                built[arm] = select_context(question,parent_ids,records,count_tokens,top_k=5,budget=budget)
                built[arm]['context_build_ms']=(time.perf_counter()-t0)*1000
            torch.cuda.synchronize()
            t0=time.perf_counter()
            windows=make_windows(parent_ids,records,protocol['window_sentences'],protocol['window_overlap'])
            split_ms=(time.perf_counter()-t0)*1000
            torch.cuda.synchronize()
            t0=time.perf_counter()
            values,lengths=score(question,[w['text'] for w in windows])
            torch.cuda.synchronize()
            score_ms=(time.perf_counter()-t0)*1000
            t0=time.perf_counter()
            built['passages768']=pack_windows(question,parent_ids,records,windows,values,count_tokens,protocol['input_budget'])
            packing_ms=(time.perf_counter()-t0)*1000
            built['passages768'].update({'context_build_ms':split_ms+score_ms+packing_ms,
                                        'split_ms':split_ms,'passage_scoring_ms':score_ms,'packing_ms':packing_ms})
            stream.write(json.dumps({'query_id':qid,'parent_record_ids':parent_ids,'windows':[
                {**w,'score':s,'scorer_input_tokens':n} for w,s,n in zip(windows,values,lengths)],
                'passage_scoring_ms':score_ms},ensure_ascii=False)+'\n')
            stream.flush()
            order=list(ARMS)
            random.Random(protocol['arm_order_seed']+':'+qid).shuffle(order)
            for arm in order:
                plan={'query_id':qid,'split':'practice','transcript':question,'arm':arm,'parent_record_ids':parent_ids,
                      'cached_reranking_ms':row['qwen_matched']['rerank_ms'],**built[arm]}
                plan['prompt_sha256']=hashlib.sha256(plan['prompt'].encode()).hexdigest()
                if not plan['retrieved_record_ids'] or plan['local_input_tokens']>28672:
                    raise ValueError('Empty evidence or input overflow before API calls')
                plans.append(plan)
            totals.append({'query_id':qid,'windows':len(windows),'input_tokens':{a:built[a]['local_input_tokens'] for a in ARMS},
                           'passage_scoring_ms':score_ms,'same_prefix_passage_prompt':built['prefix768']['prompt']==built['passages768']['prompt']})
            print(f"Prepared [{i}/24] {qid} windows={len(windows)} tokens="+
                  '/'.join(str(built[a]['local_input_tokens']) for a in ARMS),flush=True)
    config={'schema':'passage-pilot-run-v1','protocol':protocol,'protocol_sha256':digest(args.protocol),
            'source_sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in sources()},
            'input_sha256':{'rankings':digest(rankings_path),'corpus':digest(corpus)},'system_prompt':SYSTEM_PROMPT,
            'tokenizer':token_manifest,'scorer_manifest':manifest,'device':torch.cuda.get_device_name(),
            'torch':torch.__version__,'transformers':transformers.__version__,'scorer_dtype':'bfloat16',
            'qrels_not_loaded_in_selection':True,'no_automatic_full_split_expansion':True,
            'timing_scope':'Warmed additional passage scoring and context packing; cached parent reranking attributed; not end-to-end latency'}
    save(args.run/'config.json',config)
    with (args.run/'plans.jsonl').open('x',encoding='utf-8',newline='\n') as f:
        for p in plans:
            f.write(json.dumps(p,ensure_ascii=False)+'\n')
    save(args.run/'context_summary.json',{'queries':24,'plans':len(plans),'per_query':totals,
        'peak_allocated_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'sanity_batched':toy_scores,'sanity_single':single})
    save(args.run/'prepared.json',{p.name:digest(p) for p in [args.run/'config.json',args.run/'plans.jsonl',args.run/'window_scores.jsonl']})
    print('PASSAGE_PILOT_PLANS_READY',flush=True)


def check(run):
    for name,expected in json.loads((run/'prepared.json').read_text()).items():
        if digest(run/name)!=expected:
            raise ValueError('Frozen artifact changed: '+name)
    config=json.loads((run/'config.json').read_text())
    for name,expected in config['source_sha256'].items():
        if digest(ROOT/name)!=expected:
            raise ValueError('Source changed: '+name)
    plans=load(run/'plans.jsonl')
    ids=config['protocol']['query_ids']
    if len(plans)!=72 or {(r['query_id'],r['arm']) for r in plans}!={(q,a) for q in ids for a in ARMS}:
        raise ValueError('Incomplete frozen plans')
    return config,plans


def generate(args):
    config,plans=check(args.run)
    from dotenv import load_dotenv
    load_dotenv(ROOT/'rag/.env')
    from src.generator import ELMGenerator
    gen=ELMGenerator()
    if gen.model!=MODEL:
        raise ValueError('Generator differs from frozen model')
    out=args.run/'answers'
    fingerprint={'config_sha256':digest(args.run/'config.json'),'plans_sha256':digest(args.run/'plans.jsonl'),
                 'model':gen.model,'endpoint':str(gen.client.base_url),'temperature':0.1}
    done={}
    if out.exists():
        if not args.resume or json.loads((out/'generation_config.json').read_text())!=fingerprint:
            raise ValueError('Resume requires identical generation fingerprint')
    else:
        if args.resume:
            raise ValueError('No generation to resume')
        out.mkdir()
        save(out/'generation_config.json',fingerprint)
    for arm in ARMS:
        path=out/(arm+'.jsonl')
        for r in load(path) if path.exists() else []:
            key=(r['query_id'],r['arm'])
            if key in done or r['arm']!=arm or not r['answer'].strip() or r['finish_reason']!='stop':
                raise ValueError('Saved answer needs inspection')
            done[key]=r
    expected={(p['query_id'],p['arm']):p for p in plans}
    for key,r in done.items():
        if key not in expected or r['prompt_sha256']!=expected[key]['prompt_sha256']:
            raise ValueError('Saved answer does not match plan')
    pending=[q for q in config['protocol']['query_ids'] if any((q,a) not in done for a in ARMS)]
    allowed=set(pending[:args.max_new_queries] if args.max_new_queries else pending)
    for plan in plans:
        key=(plan['query_id'],plan['arm'])
        if key in done or key[0] not in allowed:
            continue
        value=gen.generate(plan['prompt'])
        if not value['answer'].strip():
            raise ValueError('Empty answer; stopping')
        row={**plan,**value,'generation_ms':value['latency_ms'],
             'attributed_selection_plus_generation_ms':plan['cached_reranking_ms']+plan['context_build_ms']+value['latency_ms']}
        with (out/(plan['arm']+'.jsonl')).open('a',encoding='utf-8',newline='\n') as f:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
        done[key]=row
        print(f"Generated [{len(done)}/72] {key[0]} {key[1]} tokens={plan['local_input_tokens']} finish={value['finish_reason']}",flush=True)
        usage=value.get('usage') or {}
        if value['finish_reason']!='stop' or usage.get('prompt_tokens')!=plan['local_input_tokens']:
            raise ValueError('Completion or server token count mismatch; inspect before continuing')
    save(out/'generation_summary.json',{'completed':len(done)==72,'generated':len(done),'expected':72})


def smoke(args):
    config,_=check(args.run)
    ids=config['protocol']['query_ids'][:2]
    for arm in ARMS:
        answers=index(load(args.run/'answers'/(arm+'.jsonl')))
        with (args.run/'answers'/('smoke_'+arm+'.jsonl')).open('w',encoding='utf-8',newline='\n') as f:
            for q in ids:
                f.write(json.dumps(answers[q],ensure_ascii=False)+'\n')


def summarise(args):
    config,plans=check(args.run)
    ids=config['protocol']['query_ids']
    judged={}
    result={'status':'completed_exploratory_24_query_pilot','queries':len(ids),'arms':{},'paired':{},
            'limitations':config['protocol']['limitations']+['Single generation and judgement per answer; same generator and judge model',
              'Query bootstrap does not include generation/judge repeat uncertainty; score-conditioned sample is not a benchmark gain',
              'Attributed stage costs are not measured audio-to-answer latency']}
    for arm in ARMS:
        answers=index(load(args.run/'answers'/(arm+'.jsonl')))
        judged[arm]=index(load(args.run/('judge_json_'+arm)/'answer_judge_per_query.jsonl'))
        s=json.loads((args.run/('judge_json_'+arm)/'answer_judge_summary.json').read_text())
        if set(answers)!=set(ids) or set(judged[arm])!=set(ids) or s['judge_model']!=MODEL or s['repeats']!=1 or s['judge_response_format']!='json_object':
            raise ValueError('Incomplete or incompatible judge outputs')
        for q in ids:
            if abs(sum(judged[arm][q][d] for d in ('correctness','grounding','completeness'))/12-judged[arm][q]['normalised_answer_score'])>1e-12:
                raise ValueError('Score normalisation mismatch')
        values=list(answers.values())
        result['arms'][arm]={k:v for k,v in s.items() if k.startswith('mean_')}
        result['arms'][arm].update({'all_zero':sum(judged[arm][q]['normalised_answer_score']==0 for q in ids),
            'mean_input_tokens':statistics.mean(r['local_input_tokens'] for r in values),
            'mean_output_tokens':statistics.mean(r['usage']['completion_tokens'] for r in values),
            'mean_generation_ms':statistics.mean(r['generation_ms'] for r in values),
            'median_generation_ms':statistics.median(r['generation_ms'] for r in values),
            'mean_context_build_ms':statistics.mean(r['context_build_ms'] for r in values),
            'mean_attributed_selection_plus_generation_ms':statistics.mean(r['attributed_selection_plus_generation_ms'] for r in values)})
    rng=random.Random(config['protocol']['bootstrap_seed'])
    samples=[rng.choices(ids,k=len(ids)) for _ in range(config['protocol']['bootstrap_replicates'])]
    for new,base in [('prefix768','full5'),('passages768','full5'),('passages768','prefix768')]:
        result['paired'][new+'_minus_'+base]={}
        for metric in ('correctness','grounding','completeness','normalised_answer_score'):
            delta={q:judged[new][q][metric]-judged[base][q][metric] for q in ids}
            estimates=sorted(statistics.mean(delta[q] for q in sample) for sample in samples)
            result['paired'][new+'_minus_'+base][metric]={'mean_delta':statistics.mean(delta.values()),
                'query_bootstrap_95pct_interval':[estimates[50],estimates[1949]],
                'improved':sum(v>1e-12 for v in delta.values()),'worse':sum(v < -1e-12 for v in delta.values()),
                'tied':sum(abs(v)<=1e-12 for v in delta.values())}
    save(args.run/'comparison.json',result)
    print(json.dumps(result,indent=2),flush=True)
    print('PASSAGE_PILOT_COMPLETE',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['prepare','generate','smoke','summarise'])
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--protocol',type=Path,default=ROOT/'rag/experiments/passage-pilot-v1.json')
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--max-new-queries',type=int)
    args=ap.parse_args()
    if args.max_new_queries is not None and args.max_new_queries<1:
        ap.error('max-new-queries must be positive')
    {'prepare':prepare,'generate':generate,'smoke':smoke,'summarise':summarise}[args.stage](args)


if __name__=='__main__':
    main()
