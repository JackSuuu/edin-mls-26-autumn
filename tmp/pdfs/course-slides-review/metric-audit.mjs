import fs from 'node:fs/promises';
import crypto from 'node:crypto';

const root = 'D:/Desktop/uni_of_edinburgh/year4/machine_learning_systems/cw/edin-mls-26-autumn';
const paths = {
  dense: 'rag/results/diagnostic-retrieval-3713758/rankings.jsonl',
  rerank: 'rag/results/selection-pilot-3713795/plan/rerankings.jsonl',
  sample: 'rag/experiments/selection-pilot-sample.jsonl',
  baseline: 'rag/results/diagnostic-retrieval-3713758/summary.json',
  practice: 'rag/results/selection-run-full/comparison.json',
  dev: 'rag/results/dev-validation/comparison.json',
};
const data = {}, hashes = {};
for (const [name, relative] of Object.entries(paths)) {
  const bytes = await fs.readFile(`${root}/${relative}`);
  hashes[relative] = crypto.createHash('sha256').update(bytes).digest('hex');
  data[name] = relative.endsWith('.jsonl')
    ? bytes.toString('utf8').split(/\r?\n/).filter(Boolean).map(JSON.parse)
    : JSON.parse(bytes.toString('utf8'));
}
// All 793 positive practice qrels have grade 1, independently checked in the archive.
function metrics(ids, relevant) {
  const recall = k => ids.slice(0, k).filter(id => relevant.has(id)).length / relevant.size;
  const first = ids.slice(0, 10).findIndex(id => relevant.has(id));
  const dcg = ids.slice(0, 10).reduce((sum, id, index) => sum + (relevant.has(id) ? 1 / Math.log2(index + 2) : 0), 0);
  const ideal = Array.from({ length: Math.min(relevant.size, 10) }, (_, i) => 1 / Math.log2(i + 2)).reduce((a, b) => a + b, 0);
  return { 'Recall@5': recall(5), 'Recall@10': recall(10), 'Recall@20': recall(20), 'Recall@100': recall(100),
    'MRR@10': first < 0 ? 0 : 1 / (first + 1), 'NDCG@10': dcg / ideal };
}
function aggregate(rows) {
  return Object.fromEntries(Object.keys(rows[0]).map(key => [key, rows.reduce((sum, row) => sum + row[key], 0) / rows.length]));
}
const dense = new Map(data.dense.map(row => [row.query_id, row]));
const ce = new Map(data.rerank.map(row => [row.query_id, row]));
const full = aggregate(data.dense.map(row => metrics(row.record_ids, new Set(Object.keys(row.positive_qrel_ranks)))));
for (const [key, value] of Object.entries(full)) {
  if (Math.abs(value - data.baseline.metrics[key]) > 1e-12) throw new Error(`Baseline reproduction failed: ${key}`);
}
const pilotRows = {};
for (const { query_id } of data.sample) {
  const original = dense.get(query_id), ranked = ce.get(query_id);
  if (!original || !ranked || original.record_ids.length !== 100 || new Set(ranked.record_ids).size !== 100
      || original.record_ids.some(id => !ranked.record_ids.includes(id))) throw new Error('Candidate mismatch');
  const dr = new Map(original.record_ids.map((id, index) => [id, index + 1]));
  const cr = new Map(ranked.record_ids.map((id, index) => [id, index + 1]));
  const rrfScore = id => 1 / (60 + dr.get(id)) + 1 / (60 + cr.get(id));
  const fused = [...original.record_ids].sort((a, b) => rrfScore(b) - rrfScore(a) || dr.get(a) - dr.get(b));
  const protectedIds = original.record_ids.slice(0, 2);
  // No relevance labels are consumed until all orders have been constructed.
  const orders = { dense: original.record_ids, cross_encoder: ranked.record_ids, dense_ce_rrf_equal_c60: fused,
    protect_dense2_then_cross_encoder: [...protectedIds, ...ranked.record_ids.filter(id => !protectedIds.includes(id))] };
  const relevant = new Set(Object.keys(original.positive_qrel_ranks));
  for (const [method, ids] of Object.entries(orders)) (pilotRows[method] ??= []).push(metrics(ids, relevant));
}
const pilot = Object.fromEntries(Object.entries(pilotRows).map(([method, rows]) => [method, aggregate(rows)]));
const costs = {};
for (const split of ['practice', 'dev']) {
  const arms = data[split].per_arm;
  costs[split] = {
    dense_budget_input_reduction_vs_dense20: 1 - arms.dense_budget.mean_server_input_tokens / arms.dense20.mean_server_input_tokens,
    dense20_input_multiplier_vs_dense5: arms.dense20.mean_server_input_tokens / arms.dense5.mean_server_input_tokens,
    dense_budget_mean_answer_score: arms.dense_budget.mean_judge_normalised_answer_score,
    dense20_mean_answer_score: arms.dense20.mean_judge_normalised_answer_score,
    dense5_generation_median_ms: arms.dense5.median_generation_ms,
    rerank5_generation_median_ms: arms.rerank5.median_generation_ms,
    dense5_attributed_selection_generation_median_ms: arms.dense5.median_attributed_selection_plus_generation_ms,
    rerank5_attributed_selection_generation_median_ms: arms.rerank5.median_attributed_selection_plus_generation_ms,
  };
}
const result = { source_hashes: hashes, new_model_or_api_calls: 0,
  full_practice_dense_baseline: { queries: data.dense.length, metrics: full, matches_saved_baseline: true },
  original_pilot_supplement: { queries: data.sample.length, metrics: pilot }, costs,
  limitations: ['30-query metrics must not be substituted for full 350-query or 125-query results',
    'Fusion policies are post-hoc exploratory analyses', 'No answer generation, new judging or new latency measurement',
    'Attributed selection-plus-generation medians are not measured audio-to-answer latency',
    'No equivalence or stable speedup claim follows from similar means or single-pass API timing'] };
await fs.mkdir(`${root}/rag/results/multimetric-audit`, { recursive: true });
await fs.writeFile(`${root}/rag/results/multimetric-audit/summary.json`, JSON.stringify(result, null, 2));
console.log(JSON.stringify({ full_practice_dense_baseline: result.full_practice_dense_baseline, original_pilot_supplement: result.original_pilot_supplement, costs }, null, 2));
