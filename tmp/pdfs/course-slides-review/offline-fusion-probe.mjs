import fs from 'node:fs/promises';
import crypto from 'node:crypto';

const root = 'D:/Desktop/uni_of_edinburgh/year4/machine_learning_systems/cw/edin-mls-26-autumn';
const inputs = {
  dense: 'rag/results/diagnostic-retrieval-3713758/rankings.jsonl',
  reranked: 'rag/results/selection-pilot-3713795/plan/rerankings.jsonl',
  sample: 'rag/experiments/selection-pilot-sample.jsonl',
};
const sources = {};
const hashes = {};
for (const [key, relative] of Object.entries(inputs)) {
  const bytes = await fs.readFile(`${root}/${relative}`);
  hashes[relative] = crypto.createHash('sha256').update(bytes).digest('hex');
  sources[key] = bytes.toString('utf8').split(/\r?\n/).filter(Boolean).map(JSON.parse);
}
const original = new Map(sources.dense.map(row => [row.query_id, row]));
const reranked = new Map(sources.reranked.map(row => [row.query_id, row]));

// Selection consumes only candidate orders; qrels are read after selection.
function selectOrders(denseIds, rerankIds) {
  const denseRank = new Map(denseIds.map((id, index) => [id, index + 1]));
  const rerankRank = new Map(rerankIds.map((id, index) => [id, index + 1]));
  const rrf = [...denseIds].sort((a, b) => {
    const score = id => 1 / (60 + denseRank.get(id)) + 1 / (60 + rerankRank.get(id));
    return score(b) - score(a) || denseRank.get(a) - denseRank.get(b);
  });
  const protectedIds = denseIds.slice(0, 2);
  return {
    dense5: denseIds.slice(0, 5),
    rerank5: rerankIds.slice(0, 5),
    dense_ce_rrf_equal_c60: rrf.slice(0, 5),
    protect_dense2_fill_rerank3: [...protectedIds, ...rerankIds.filter(id => !protectedIds.includes(id))].slice(0, 5),
  };
}

const selected = sources.sample.map(({ query_id }) => {
  const dense = original.get(query_id);
  const ce = reranked.get(query_id);
  if (!dense || !ce || dense.record_ids.length !== 100 || ce.record_ids.length !== 100
      || new Set(dense.record_ids).size !== 100 || new Set(ce.record_ids).size !== 100
      || dense.record_ids.some(id => !ce.record_ids.includes(id))) throw new Error('Candidate mismatch');
  return { query_id, selections: selectOrders(dense.record_ids, ce.record_ids) };
});
const details = selected.map(row => {
  const relevant = new Set(Object.keys(original.get(row.query_id).positive_qrel_ranks));
  const recall = ids => ids.filter(id => relevant.has(id)).length / relevant.size;
  return { ...row, recall_at_5: Object.fromEntries(Object.entries(row.selections).map(([method, ids]) => [method, recall(ids)])) };
});
const methods = Object.fromEntries(Object.keys(details[0].recall_at_5).map(method => {
  const values = details.map(row => row.recall_at_5[method]);
  const deltas = details.map(row => row.recall_at_5[method] - row.recall_at_5.dense5);
  return [method, {
    mean_recall_at_5: values.reduce((a, b) => a + b, 0) / values.length,
    wins_vs_dense5: deltas.filter(x => x > 1e-12).length,
    ties_vs_dense5: deltas.filter(x => Math.abs(x) <= 1e-12).length,
    losses_vs_dense5: deltas.filter(x => x < -1e-12).length,
  }];
}));
if (Math.abs(methods.dense5.mean_recall_at_5 - 0.28833333333333333) > 1e-12
    || Math.abs(methods.rerank5.mean_recall_at_5 - 0.27166666666666667) > 1e-12) throw new Error('Pilot baseline mismatch');
const result = {
  queries: details.length, split: 'practice', status: 'post-hoc exploratory replay of original 30-query pilot',
  new_model_or_api_calls: 0, input_files_sha256: hashes, methods,
  limitations: ['No new answers, answer judging or timing measurement', 'Fixed policies chosen after viewing earlier results', 'No held-out validation or claim of answer-quality improvement'],
  per_query: details,
};
await fs.writeFile(`${root}/tmp/pdfs/course-slides-review/offline-fusion-probe.json`, JSON.stringify(result, null, 2));
console.log(JSON.stringify({ queries: result.queries, status: result.status, new_model_or_api_calls: 0, methods }, null, 2));
