# Why 121 Qwen-4B answers received zero

## Scope and main finding

This offline diagnosis concerns the retained job `3724290`: 350 practice
questions, frozen ASR transcripts, Qwen-4B matched-prefix reranking and five
full records supplied to EuroLLM. The original
[answer report](qwen4-answer-evaluation.md) records the controlled comparison.
No new generation, judging or ASR calls were made. Original answers, scores and
evaluation code remain unchanged.

The 121 zeros have several distinct failure mechanisms. Missing labelled
evidence affects some cases; other cases contain relevant evidence that the
generator misuses or ignores. Direct source comparisons also establish errors
in some judge explanations and ambiguities in some references. These mechanisms
can coexist within one question. We cannot assign an exclusive causal diagnosis
to every zero from a single generation and judgement.

The [inventory](../results/qwen4-answers-3724290/zero-diagnosis-v1/cases.jsonl)
contains all 121 questions, both answers and judgements, selected and gold
texts, and relevant-document positions. The
[summary](../results/qwen4-answers-3724290/zero-diagnosis-v1/summary.json)
records source hashes and reproducible counts. The assistant read all 121
questions and judge explanations, then compared full source texts for all 30
full-qrel-coverage zeros and eight additional targeted cases. The
[source-review notes](../results/qwen4-answers-3724290/zero-diagnosis-v1/source-review.json)
record those 38 comparisons. This is AI-assisted, score-conditioned review,
not independent blinded human validation. The other 83 cases have a mechanical
inventory and an explanation review, not an exhaustive source-based diagnosis.

## Where labelled evidence is lost

Here, a qrel is a positive document-relevance label supplied with the coursework.
Its presence establishes document-ID coverage, not sufficient information or
correct applicability.

| Observation within the 121 zeros | Questions | What it establishes |
| --- | ---: | --- |
| No positive qrel in dense top 100 | 32 | Reranking the same candidate pool cannot recover a labelled document |
| Positive qrel in top 100, none selected in Qwen top 5 | 30 | The selection omits available labelled evidence |
| Some, but not all, positive qrels selected | 29 | Labelled coverage is partial; the missing records may or may not add necessary information |
| All positive qrels selected | 30 | Missing labelled document IDs cannot by itself explain these zeros |

For the 30 cases with gold in the pool but none selected, the first positive
qrel appears at ranks 6-10 in 12 cases, 11-20 in nine and 21-100 in nine.
Increasing context to ten would therefore expose a labelled document in 12 of
these cases, but does not establish that their answers would improve. Earlier
dense20 results already caution against equating more coverage with better
answers. Adding documents can also add distracting or incompatible material.

Across all 350 questions, the zero rate is **62/104 = 59.6%** without a top-5
qrel hit, **29/115 = 25.2%** with partial coverage and **30/131 = 22.9%** with
full coverage. These are descriptive associations, not causal effects. In
particular, 42 questions without a positive-qrel hit still receive a nonzero
score. The labels do not exhaust every potentially useful passage.

Of the 36 positive-to-zero regressions relative to dense5, 16 have no selected
qrel, seven partial coverage and 13 full coverage. Retrieval-label coverage
alone therefore cannot explain all regressions either.

## Verified evidence-use failures

### Combining text from incompatible situations

For `q_371c6ca10bf1bf5e`, about an Irish pension account, both gold records are
at ranks 1 and 2. The answer includes their legal/tax benefits, but then says
pension contribution limits can be carried forward and offset against future
profits. Selected non-qrel record `r_dc4b5d2c6048003d` discusses a **corporate
tax loss** carried forward against future company profits. The generator
transfers that rule to pension contributions. Another selected record supplies
the answer's mental-partitioning point.

This is a concrete applicability error across retrieved texts. A claim can be
copied from a document and still be wrong for the question. The baseline
[context builder](../src/context_builder.py) concatenates five complete records;
the [generation instruction](../src/generator.py) asks for evidence support but
does not explicitly require separating account types, jurisdictions, roles or
conditions. This code behaviour is consistent with the observed failure, but
a prompt/context control is needed to establish whether changing it helps.

The same pattern appears in other reviewed cases:

- `q_9cd791a970ab648f`: a credit-card-check availability question acquires online
  merchant authorization/capture timing from non-qrel `r_52061f3956b4c32a`.
- `q_45ce8f0db1865a56`: an Indian salary/PF reference is displaced by Illinois
  W-4 and US Social Security/Medicare instructions from other selected records.
- `q_8706e2925ba6d565`: the question asks about lending money at a prime-linked
  rate. Gold at rank 2 discusses savings/CDs; the answer instead explains how
  to obtain a personal loan and collateral, reversing borrower and lender roles.
- `q_2803366d84526278`: the answer begins with the gold transfer/business-tax
  distinction, then adds Help to Buy ISA and foreign-remittance advice from
  other selected situations.

These examples support testing relevance and applicability constraints during
generation. They do not establish the prevalence of this mechanism among all
121 zeros, nor prove that the remaining claims should receive zero credit.

### Ignoring a directly supplied answer

For `q_a346997622ff7b7e`, gold is the first record in the actual generator
context. It describes merging duplicate UANs, transferring EPF and an online
route. The candidate nevertheless says none of the documents addresses two
UANs linked to one PAN. Dense5 received 1.0 for this question. The change in
context and the single stochastic generation both remain possible contributors.

For `q_441d4eaaf425ff98`, gold at rank 1 starts with an explicit negative answer
about whether a second-mortgage settlement counts as a short sale. The candidate
incorrectly declares insufficient evidence. Its judge explanation also misses
that explicit answer. Generator and judge errors can therefore coexist.

## Judge and reference problems

The [existing audit](../results/qwen4-answers-3724290/answer-evaluation-audit.json)
already verifies that scores correspond to the intended candidate, gold, cache
key and raw response. The zeros are parsed model judgements, not values filled
in after failed API calls. In the new inventory, all 121 answers are nonempty,
finish with `stop`, use 416-2,500 measured input tokens, and contain no recorded
context truncation. These observations do not support a simple missing-output
or context-truncation explanation.

### Incorrect statements about supplied inputs

Several saved judge explanations contradict the texts they evaluate:

- `q_148262b1f4e5ab84`: the judge calls Orbitz contract flexibility and possible
  Chase recourse unsupported, although both appear explicitly in the sole gold
  record. The candidate still has an apparent site/entity attribution problem;
  the incorrect explanation does not make it wholly correct.
- `q_890a6df9ce937b08`: both gold and candidate say the MIR lacks issuance time.
  The judge calls this a contradiction and says the candidate omitted that
  point. The candidate's broader impossibility statement still needs scrutiny.
- `q_039f48f42921b2be`: gold explicitly discusses Consumer Reports and calling
  other dealers, contrary to the judge. It also attributes a salesperson-authority
  statement to the candidate that the candidate did not make.
- `q_41e3c12e4d573cb5`: candidate and selected records discuss news services;
  gold discusses bank transfers/AML. The judge attributes the banking/AML
  content to the candidate, reversing the inputs.
- `q_9a684db4ff009865`: the Qwen candidate explicitly states a settlement wait
  under its cash-account condition; the judge says it recommends immediate
  reuse without waiting. This concerns the saved historical reference, not
  a claim about current settlement rules.

The source-review notes identify concrete input-misreading flags for further
checking. Those flags are observations about explanations. They are not
replacement scores, an independent judge error-rate estimate, or evidence that
every flagged candidate deserves a high score.

### Partial credit, refusals and conflicting references

Across the full 350-query Qwen arm, correctness is zero on exactly 121 questions,
grounding on exactly 121, and completeness on exactly 121: they are the same
all-zero cases. There are no single-dimension zeros. This clustering, together
with the inspected contradictions, warrants rubric calibration. It does not
establish why the judge produces that pattern.

For `q_8de6d5607824f3c0`, gold does not supply the requested numerical oil
return statistics. The judge says the refusal correctly identifies the missing
information and is grounded, but outputs grounding 0. For
`q_33e70afa6a74be26`, it similarly recognises the absence of Walmart-specific
evidence and describes the refusal as grounded, yet emits grounding 0. The
[rubric](../src/evaluation/answer_judge.py) explicitly permits a qualified
insufficiency response when gold itself is insufficient. These cases need
independent rubric-based review; their precise replacement scores are unknown.

For `q_b7390fb1b4a9ad79`, one gold record says CAGR should be read letter by
letter, while another says it rhymes with "bagger". Both are selected. The
candidate follows the first; the judge treats the second as authoritative.
Conflict handling belongs in both answer analysis and evaluation interpretation.

## Connection to ASR and question construction

Some supplied questions are much less specific than their labelled answers.
For `q_dd94af61414916d5`, the entire transcript is "What information should I
research?", while gold expects an Essex Chemical/Dow acquisition answer with
deal-specific figures. No company or transaction appears in the transcript.
For `q_41e3c12e4d573cb5`, the news-wire question and money-transfer gold imply
different intents. These are observed question/reference gaps. Their origin
could be the spoken question, transcription or dataset construction; the asset
manifest supplies no clean reference transcript to distinguish them.

Two other cases motivate a more focused ASR/RAG boundary test:

- `q_c5a5320bcfd3a373` uses "four hundred three B" and "four hundred one K".
  Gold concerns a 403b-to-401k retirement rollover and is absent from top 100;
  the retrieved documents concern generic bank/PayPal transfers instead.
- `q_6f16eb6280a1f38c` says HSD while the first selected record concerns HST.
  The candidate recognises the expanded tax term but refuses over the acronym
  mismatch.

Listen to the original audio before calling either an ASR recognition error.
Even a correctly recognised spoken form may need normalisation for retrieval.
Test a general spoken-number/acronym normaliser, not query-ID-specific repairs
derived from qrels. Measure its effect on retrieval and final answers separately
from word error rate. It cannot restore contextual details never present in
the audio.

## Next controlled tests

### Additional offline checks: fusion and document position

The [cached-ranking controls](../results/qwen4-answers-3724290/zero-diagnosis-v1/selection-controls-v2.json)
evaluate two fixed alternatives without model calls or a parameter sweep.
The [runner](../extensions/explore_qwen_selection_controls.py) constructs rankings
without qrels, then uses qrels only to evaluate them. This is exploratory
practice analysis after inspecting failures, not a preregistered comparison.

- Equal reciprocal-rank fusion of dense and Qwen, using a rank constant of 60,
  gives Recall@5 **0.501158**, MRR@10 **0.558525** and NDCG@10 **0.497306**.
  The respective Qwen-only values are **0.526226**, **0.595635** and **0.534080**.
  Recall@5 improves on 29 questions, declines on 42 and ties on 279. Within the
  original 121-zero subset, it introduces a qrel hit in six previously no-hit
  contexts but removes the last qrel hit in ten previously hit contexts.
- Keeping Qwen's first four documents and reserving the fifth slot for dense's
  first document, if absent, gives Recall@5 **0.522655**, MRR@10 **0.594967** and
  NDCG@10 **0.533427**. Recall@5 improves on three questions, declines on three
  and ties on 344. It does not change hit/no-hit status within the zero subset.

These controls do not support prioritising these two simple fusion policies
over Qwen-only ranking on the observed retrieval metrics. Neither produces new
answers, so neither establishes an answer-quality improvement or decline.

In **38 of the 121 zeros**, the first document already has a positive qrel.
This includes **20 of the 30 full-coverage zeros**. Moving relevant evidence to
the beginning cannot by itself explain or repair these cases: it is already
there. The observation does not establish that the first document answers the
whole question or that a top-1 context would fail. A top-1 generation control
still tests the effect of removing the other four documents.

### Deployable alternatives and their limits

**Condition-aware evidence extraction is the first generation-side candidate.**
Keep the existing Qwen top-5 IDs fixed. Ask the generator to identify the
requested action, account/instrument type, jurisdiction and borrower/lender
role when stated; leave missing attributes unknown. Require a short list of
applicable source quotes with document IDs and relevant conditions, followed
by the final answer. Check mechanically that each quote occurs in its cited
selected record, and score only the final-answer field. Quote presence alone
does not establish that the claim applies to the question. This can start as
one structured generation call, with its added output tokens measured; a
separate verification call is a later, more expensive ablation.

For example, an Irish-pension answer should not reuse a company's tax-loss
carry-forward rule merely because both mention tax benefits. A useful extraction
must retain what the rule applies to. This differs from ranking entire records:
the record-level reranker estimates relevance, while the extraction/answer step
must preserve the conditions of each particular claim. We are adapting existing
retrieval and prompting methods, not claiming a new algorithm.

**Passage selection under a fixed maximum input budget is a stronger context
alternative.** Split retrieved records into short overlapping sentence windows,
score question/window pairs, and select passages while retaining neighbouring
qualifications. Compare against document-prefix packing under the same maximum
input budget and generation instruction. Measure actual tokens, since equal
caps do not make realised input lengths identical. This changes selection
granularity rather than merely changing the reranker model. It may lose
necessary context or add reranking cost, so do not assume it is an improvement.

**Answer verification targets residual claim errors.** A second call can check
each claim against its quoted text and applicability conditions, removing or
qualifying unsupported statements. It must check roles and scope, not only
whether a citation exists. The extra call increases latency and API usage; a
same-model verifier may repeat the generator's mistake. Compare it separately
after the single-call extraction variant rather than bundling both changes.

**Spoken-form normalisation targets first-stage retrieval.** Normalise financial
account names and acronyms conservatively, retaining the original transcript
alongside the canonical form. Compare raw, normalised and combined-query
retrieval using the same candidate count. Verify proposed corrections against
the audio on a fixed sample; do not repair individual query IDs using qrels.
This can address a 403b/401k representation mismatch, but cannot reconstruct an
omitted company name, country or transaction. Do not infer those facts merely
to match a reference answer.

### Recommended sequence

1. **Check scoring reliability before interpreting small gains.** Use a fixed
   diagnostic set with known input contradictions, justified refusals and
   positive controls. Obtain independent human rubric-based ratings and, if
   available, a separate judge. Record per-dimension disagreements. Keep these
   results separate from the unchanged official scores. The all-zero JSON
   example in the current judge prompt is a possible calibration confound to
   test separately; no evidence here establishes an anchoring effect.
2. **Test whether distracting records cause failures.** Reuse the already frozen
   24-question diagnostic sample. Compare saved Qwen top-5 answers with new
   answers using gold-only records under the same generation prompt. This is
   an oracle-context diagnostic, unavailable at inference and not a strict
   upper bound. It removes non-qrel material without changing the generator.
   Also compare deployable Qwen top-1 with top-5 under the same prompt, measuring
   tokens and missing coverage. Do not call either comparison token-matched.
3. **Test one deployable generation change.** Keep Qwen top-5 fixed and change
   only the instruction: answer the requested issue first, retain account,
   jurisdiction and borrower/lender conditions, cite the supporting document,
   identify conflicting evidence, and avoid adding advice from unrelated
   situations. Require a concise supported answer and a specific statement of
   any missing information. This is a prompt-only ablation; it does not use qrels
   or require another reranker. Citations alone do not prove applicability.
4. **Then address first-stage misses.** For the 32 zero cases with no labelled
   document in top 100, test spoken-form normalisation or a revised candidate
   retriever. Another reranker over the unchanged pool cannot recover those
   labelled IDs. Previous aggregate BM25 results do not establish whether a
   specific normalisation helps these cases. Freeze one candidate-generation
   change and evaluate it on the full set before claiming improvement.

If a prompt/context pilot supports one deployable change, freeze it and compare
on all 350 questions with the original generator and official evaluator. Report
zero counts, zero-to-positive and positive-to-zero transitions, dimension means,
paired intervals, retrieval, input/output tokens and measured latency. The
score-conditioned pilot is for diagnosis, not a benchmark improvement claim;
retesting practice after tuning also does not establish held-out generalisation.
Retain an untouched split for final evaluation where the course protocol allows.

Mean dimension scores of 2 or 3 remain targets, not predicted outcomes. Better
retrieval, better evidence use and better measurement require separate evidence.

## Reproduction

Run the [offline inventory script](../extensions/diagnose_qwen_zero_answers.py)
from the repository root, with the course asset ZIP present. Choose a fresh
output directory; the script refuses to overwrite a frozen snapshot:

```sh
python rag/extensions/diagnose_qwen_zero_answers.py --out-dir /tmp/qwen-zero-reproduction
```

The inventory validates full query coverage, score normalisation and selected
document IDs, and checks that it reproduces the frozen 121 zeros and 36
regressions. Source-review notes are separate interpretive annotations; running
the script does not reproduce an independent semantic review.
