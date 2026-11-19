# Document chunking at a 768-token input ceiling

Another sample from the existing course practice split, not a new test set.
The run excludes all 24 queries from `passage-pilot-v1.json`; all 72 answers
and judgements are fresh. A fixed random order was screened by reranked
document position and A's prompt coverage, without consulting answer scores.
The first 12 eligible queries per stratum were retained (211 screened).
See `rag/results/document-chunking-768-v1/` for plans, exact prompts and
answers, judge results, rankings, source offsets, and paired differences.

A uses the document-ranked top five and packs prefixes; B scores overlapping
three-sentence windows from those same documents; C scores windows across the
same dense top 20, ranks parents by their highest-scoring window, and packs
using B's rule. All use the same Qwen-4B scorer, EuroLLM generator and JSON
judge. Scoring inputs are capped at 2,048 Qwen tokens including its template;
this is separate from the 768-token generator input ceiling. The server
confirmed the local input count for all 72 prompts. Labelled source presence
is not a measure of sufficient evidence or correct evidence use.

| Practice stratum | Arm | Labelled document in top five | Labelled text in prompt | Mean answer score | Mean input tokens | Mean document / passage scoring (ms) | Mean API generation (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Evidence already in A prompt | A | 12/12 | 12/12 | .625 | 734.75 | 1288 / — | 3566 |
| | B | 12/12 | 12/12 | .590 | 727.92 | 1288 / 441 | 3989 |
| | C | 12/12 | 12/12 | .493 | 755.83 | — / 1758 | 3799 |
| Labelled evidence outside document-ranked top five | A | 0/12 | 0/12 | .132 | 754.75 | 1362 / — | 4253 |
| | B | 0/12 | 0/12 | .181 | 741.75 | 1362 / 573 | 4922 |
| | C | 3/12 | 3/12 | .278 | 742.67 | — / 2060 | 4221 |

B reuses A's document ranking; its document time is attributed. C's passage
ranking does not need the screening document-scoring step at deployment.
Timings are stage observations, not audio-to-answer times or causal speedups.
B−A on the first stratum is −.0347 (2 wins, 5 losses, 5 ties); C−B on the
second is +.0972 (4 wins, 1 loss, 7 ties). See `summary.json` for all 24
individual differences.

## Trace the evidence, not just the document IDs

**Already present — prime-rate lending** (`q_8706e2925ba6d565`,
`r_15e1953945b06f0d`). The source and A's actual prompt say:
“In the US these are called certificates of deposit or savings accounts.”
B kept the parent at rank 2, but its actual excerpt begins “Every
run-of-the-mill bank offers them.” The US scope and the explicit antecedent
are missing at that point, although the retained excerpt later mentions CDs
and savings accounts and retains the prime-rate explanation. A's answer
calls CDs or savings accounts the closest way to earn such a rate; B also
mentions CDs, but leads with becoming a “hot-shot Wall Street money manager,”
confusing the lender and borrower roles. The score falls from .500 to .000.
This shows a loss of useful framing with the label still present, not proof
that the omitted sentence alone caused the answer change.

**Outside the top five — buying a home** (`q_d1397ae30111620c`,
`r_623105da59cf298f`). The parent moves from document rank 8 to chunk-based
rank 3. B has no text from it. C's prompt contains the original span
“Make and stick to a budget. Start building a down payment and emergency
fund. Pay down consumer debt/student loans.” C's answer explicitly includes
the emergency fund, down payment and debt repayment; B's answer discusses
setting a budget but omits that combination. The judge score moves from
.417 to .583. Other selected evidence also changed, so this trace shows
transfer in one case, not an isolated causal effect of those sentences.

For the other two recovered labelled parents, the W-4 example moves a
parent from rank 7 to 2 and sends the withholding/default-deductions text,
but the score remains .417. The credit/debit account example moves a
parent from rank 10 to 4 and sends the business-perspective definition;
the answer mentions assets and liabilities and scores .417 versus B's .000,
but B already described that basic distinction. Thus two of the three
labelled-prompt recoveries score higher, while two additional wins occur
without any labelled document entering the prompt. The mean score gain
cannot all be credited to recovery of qrel-positive evidence.

## Suggested text for §4.5

On a different 24-query practice sample excluding the earlier pilot, we
regenerated and judged all three arms under a 768-token input ceiling.
When labelled evidence was already in the prompt, chunking within the same
five documents preserved labelled-source coverage but changed the mean
score from 0.625 to 0.590, and could omit a useful qualifying sentence.
When labelled documents lay at ranks 6–20, chunk-based reranking put one
into the top five and its text into the prompt on 3 of 12 queries; two of
these three scored higher, and the overall mean increased from 0.181 to
0.278 relative to fixed-top-five chunking. These exploratory results
demonstrate some evidence recovery, not a general answer-quality gain:
some scores improved without a labelled-source recovery, and each answer
was generated and judged only once.
