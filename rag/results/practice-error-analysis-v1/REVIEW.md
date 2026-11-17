# Practice error diagnostic — offline, exploratory

Reviewer: OpenAI coding assistant; unblinded exploratory review, not independent human adjudication.

Uniform sample: 20/121 Qwen judge-zero practice answers; seed 20261118.
No new inference, generation, judging or dev analysis. Original scores unchanged.
Gold = positive-qrel documents; selected = actual outgoing prompt documents.
A missing qrel ID does not prove missing answer evidence. No causal snippet effect is measured.

## Population counts (not inferred from the sample)

| Arm | Queries | Judge all-zero | No qrel ID | Partial | All qrel IDs |
|---|---:|---:|---:|---:|---:|
| dense5 | 350 | 138 | 90 | 29 | 19 |
| qwen4_matched5 | 350 | 121 | 62 | 29 | 30 |

All 700 saved prompts exactly match full corpus text and their saved hashes.
This rules out local context-builder prefix truncation for these runs, not remote-server truncation.
All 700 scores match saved raw judge responses and caches; no empty answers.
The three dimension-zero sets coincide in each arm; this flags calibration, not proven judge bias.
The frozen judge prompt uses a zero-filled JSON example; anchoring is possible but untested.
Paired score states: {"positive_positive": 176, "positive_zero": 36, "zero_positive": 53, "zero_zero": 85}.

## Sample review

Sample qrel-ID coverage: {"full": 5, "none": 13, "partial": 2}.
Tags are non-exclusive observations/hypotheses, not verified causes or population error rates.
Judge-text conflicts are review flags, not corrected scores.

### 1. `q_071078c6a43f6ed1`

**Question:** What distinguishes index and accumulation tracker funds?

**Qrel-ID coverage:** none; Qwen score: 0; paired Dense5 score: 0.0000.

**Tags:** topic_substitution.

问题比较 index 与 accumulation tracker；回答转成 index funds 与 ETFs 的费用、流动性比较，没有解释分红再投资。第一篇选中文档本身就在讨论 ETFs。标注文档缺失，但题目术语也不足以确认原始提问意图；不能单独归因于 reranker。

**Snippet assessment:** 弱：现有 top-5 没有明确提供需要的 accumulation 区分，剪短不保证补足信息。

> [answer] Index funds and ETFs may have different expense ratios

### 2. `q_0a12138b81eb7e2a`

**Question:** Where can I find the expense ratios on my financial statement?

**Qrel-ID coverage:** none; Qwen score: 0; paired Dense5 score: 0.0000.

**Tags:** reference_scope_mismatch.

回答中的 quarterly reports 和 prospectus 直接来自选中文档，而非凭空编造；gold 文档讨论 Vanguard 费用和 NAV。judge 按 gold-only 规则不认可其他检索证据，不能据此认定该答案事实错误。NAV 中反映费用也不等于 statement 显示 expense ratio。

**Snippet assessment:** 弱：优先检查参考覆盖和题意，不宜将分数变化当作片段改善事实质量的证据。

> [selected/r_c84ad42a3725dd6d] that info's in the quarterly reports... and in the prospectus

### 3. `q_1024d33a7dfa33ef`

**Question:** What factors should I consider when identifying stocks that are on sale?

**Qrel-ID coverage:** none; Qwen score: 0; paired Dense5 score: 0.0000.

**Tags:** off_topic_detail, reference_scope_mismatch, snippet_candidate.

多数回答来自估值相关文档，但把另一篇关于出售股票税基确认的 Written Confirmation 也列为识别便宜股票的因素。这是可定位的离题信息纳入；gold 中对估值的不同观点亦令零分不能直接代表完全无用。

**Snippet assessment:** 有条件值得：可检验删除出售税基材料后是否仍出现离题建议；当前仅证明共现，不证明干扰的因果关系。

> [answer] Written Confirmation

> [selected/r_de6f003a8b75bde9] Specific share identification.

### 4. `q_32480a484879c3ec`

**Question:** What is the process for receiving eBay account and sim cards at my address?

**Qrel-ID coverage:** full; Qwen score: 0; paired Dense5 score: 0.0000.

**Tags:** query_reference_mismatch, judge_text_conflict.

问的是 receiving eBay account and SIM cards 的 process；gold 实际建议查信用报告、邮件转寄和误投地址。回答按字面问题表示证据不足。judge 自己承认回答 correctly identifies 证据不足、is grounded，却给三维全零；需要人工复核，而不是强行记为生成失败。

**Snippet assessment:** 弱：主要是字面问题与参考情境不匹配；裁剪不会恢复缺失的情境。

> [judge] The answer is grounded in the evidence

> [gold/r_f2f5f58748a1c66e] Check with the Post Office whether anyone has set up a redirect

### 5. `q_371c6ca10bf1bf5e`

**Question:** What advantages does having a pension retirement account in Ireland offer?

**Qrel-ID coverage:** full; Qwen score: 0; paired Dense5 score: 0.0000.

**Tags:** jurisdiction_scope_blending, snippet_candidate.

全部两个 gold 文档都已进入 prompt。回答把 pension maximum contributions 与 carry-forward/offset against future profits 连在一起；选中内容既有明确 UK allowance，也有公司亏损抵未来利润的讨论，还有明确 US retirement account 语境。回答将其组织成 Ireland 优点，未保留适用主体/法域边界。已能定位范围混合，但尚不能证明删片段会纠正。

**Snippet assessment:** 本样本最强候选：保留爱尔兰适用信息及退休前提款限制，移除或标明 UK/US/公司税语境；若选择时使用 gold，必须标为 oracle 诊断而非可部署方法。

> [answer] these can be carried forward and offset against future profits

> [selected/r_1ada1a124c5a8a12] In the UK you have an allowance

> [selected/r_422ea5e23cebc8b1] In the US, when we say

### 6. `q_3bd06eaf675d1a65`

**Question:** Can you explain the reasons behind the negative items on my credit report?

**Qrel-ID coverage:** none; Qwen score: 0; paired Dense5 score: 0.3333.

**Tags:** reference_specificity_gap, judge_text_conflict.

回答解释一般信用报告负面项目，gold 主要讨论信用历史短、无活跃信用卡和系统总会报最弱项；这是具体情境缺失。judge 还指责回答把责任给 credit bureaus，但答案结尾明确说责任属于 creditors，存在可核对的误读。不能因此替整个答案改分。

**Snippet assessment:** 弱：先找到对应情境的证据；同时人工复核 judge 理由。

> [answer] this responsibility lies with the creditors who provide the data

> [judge] incorrectly attributes the reasoning to the credit bureaus

### 7. `q_3d50f31d4e5c576c`

**Question:** How do titles, financing, and insurance work together?

**Qrel-ID coverage:** none; Qwen score: 0; paired Dense5 score: 0.0000.

**Tags:** query_ambiguity, topic_substitution.

问题未说车辆或房产；检索和回答主要转向房地产 title insurance，gold 则讨论车辆持有人与驾驶人不同。可观察到领域偏移，但没有原始问题情境，不能称为已确认的 ASR 错误。

**Snippet assessment:** 弱：优先澄清 vehicle vs real estate，而不是在房产证据中再选片段。

> [answer] residential real estate transaction

> [gold/r_d39a0d1dbed6f81c] a vehicle being in one person's name

### 8. `q_528e698db2500a86`

**Question:** Can an F one student visa holder borrow a car from BMW?

**Qrel-ID coverage:** none; Qwen score: 0; paired Dense5 score: 0.0000.

**Tags:** query_reference_mismatch, reference_scope_mismatch.

F1 学生成功取得 dealership auto loan 的说法确实在选中文档中，答案也明确没有 BMW-specific policy。gold 则讨论融资被拒、撤销购车和里程费。参考未命中不等于生成没有证据；也不能据此确认 BMW 会贷款。

**Snippet assessment:** 弱：先核实问题是询问一般资格还是已发生的融资撤销。

> [selected/r_9091b20b526eaccd] including one on F1

> [answer] the evidence does not specifically mention BMW

### 9. `q_591325342f8c340f`

**Question:** How can I finance a sabbatical to prepare my house for sale?

**Qrel-ID coverage:** none; Qwen score: 0; paired Dense5 score: 0.0000.

**Tags:** reference_scope_mismatch, condition_omission.

虽未检索到 gold ID，top-5 的另一个文档明确提到 HELOC，回答也包含该建议。因此 Recall=0 不代表 HELOC 证据完全缺失。但 gold 的权益要求、银行差异、最坏情况下能否负担等限制没有完整保留，答案又加入 flip/bridge financing 等不同情境。

**Snippet assessment:** 有风险：剪短可能进一步丢掉权益和还款条件；优先保留条件并识别具体用途。

> [selected/r_515c5b86e5dc3220] a Home Equity Line of Credit

> [gold/r_ce41f6c8f083547e] assuming at least 20%

### 10. `q_66bb08d4774227f7`

**Question:** What is the delivery time for a Tangerine no fee money transfer via email?

**Qrel-ID coverage:** full; Qwen score: 0; paired Dense5 score: 0.0000.

**Tags:** condition_omission, judge_text_conflict.

gold 已位于第一篇。回答保留 email 约2小时和到账2–3工作日，但用 after that 省略收件人点击并填银行信息这一触发条件。judge 却说答案断言 after clicking the link，原回答没有该短语；属于条件不完整加 judge 误述，不宜称为时间数字完全错误。

**Snippet assessment:** 低优先级：正确片段本已很短且排第一，裁剪未必比改进条件保留更有帮助。

> [answer] 2-3 business days after that

> [gold/r_6b6992a642b0f134] click the link, and enter their bank account number

> [judge] after clicking the link

### 11. `q_7fb7c14235f8ec06`

**Question:** Why am I unable to purchase an asset at the ask price?

**Qrel-ID coverage:** none; Qwen score: 0; paired Dense5 score: 0.0000.

**Tags:** reference_specificity_gap, reasoning_mismatch.

gold 解释快速行情中最低 ask 已被他人买走，回答却说卖方不愿以 ask 成交，并同时建议 market order。关键动态市场机制缺失，所给因果解释不匹配。

**Snippet assessment:** 弱：需要先检索到快速行情/报价过期机制的证据，不能假设现有 top-5 剪短即可补全。

> [answer] unless a seller is willing to accept that price

> [gold/r_7e046cb62cfe1b65] taken out the lowest ask price

### 12. `q_8f81a59b842b2036`

**Question:** Where can I find recent updates on major shareholders changing positions in a specific stock?

**Qrel-ID coverage:** none; Qwen score: 0; paired Dense5 score: 0.0000.

**Tags:** reference_scope_mismatch.

选中的 Schedule 13D 文档明确提供 SEC EDGAR 和变更修订，回答实质利用了它；gold 只有 SEC 电子提交及公司网站的简短说明。judge 承认 SEC/13D 与 gold 相符，但仍全零。这里优先考虑参考范围与分数校准问题；不证明所有证券披露细节都准确。

**Snippet assessment:** 弱：零分并非证明缺少可回答证据，不应据它启动片段优化。

> [selected/r_135a4ed36c155e21] the SEC’s EDGAR database

> [answer] SEC's EDGAR database

### 13. `q_b7bd4791c356567a`

**Question:** What is the strategy for trading large volumes with minimal profit per share?

**Qrel-ID coverage:** none; Qwen score: 0; paired Dense5 score: 0.0000.

**Tags:** reference_scope_mismatch, risk_caveat_omission.

答案主要在讲 iceberg orders、流动性与分拆执行，这些有选中文档依据；gold 重点是低收益高下行风险及机会成本。回答偏执行策略而没有覆盖策略是否值得做。简短问题不足以唯一确定原场景，不能仅因 gold 缺少 iceberg 就叫幻觉。

**Snippet assessment:** 弱：应补策略风险视角，而非只压缩执行建议。

> [answer] hidden (iceberg) orders

> [gold/r_2e88b885114c0a13] your upside is low

### 14. `q_bc1c349d33c5f4b0`

**Question:** How can I efficiently search for financial institutions, credit cards, etc. at least in Canada?

**Qrel-ID coverage:** none; Qwen score: 0; paired Dense5 score: 0.4167.

**Tags:** reference_scope_mismatch, off_topic_detail.

Redflagdeals 推荐有直接检索依据；但银行交易导入软件和跨国信用记录材料被整理成 search options，部分不等于比较金融产品的搜索工具。gold 则主要劝不要逐高息账户而非提供搜索方法。这里有任务适配问题，也有参考范围差异。

**Snippet assessment:** 有条件但非首选：可保留直接工具推荐，不过没有完整搜索流程的证据。

> [selected/r_29cac308571f3b27] redflagdeals.com

> [answer] Less Accounting and Wave

### 15. `q_c3dad6ee3fe9e058`

**Question:** What are the tax write offs available for individuals, and how much do I need to spend before the end of the year to maximize these deductions?

**Qrel-ID coverage:** none; Qwen score: 0; paired Dense5 score: 0.0000.

**Tags:** query_reference_mismatch, judge_text_conflict, source_quality_risk.

问题和答案主体都是 individuals 的扣除，gold 却主要讨论小企业再投资。judge 将整个回答描述成 business tax considerations，遗漏了大量个人扣除内容。另有实质风险：答案把 student loan interest 称为 tax credit，这一混淆能追溯到选中文档本身；这是来源质量/条件问题，不是简单压缩就能保证解决。

**Snippet assessment:** 弱且有风险：query relevance 不会自动识别过时或错误税务说法；本分析不作现实税务建议。

> [answer] Student Loan Interest Credit

> [selected/r_719cbfa7faa8d64d] The most common tax credit for most people is student loan interest

> [judge] Instead, it discusses business tax considerations

### 16. `q_c5a5320bcfd3a373`

**Question:** How to transfer money from four hundred three B to four hundred one K.

**Qrel-ID coverage:** none; Qwen score: 0; paired Dense5 score: 0.0000.

**Tags:** entity_recognition_gap, topic_substitution.

four hundred three B / four hundred one K 在问题中是拼读形式；gold 是退休计划 rollover，top-5 和答案却是 PayPal、SWIFT 等普通汇款。可确认实体识别/检索主题失配，但未做原音频核对或规范化对照，不能声称 ASR 本身识别错了。

**Snippet assessment:** 很弱：需要首先找回退休计划证据；当前候选文档剪短不能产生 rollover 流程。

> [answer] PayPal

> [gold/r_823a1ad83e20d75e] move money from a 403b to a 401k plan

### 17. `q_ce474d8f8d548797`

**Question:** Loan for car purchase, cash third party or bank loan.

**Qrel-ID coverage:** full; Qwen score: 0; paired Dense5 score: 0.0000.

**Tags:** reference_scope_mismatch, reference_point_omission, snippet_candidate.

gold 位于第二篇，明确 dealer 只关心即时无风险收款而不关心资金来源。答案整合了其他文档的 cash vs loan 和 HR 建议，却未说这一核心点。说明参考观点虽可见仍被遗漏，但标题式问题本身也允许一般融资比较，不能把所有非 gold 内容称为错误。

**Snippet assessment:** 有条件值得：检验聚焦交易结算要求能否保留该要点；须避免把 gold-aware 选择当成普通检索方法。

> [gold/r_11004e7a833d09ed] they care about it becoming their money immediately

> [answer] Employer Assistance

### 18. `q_e16177bbeb991130`

**Question:** What are some recommended software packages for technical analysis?

**Qrel-ID coverage:** full; Qwen score: 0; paired Dense5 score: 0.0000.

**Tags:** reference_scope_mismatch, judge_text_conflict.

答案包括 gold 的 MetaStock、Incredible Charts、VectorVest，并引用其他检索文档的 Jstock/Bloomberg。judge 一面说 Bloomberg 的 any kind of data unsupported，一面又说 gold only mentions 同样措辞；实际这句在检索的非 gold 文档。零分与覆盖 gold 要点也存在明显张力，应复核而不直接改分。

**Snippet assessment:** 弱：可能只是删除 gold 外信息更迎合 judge；不能等同事实质量改善。

> [selected/r_5c371e52b3288173] It provides any kind of data you can imagine.

> [judge] The gold evidence only mentions that Bloomberg Professional provides any kind of data you can imagine

### 19. `q_e86dfaec0aa23a50`

**Question:** Which assets would be most valuable in a post-apocalyptic scenario?

**Qrel-ID coverage:** partial; Qwen score: 0; paired Dense5 score: 0.0000.

**Tags:** condition_omission, judge_text_conflict.

答案保留大量 gold 支持的生存物资，但把 gold/silver 在危机后的用途变为 during the crisis，且弱化实物安全条件。另一方面 judge 声称 gasoline/helicopter、knowledge/connections 不在 gold，实际多篇 gold 原文明确包含它们。因此真实范围错误与评估误读并存。

**Snippet assessment:** 有风险：若只留关键词会丢 during/after 与安全条件；不能把 judge 全零视作整份列表虚构。

> [answer] they can be exchanged for necessary items during the crisis

> [gold/r_bb78de160a91c5cd] Gold and silver are for after the crisis, not during.

> [gold/r_9ab98c1b831ede78] I'd prefer an ultralight helicopter.

> [gold/r_8aaa1b0a0e0c599a] connections will likely also be important

### 20. `q_f76c520706cfe2c3`

**Question:** What is the purpose of an options spread risk graph for beginners in options trading?

**Qrel-ID coverage:** partial; Qwen score: 0; paired Dense5 score: 0.0000.

**Tags:** query_reference_mismatch, reference_point_omission.

问题泛问初学者风险图用途，回答按第一篇文档解释 vertical spread 的非线性风险；gold 讨论某个有两个到期日交易的图只显示当前和一个到期日。答案没保留这一特定限制，但不能从泛问题断言它必须复述该案例。judge 自己先承认风险/收益用途，又否认 P/L 随标的变化是主要用途，评分依据应复核。

**Snippet assessment:** 弱：先确认需要一般概念还是具体交易诊断，避免裁剪强化错误情境。

> [gold/r_e725b29f54e87c4a] Your trade has two expirations

> [answer] nonlinear risk profile of vertical spreads

## Second source check (not human adjudication)

OpenAI coding assistant; second unblinded source check, not independent human review

The user reports author confirmation of these three representative cases; this is not a complete 20-case mechanism taxonomy or causal intervention.

### `q_371c6ca10bf1bf5e`

The retrieved company-tax text assigns carry-forward/offset to a tax loss. The answer assigns it to pension contribution limits. This directly supports a scope/subject substitution observation without needing to adjudicate Irish tax law. It does not prove removing that document would correct the answer.

> [selected] there will be a tax loss that can be carried forward and offset against future profits

> [answer] There are maximum contribution limits, but these can be carried forward and offset against future profits

### `q_e16177bbeb991130`

The gold document names MetaStock, Incredible Charts and VectorVest and the answer covers all three. Extra software recommendations have retrieved support but not necessarily gold support. The judge's Bloomberg sentence is internally contradictory and misattributes selected-only wording to gold. This flags the explanation and score calibration, not a demonstrated fully correct answer or a replacement score.

> [gold] I used to use MetaStock Trader

> [gold] I currently use Incredible Charts

> [gold] I'm looking at VectorVest

> [answer] 2. MetaStock Trader

> [answer] 3. Incredible Charts

> [answer] 4. VectorVest

> [selected] It provides any kind of data you can imagine.

> [judge] The gold evidence only mentions that Bloomberg Professional provides any kind of data you can imagine

### `q_e86dfaec0aa23a50`

Two findings coexist: the answer transfers gold/silver usefulness into the crisis despite gold saying after, while the judge denies helicopters and connections that are present in gold. A judge-explanation error therefore does not vindicate the whole answer. No score was changed.

> [gold] Gold and silver are for after the crisis, not during.

> [answer] they can be exchanged for necessary items during the crisis

> [gold] I'd prefer an ultralight helicopter.

> [gold] connections will likely also be important

> [judge] the mention of helicopters, gasoline
