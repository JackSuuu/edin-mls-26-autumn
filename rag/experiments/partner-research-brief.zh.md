# RAG 研究交接：问题、现有证据与后续方法

更新：2026-11-12。供两位组员讨论、分工和撰写报告。文中的“已完成”指已有代码或保存的实验结果；“候选方法”仍需实验验证。方法清单覆盖与当前系统相关的主要路线，不代表每种方法都会提高分数，也不要求全部实现。

## 1. 我们目前在研究什么

暂定研究问题：

> 在课程音频问答系统中，有限上下文预算下的证据选择，如何影响答案质量与计算成本？

我们从课程 baseline 的实际失败出发，使用已有技术检验自己的假设。独立贡献主要体现在问题选择、实现、实验控制、测量和机制分析。采用并引用 reranking、BM25 等已有方法，不需要声称发明新算法。

最初的假设是：扩大检索候选集合，再选择少量证据，可能比直接扩大生成上下文更划算；即使检索指标提高，证据条件冲突、输入组织和生成错误仍可能限制答案质量。

首轮实验没有显示所选 reranker 的优势。接下来应调查它为什么误排，以及候选集合大小会怎样影响结果。BM25 和混合检索是可选择的后续路线，目前尚未在本项目中测试。

与第一部分 ASR 的联系包含两个子问题：ASR 与证据选择各阶段怎样影响端到端延迟；ASR 转写误差是否限制下游检索与答案质量。当前 RAG 主线保持不变，端到端计时与小规模参考转写对照作为连接两部分的补充实验，尚未完成。

## 2. Coursework 对研究和报告的要求

课程要求比较一个技术改动与明确的 baseline，同时衡量质量和相关系统成本，解释取舍、失败和局限。只替换生成语言模型不足以构成系统调查。参见[项目说明](../../README.md)、[RAG 指南](../README.md)和[报告指南](../../report/README.md)。

上一场讨论已经逐页检查课程发放的五份正式 PDF。小组报告 rubric 的权重是：技术工作与系统理解 30 分，实验设计与证据 30 分，分析与结论 30 分，表达 10 分。超过 baseline 本身不是独立评分项目。严谨的负结果可以符合高档描述，但不能据此保证具体分数。正式依据是课程发放的[小组报告 rubric](../../../02_group_report_rubric.pdf)。该 PDF 位于本仓库外，搭档需要从课程材料获取。

整体 coursework 中，小组报告占 70%，个人评审占 30%；视频辅助评分，没有独立分数。报告正文最多六页，涵盖 ASR 与 RAG，二者没有固定分数配额。关键证据必须在正文。ASR 四份任务指南各至少一道报告问答也需要保留。Jack 负责的 CUDA 实现与 profile 证据应与 RAG 结果一起安排篇幅。

视频最多十分钟，演示与讲解最多八分钟，最后两分钟用于课程发布的两个问题。最终提交还要处理匿名要求、AI 声明与学生信息页。具体要求、截止时间和上传安排以课程 PDF 与 Learn 为准。团队仓库和含个人路径的原始日志不能直接当作匿名提交包。

## 3. 系统和 baseline

完整流程是：音频问题 → ASR 转写 → 查询编码 → 检索文档 → 选择与拼接上下文 → ELM 生成答案。

Jack 已实现四组 ASR CUDA 任务。我们保存的 baseline 运行中，所提供的 CUDA 测试通过；这不能代替全数据集 ASR 准确率测量。RAG 调查目前固定 ASR 转写，避免每次更改检索方法时上游输入也发生变化。

现有设置：

- 语料 24,000 条，practice 350 条音频问题，dev 125 条。
- 使用课程 enc_a 查询编码器和对应语料向量，通过 NumPy 做精确向量检索。
- 默认生成上下文取原始前五篇完整文档，没有显式 token 预算。
- 生成模型是远端 ELM 的 `utter-project/EuroLLM-22B-Instruct-2512`。
- ASR 在 GPU 上运行；查询编码、精确检索及本轮 reranker 在 CPU 上运行；生成与裁判通过远端服务调用。
- ASR 转写缓存允许后续检索与证据选择实验独立运行，不必重复占用 GPU。

全部 350 条 practice 的原始检索结果：Recall@5 为 0.421337，Recall@10 为 0.479594，Recall@20 为 0.578524，Recall@100 为 0.769856，MRR@10 为 0.488746，NDCG@10 为 0.420461。设置和原始结果入口见[baseline 记录](baseline-3713067.md)。

原始答案测试只生成和评判了前五题，平均综合分 0.4。它是诊断样本，不能代表全部 350 题，也不能与后来另一批 30 题的平均分直接比较。

### 3.1 ASR 与 RAG 怎样组成同一个研究问题

ASR 的输出就是 RAG 的查询输入：[ASR 接口](../src/asr_adapter.py)返回转写后，[完整问答流程](../src/rag.py)用它编码查询、检索资料、构建上下文和生成答案。转写的内容与产生转写的开销都会传到后续系统。

RAG 主问题仍是有限预算下的证据选择。报告可以用两个可检验的子问题连接 Jack 的 ASR 工作：

- 成本：ASR 算子实现与下游证据选择分别贡献多少延迟，局部优化最终能节省多少用户等待时间？
- 质量：关键内容的转写误差怎样影响候选覆盖与最终答案，证据选择方法对这些误差是否敏感？

这两个问题不要求我们重新训练 ASR 模型，也不预设 ASR 是当前答案错误的主要来源。

### 3.2 CUDA 算子正确性、转写质量和运行时间分别检查

Jack 的自写 CUDA 算子改变计算实现，研究目标是在满足正确性要求的前提下解释性能。不能从“使用自写 CUDA”推断识别更准确，也不能从单个算子加速推断整段 ASR 或完整问答同倍加速。

应分别报告算子数值测试、完整模型转写检查和性能测量。若比较 PyTorch 参考与自写 CUDA，固定模型权重、音频、解码设置及设备，检查转写是否一致；若存在差异，再分析其对检索和答案的影响。现有测试通过尚不能证明全部 350 条转写准确。

### 3.3 成本连接：测量音频到答案的完整延迟

当前流程按顺序执行，可将一次请求的延迟分解为：

```text
T_total = T_ASR + T_query_encoding + T_retrieval
        + T_reranking + T_context_preparation + T_generation
        + 其余接口与传输开销
```

没有重排时，该阶段为零。必须用一次完整请求的实际墙钟时间核对分解结果，避免遗漏、重复计时或把不同机器的数字直接相加。冷启动的模型加载与预热应单独报告；稳态比较复用已加载模型，计时须确保 GPU 工作已经完成。

例如，假设 ASR 原来占 2 秒，其他阶段占 8 秒，总共 10 秒；ASR 加速两倍后，总时间是 9 秒，整体加速约 1.11 倍。这只是解释局部与整体差异的例子，不是本项目的测量结果。

本轮特定 CPU 配置的 top-100 重排平均约 26.20 秒/题，可能成为显著开销，但目前不能据此计算含 ASR 的实际端到端加速比。补测时应在匹配资源下比较参考/自写 ASR，以及 baseline/选定 RAG 策略，记录各阶段与总时间。两组实现的实际快慢均按结果报告。

### 3.4 质量连接：检查转写误差怎样传播

ASR 若改变缩写、专有名词、数字或否定词，可能使检索偏离原问题。Reranker 接收同一份转写，即使其相关性判断正确，也可能围绕一个已改变意思的问题选择资料。候选池缺少原问题需要的证据时，固定候选重排无法补回。

普通词语的转写差异也可能不影响检索。因此应观察具体误差及下游变化，不能只依据总词错误率解释答案质量，更不能根据当前 CAGR 案例直接断言错误来自 ASR。

现有实验冻结实际 ASR 转写，回答的是“给定实际转写，下游策略怎样表现”。这是隔离 RAG 变量的控制。现有本地离线审计没有核对原始音频；缓存文本一致只表示保存的转写一致，不证明转写准确。ASR 误差影响目前仍是待检验问题。

### 3.5 小规模参考转写对照

在查看该补充实验结果前固定一批 practice 音频。参考转写应使用课程提供的可靠文本，或由组员听音频、独立核对得到，不能根据 qrels 或参考答案反推。保留原始转写、参考转写、修正内容与核查记录。

对同一批问题比较四个条件，保持语料、生成模型、预算及其他设置一致：

1. 实际 ASR 转写 + baseline RAG。
2. 实际 ASR 转写 + 选定 RAG 改动。
3. 参考转写 + baseline RAG。
4. 参考转写 + 同一个 RAG 改动。

比较 1 与 2，检查 RAG 改动在实际输入下的作用；比较 1 与 3，检查固定下游策略时修正转写的影响；结合 2 与 4，检查转写质量是否改变 RAG 改动的收益。不能只比较条件 1 与 4，否则同时改变两部分，无法区分来源。

参考转写条件是用于定位瓶颈的理想输入对照，不是系统自动取得的成绩。如果报告词错误率 WER，应明确参考文本及大小写、标点等规范化规则，并补充关键术语和条件是否保留的检查。对这些质量指标的计算尚未完成。

### 3.6 固定原始问题意图进行补充评价

课程[答案评价脚本](../scripts/evaluate_answers.py)把候选中的 ASR 转写作为问题。若比较不同转写，裁判看到的问题也可能变化，因而难以判断哪个答案更符合用户原意。

ASR 对照应额外以同一份根据音频核对的问题意图评价四个条件，并使用相同参考证据与评价规则。保留课程协议分数，单独报告固定原意的补充评价，不混合平均或静默替换原分。检索 qrels 仍按原 query ID 对齐。人工核查还应检查答案对实际生成上下文的忠实性。

建议范围是“一条 RAG 主线，补充 ASR 成本与转写质量两个连接点”。已有 RAG 实验继续作为固定转写下的结果；新增对照完成前，不声称已证明 ASR 鲁棒性、转写改善或端到端加速。

## 4. 为什么选择证据选择这个问题

在固定转写上重放全部 350 题检索后，我们得到以下逐题诊断：

- 197/350 题在第 6 至 100 名还有前五名遗漏的正标注文档。
- 其中 101/350 题前五名没有正标注文档，但前一百名至少有一篇。
- 42/350 题前一百名都没有正标注文档，固定候选池的 reranker 无法补回这些标注。
- 95/350 题前五名已经覆盖全部正标注文档；112/350 题覆盖部分标注。

这说明候选中有可供选择的额外标注证据，但还没有证明这些证据能够改善答案。

如果离线使用 qrels，即相关文档标注，知道候选中哪些文档相关，再从前一百条选至多五篇，平均 Recall@5 的理想上限约为 0.764417。这是该候选集合、该文档数限制下的标注指标上限。它不是可部署方法或答案质量上限。部分问题的正标注文档多于五篇，因此它略低于 Recall@100。

标注可能不完整。未被标注的文档也可能有用；命中一个相关文档 ID，也不能保证内容足够回答问题。这批 350 题诊断的完整新增运行归档目前在本地，团队复核可从冻结转写使用[诊断收集器](../extensions/collect_retrieval_diagnostics.py)重新生成。

## 5. 首轮五组实验已经完成

我们在看结果前固定了 30 条 practice 查询：排除此前查看的前五题，对余下 ID 按 `SHA256(seed + ':' + query_id)` 排序取前 30 条，seed 为 `2026-11-12-evidence-audit-v1`。所有配置使用同一批转写、语料、生成模型和提示词；每题内随机化配置调用顺序。dev 尚未使用。

以下各项依次列出平均完整输入 token 数、平均选中文档标注召回率、平均自动答案综合分：

1. 原始 top-5：1297.4 tokens；0.2883；0.3056。
2. 原始 top-20：4846.6 tokens；0.5064；0.3389。
3. 从原始 top-100 重排后取 top-5：1375.8 tokens；0.2717；0.2889。
4. 原始排序，4096-token 输入预算：4096.0 tokens；0.4664；0.3750。
5. 重排序，4096-token 输入预算：4095.8 tokens；0.4164；0.3028。

top-5 比较控制文档数，4096-token 比较控制输入预算，二者回答不同问题。预算包括系统提示、问题、文档标签和生成前缀。按排名依次装入文档，最后一篇可截断，之后停止。

Reranker 使用 `cross-encoder/ms-marco-MiniLM-L6-v2`，固定 revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`，batch size 16，问题与文档合计最多 512 tokens，只截断文档。使用原始 logits 降序排列，同分保留原顺序，没有在课程数据上微调。生成 tokenizer 固定 revision `8c6ed5c8d18527d9daed994f645032b86c53f0bb`。

150 个答案及 150 次统一 JSON 模式评分均完成。全部服务端输入 token 计数与本地一致，没有预算超限。一个早期 token 计数错误导致的试运行已经排除并重新生成；原先非 JSON 模式的零散评分没有混入这轮比较。JSON 约束解决输出格式问题，不保证裁判判断正确。

配对自动分数差的 query bootstrap 95% 区间为：top-20 对 top-5，差值 +0.0333，区间 [-0.0583, 0.1333]；重排 top-5 对原始 top-5，差值 -0.0167，区间 [-0.1000, 0.0694]；同预算重排对原始排序，差值 -0.0722，区间 [-0.1583, 0.0139]。这些区间均包含零，也没有包含重复生成与裁判系统性错误的不确定性。

目前可以写：这批样本没有显示所选 top-100 reranker 的优势。不能把某个平均分最高写成已确认最优，也不能据此认定所有 reranking 方法均无效。

详细证据见[首轮报告](selection-pilot-results.md)、[逐题与汇总指标](selection-pilot-metrics.json)及[固定样本](selection-pilot-sample.jsonl)。

## 6. 已发现的机制、成本和评价问题

### 6.1 重排会把已经找到的证据排出去

CAGR 发音题中，两篇讨论发音的文档原来排第 1、2，重排后变为第 11、38。原始上下文包含发音信息；重排 top-5 的答案则表示没有这类证据。这两篇配对长度只有 34、97 tokens，没有遭到 512-token 截断。因此不能用长文档截断解释这个案例。

裁判同时误读了这题参考资料中的不同观点。这说明排序错误与评价错误可以一起发生，不能只看最后一个分数。

### 6.2 增加上下文可以带入遗漏信息，也有开销

期权估值题的标注文档排在第 11。top-20 答案包含了该文档讨论的具体方法，top-5 则遗漏。该案例支持继续研究证据覆盖，但单个案例不足以证明总体改善。

top-20 的平均输入是 top-5 的约 3.74 倍。这个比例描述输入 token，不是延迟或货币成本倍数。

### 6.3 文档 ID 覆盖与保留答案片段有区别

两个预算组各有 29/30 题截断了一篇文档。重排预算组的选中 ID 召回为 0.4164，只计算完整保留文档时为 0.3998。前者可能把已丢失关键句的文档也计作命中；后者也可能漏记截断后仍保留有用证据的情况。要判断证据保留，需要进一步检查文本片段。

### 6.4 自动 judge 和 qrels 都有边界

课程裁判分别给正确性、依据性和完整性打 0 至 4 分，综合分为三项之和除以 12。它使用 ASR 转写作为问题，并根据 qrels 重建参考证据。其 grounding 分数不能直接证明答案忠实于生成模型实际收到的上下文。

我们观察到裁判添加问题中没有的条件、忽略参考中的一种观点、错误处理相互冲突的建议。信用评分题也出现“前五篇未命中 qrels，但实际含有用资料”。目前这些是有目的的文本检查，尚未由两位组员做盲审，也不能估计代表性错误率。

### 6.5 重排成本必须算进去

原始 top-5、top-20、重排 top-5 的远端生成延迟中位数约为 4.03、4.23、2.93 秒。它们来自单次调用，输出长度和服务负载不同，不能据此声称重排稳定加速。

top-100 重排在 `opencast` 的 2 CPU 配置上平均约 26.20 秒/题。生成在另一节点 `saxa` 上完成，复用了前一作业的重排分数与其计算成本。因此两者相加只能称归属成本估计，尚非同节点端到端测量。GPU 加速、预热、批处理是否值得，仍需另外测量。

## 7. 可以提高不同 benchmark 的方法

先区分目标：候选 Recall@100 衡量候选覆盖；Recall@5、MRR、NDCG 衡量靠前结果；答案评分衡量生成结果；延迟、吞吐、内存与 token 衡量成本。改善其中一项不保证其他项同时改善。

### 7.1 调整候选数量与最终上下文数量

状态：已完成 top-5、top-20 和 top-100 重排的首轮比较；已做候选深度事后诊断。

候选数 N 决定有多少文档可供选择，最终上下文数量 k 决定交给生成模型多少篇。扩大 N 为选择器提供更多相关文档，也提供更多干扰文档；扩大 k 会增加输入，且同样数量的文档长度可能相差很大。

利用相同的已保存分数，从前 20、50、100 条中分别选重排 top-5，平均标注召回为 0.3117、0.2883、0.2717。它提示较大候选池可能增加误排机会，但属于事后探索，没有对应的新答案实验或新延迟测量。下一轮应检验，而不是把 N=20 宣称为最优。见[逐题候选深度诊断](selection-candidate-depth-diagnostic.json)。

### 7.2 Reranking：对候选重新排序

状态：MiniLM-L6 已实现并完成首轮；其他模型、长文本处理和原始分数融合尚未测试。

向量检索分别编码问题与文档。交叉编码器将二者一起处理，对每个组合打相关性分，再重排。它有机会把主题相关与直接回答问题的文档区分开，但会增加模型推理，也会误判。说明见[Sentence Transformers 官方文档](https://sbert.net/examples/cross_encoder/applications/README.html)与[当前模型卡](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)。

固定一百条候选重新排序不会改变 Recall@100；取前五条时 Recall@5 可升可降。候选外的文档需要另一个检索步骤才能进入。

可测试的改动包括减小候选池、采用另一种匹配任务的 reranker、对长文档分窗打分，以及融合原始排名与重排排名。先验证一个变量，再决定是否增加模型。LLM listwise reranking 也是路线，但 API 费用、顺序敏感性和输出解析都需计入。

### 7.3 BM25：根据词项匹配进行检索

状态：尚未实现或评测。

BM25 根据查询词在文档中的出现情况评分：查询词匹配、词项在语料中的稀有程度、词频饱和及文档长度归一化会影响分数。它不依赖当前的语义向量。对于缩写、专有名词或明确术语匹配，它可能补充 dense retrieval 的遗漏；同义表达或 ASR 拼写错误也可能让它失败。方法来源为 Robertson 与 Zaragoza 的 [The Probabilistic Relevance Framework: BM25 and Beyond](https://doi.org/10.1561/1500000019)。

最小实验应在相同的 24,000 条记录上建索引，保留原 record ID，固定分词、大小写和标点处理、BM25 实现及 k1、b。用同一份冻结转写，与 dense 比较 Recall@5/20/100、MRR@10、NDCG@10，并记录索引构建、内存和查询时间。k1 控制词频饱和，b 控制长度归一化。初轮不要同时改变文档切分与查询改写。

除平均分外，检查 BM25 找回了多少 dense top-100 遗漏的正标注文档，在哪些查询上损失证据。BM25 的优势需要数据验证。

### 7.4 Dense + BM25 混合检索

状态：尚未测试；可在 BM25 单独比较后决定是否增加。

两条检索路线可能找回不同文档。可以分别检索后去重，再用 Reciprocal Rank Fusion，简称 RRF，融合排名。它按文档在各个列表中的名次累加 `1 / (c + rank)`，缺席该列表的项贡献为零。参数 c 控制名次权重的衰减；它与检索数量 k 不同。RRF 的原论文使用 c=60，该值不能当作本课程最优值。见[RRF 原论文](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)。

RRF 避免直接相加不同量纲的 BM25 和向量分数。需要固定两路深度、去重、融合规则与最终唯一候选数量。两路各检索 100 条可产生至多 200 个唯一文档；应同时报告原始候选并集与融合后候选的指标，并控制后续 reranker 接收的数量，不能把额外候选或额外计算隐藏起来。

合理的消融是 dense、BM25、dense+BM25；只有证据支持继续组合时，再测试 hybrid+rerank。最终生成比较仍应匹配 token 预算。

### 7.5 查询规范化、改写和扩展

状态：尚未测试。

可以调查 ASR 中的缩写空格、拼写和标点是否影响检索，再比较固定规则规范化；也可以用模型生成同义改写或多个查询后融合。多查询与 HyDE 一类方法属于更广的扩展路线，涉及新增生成调用及偏离原意的风险，不列为当前必做项。

推理时只能使用实际可获得的音频或转写。不能从 qrels、参考答案或逐题人工修正中提取正确关键词，再当作正常系统成绩。人工校正问题可以做单独标注的 oracle 诊断。原始转写仍须保存，并清楚记录检索使用哪个版本，避免无意改变评价问题。

### 7.6 文档切分与长文档处理

状态：当前只有输入预算截断；分段重检索、分窗重排未测试。

把长文档分成较小片段可能使相关句更容易被检索或评分器看到，也可能丢失前提条件、增加重复结果和索引成本。需要固定片段大小与重叠，映射回原始 record ID，再按文档级 qrels 评价；同一文档多个片段不能重复算多个相关命中。

若改变 dense 检索的片段，需要生成匹配的新向量；若只对已有候选分窗重排，应明确没有改变第一阶段检索。两者是不同实验。

### 7.7 Token 预算、去重和片段选择

状态：完整输入预算与按排名装入已完成；句子选择、MMR 多样性选择和压缩尚未测试。

在固定输入预算下，可以减少重复内容，或选取与问题相关的句子，保留更多独立证据。MMR 是兼顾相关性与减少冗余的一种已有选择方法。也可以生成摘要，但会增加模型调用，且可能删掉例外条件或引入错误。

应先比较“相同排序与预算，只改变装入策略”。抽取片段需要保留来源、否定词和适用条件。全文文档 ID 召回不能单独评价片段质量。生成式压缩及训练压缩器目前优先级较低。

### 7.8 固定文档集合，改变呈现顺序

状态：尚未测试。

保持相同文档、内容和输入长度，只比较原排序、反序或预先固定的随机顺序，可检查答案是否对证据位置敏感。来源启发是 [Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/)。该论文的现象不能直接当成 EuroLLM 的已知行为。

如果使用 qrels 把已知相关文档放到开头，应标为离线 oracle。正常系统可以按可获得的模型分数排序。这个实验主要研究生成如何利用证据，候选集合和文档级召回本身保持不变。

### 7.9 更换表示模型或采用更复杂检索器

状态：尚未测试。

可以比较不同 dense encoder、学习稀疏检索或 token 级交互模型，但需要匹配的查询与语料编码、索引和计算资源。课程同时提供 enc_a/enc_b 语料向量，不代表能直接用 enc_a 查询向量搜索 enc_b；必须先核对 enc_b 的兼容查询编码器、维度、归一化和 ID 映射。

这类路线涉及更多资产与索引工作。只有当当前证据指向表示瓶颈时才优先考虑，避免在六页报告中失去可解释的比较主线。

### 7.10 生成提示与选择性回答

状态：首轮固定生成提示，没有测试新提示。

可以要求答案区分适用条件、给出证据来源、在证据不足时说明不确定性，观察错误主张与遗漏是否减少。要防止通过普遍拒答换取较少错误，却牺牲完整性。引用格式也不能证明引用内容支持结论。

此类实验应固定上下文，只改提示；保存真实输入并检查逐条主张。不能把调整回答风格以迎合某个裁判偏好，当作已证明的质量改善。

### 7.11 按问题分配计算

状态：尚未测试；暂缓。

有些问题可能只需浅层候选，另一些可能需要重排或更多证据。动态选择候选深度、是否重排或上下文预算，可能降低平均成本，但需要一个能在未知答案时工作的决策规则。

当前向量分数和分数差还没有经过置信度校准。先测清固定策略，再讨论路由，并将理想路由器与可部署规则分开。训练 Self-RAG、Adaptive-RAG 或领域微调都超出当前最小实验范围。

### 7.12 索引、批处理、缓存和 GPU 优化

状态：已有冻结转写和重排分数缓存，用于实验复用；尚无正式性能对照。

ANN 索引、查询批处理、GPU 重排和模型量化可能降低延迟或内存、提高吞吐。它们主要改变成本，与增加语义证据是不同目标。当前 baseline 是精确检索，ANN 需要报告相对精确结果的近似误差和任务质量，不能预设它会提高召回。

缓存命中性能与首次计算性能应分开。批量查询的吞吐不能替代单请求延迟；模型加载、预热、线程、批大小、设备和数据类型应明确。GPU 或低精度计算也要核查排序及答案是否发生变化。

## 8. 下一轮实验顺序

### 第一步：完成当前主线的机制检查

在 practice 上比较相同 reranker 的候选深度，先查看选中 ID、Recall@5、MRR/NDCG 和错误升降名次。可以复用完整候选的分数做质量诊断，但不能用这份缓存推算较小 N 的真实运行时间。计时应在同一节点按实际 N 重新运行，控制预热、线程和批大小。

对明显改善、明显退步和随机抽样的问题分别检查：候选是否缺证据；选择是否漏证据；截断是否丢关键句；答案是否误用条件；裁判是否误读。检查类型允许重叠。诊断案例不能用来估计随机总体错误率。

### 第二步：决定是否增加 BM25 这一条替代路线

如果团队仍有时间，先完成 dense 与 BM25 的全 350 题检索比较和互补性诊断，再决定是否加入 RRF。这一阶段不需要 ELM 生成与裁判调用，但检索和模型计算仍需申请合适资源。

把它接回同一个研究问题：不同候选来源是否在固定证据预算下保留了更多可用信息？若没有互补性证据，就不需要为了方法数量继续堆叠 hybrid+rerank。

### 第三步：对少数有依据的配置进行答案与成本比较

保留原始 top-5、较大上下文对照及原始排序预算组，选择少数候选深度或检索来源配置。固定问题、生成设置与预算，记录完整输入、输出、输入/输出 tokens、各阶段耗时、结束原因和评分。

在固定的 30 题上继续探索时，明确它已经用于方法开发。之后应在事先固定的未查看样本上确认关键比较。新一轮远端 API 比较应交错调用、记录调用时间，避免把不同时段的服务负载差异当作方法效果。

在冻结最终方案前补充第 3 节的 ASR 联系实验：测量匹配条件下音频到答案的延迟；预先固定小规模 practice 音频，开展实际/参考转写与 baseline/选定 RAG 的四条件对照。该诊断与原有固定转写结果分别报告。若资源不足，明确未完成的部分，避免扩大结论。

### 第四步：冻结方案后进行 dev 验证

事先确定最终配置、主要指标、样本范围、裁判协议和分析规则，再跑 dev。不要根据 dev 结果反复挑选参数后继续称其为独立确认。若 API 预算只能覆盖一部分题，先固定抽样规则并报告覆盖率。

最小可交付研究已经可以围绕“上下文数量与预算、候选深度、重排失败和评价边界”完成。BM25、混合检索与更复杂方法需要根据时间和诊断结果选择。

## 9. 评价、统计与分工

检索层保存逐题 Recall@5/10/20/100、MRR@10、NDCG@10；对差异进行配对分析。答案层保留课程三维分数，同时核查真实上下文对重要主张的支持，报告样本数与裁判设置。候选 recall、证据充分性和答案正确性必须分别解释。

建议两位组员独立查看一批随机抽样答案，隐藏方法标签，记录证据是否足够、重要主张是否有支持、条件是否混淆，以及裁判判断是否合理，再讨论分歧。另设诊断案例集合解释机制，不混入代表性错误率。保存自动原分和核查意见，避免静默改分。重复 judge 能观察波动，无法自动消除共同偏差。

成本层至少记录查询编码、检索、重排、上下文准备和远端生成；区分实验排队、加载、稳态计算及网络/API 时间。若报告含 ASR 的端到端成本，应实际测量并清楚标注；固定转写的 RAG 比较则单独报告下游成本。输入 token 不是计费金额，远端服务器实际硬件也不在我们控制范围内。

建议分工如下，具体由两位组员确认：

- RAG 负责人继续候选深度、BM25 可行性和预算对照，保留逐题结果及失败案例。
- Jack 汇总 ASR 四项测试、profile、报告问答和设备配置，检查参考/自写实现的转写与运行时间，核对完整音频链路。
- 两人一起准备和核对小规模参考转写，完成端到端计时及四条件对照；开展答案盲审与争议案例核查，确定最终配置、报告结论、篇幅和视频演示。

这些是协作建议，尚未给搭档分配或发送任何任务。

## 10. 报告需要形成的论证

正文应回答：baseline 暴露了什么可测量的问题；为什么选择当前改动；哪些因素固定、哪些变化；质量与成本怎样变化；哪些证据支持解释；在哪些问题上解释失效。

可以按以下顺序组织 RAG 内容：问题与 baseline；选择策略及实现；受控实验；质量与成本结果；候选深度、证据损失和 judge 案例；局限与结论。篇幅需与 ASR 合并规划。

ASR 与 RAG 的衔接应由实验支撑：先说明算子实现的正确性和 ASR 成本，再解释固定转写下的证据选择，最后用实际测得的端到端时间及小规模转写对照检查二者的相互影响。若某部分仍未完成，应作为局限或后续工作，不能在结论中预先宣称收益。

最有价值的图包括候选深度与标注覆盖、输入 token 与答案分数、实际测得的选择成本，以及一个展示证据排名变化的案例。图注应注明数据集、样本数、固定条件和误差含义。只有实际测量后的成本才能进入正式比较。

应明确哪些组件由课程提供，哪些代码与实验由我们完成。引用已有方法，保留负结果，不把“可能原因”写成已证明机制。附录不能代替正文中的主要证据。

## 11. 已读论文与可借鉴的实验方法

另一场 `rag-baseline-exploration` 讨论重点读了十二篇，另补读一篇。我们借鉴的是问题定义和控制实验；当前代码没有复现整套论文系统。阅读内容已从该讨论重新核对。

1. [Lost in the Middle，TACL 2024](https://aclanthology.org/2024.tacl-1.9/)：保留证据，只改变位置，区分检索缺失与信息利用。
2. [The Power of Noise，SIGIR 2024](https://doi.org/10.1145/3626772.3657834)：区分主题相近干扰与随机无关内容；不要把未标注文档一概当作噪声。
3. [Sufficient Context，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/33dffa2e3d2ab74a783d1a8c292f66d9-Paper-Conference.pdf)：区分上下文是否足够与模型是否利用好上下文。
4. [RECOMP，ICLR 2024](https://openreview.net/notes/edits/attachment?id=gYLmIMyflp&name=pdf)：压缩实验应同时呈现质量、输入长度和额外计算。
5. [RankGPT，EMNLP 2023](https://aclanthology.org/2023.emnlp-main.923/)：检查初始排序敏感性、额外计算及方法边界。
6. [Fusion-in-Decoder，EACL 2021](https://aclanthology.org/2021.eacl-main.74/)：方法应对应明确瓶颈；其专门训练的架构不同于我们拼接上下文的黑盒模型。
7. [REPLUG，NAACL 2024](https://aclanthology.org/2024.naacl-long.463/)：新增模型调用时，安排能够区分内容收益与额外计算收益的对照；复现还取决于 API 能力。
8. [Self-RAG，ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/25f7be9694d7b32d5cc670927b8091e1-Paper-Conference.pdf)：将检索、相关性、证据支持和生成决策分开分析。
9. [Adaptive-RAG，NAACL 2024](https://aclanthology.org/2024.naacl-long.389/)：比较质量与每题计算，明确理想路由和实际路由的差距。
10. [ALCE，EMNLP 2023](https://aclanthology.org/2023.emnlp-main.398/)：回答正确性与引用支持是不同指标，要检查具体主张。
11. [ARES，NAACL 2024](https://aclanthology.org/2024.naacl-long.20/)：自动评估器也需要人工核查和不确定性分析。
12. [Ties Matter，EMNLP 2023](https://aclanthology.org/2023.emnlp-main.798/)：评价规则和微小差异本身需要检查；其研究主要针对机器翻译元评估，不能直接移植成我们的显著性检验。
13. [Large Language Models Can Be Easily Distracted by Irrelevant Context，ICML 2023](https://proceedings.mlr.press/v202/shi23a.html)：固定原题与答案，控制干扰变化以研究鲁棒性。

这些工作的结果都有具体模型、任务和训练条件，不能当作本课程系统必然出现的效果。我们不需要继续广泛搜集方法才能完成下一轮实验；出现具体未解释机制时再定向阅读。

## 12. 搭档可以从哪里开始复核

已有代码与首轮报告在提交 [209951f](https://github.com/JackSuuu/edin-mls-26-autumn/commit/209951f)，提交信息为 `feat(rag): add budgeted evidence selection and pilot report`。

优先阅读[首轮报告](selection-pilot-results.md)及[扩展模块说明](../extensions/README.md)。代码入口包括[选择与预算控制](../extensions/evidence_selection.py)、[实验准备和生成](../extensions/run_selection_experiment.py)、[分数独立验证](../extensions/validate_reranker.py)与[配对结果汇总](../extensions/summarize_selection_experiment.py)。课程裁判实现见[answer_judge.py](../src/evaluation/answer_judge.py)。

保存的三十题逐题指标和样本在仓库内；首轮新增的完整提示词、答案、裁判输出、哈希与 Slurm 日志在本地归档，没有随该提交全部上传。运行命令见扩展模块说明，不依赖个人 ICF 脚本。新方法应使用新输出目录，只有来源与冻结计划一致时才续跑旧实验。

## 可直接发给搭档的英文摘要

We are investigating how evidence selection under a limited context budget affects answer quality and computational cost in the coursework audio-QA system. We use established methods, while contributing the implementation, controlled comparisons and analysis. The report rubric gives 30% each to technical understanding, experimental evidence and analysis, and 10% to presentation; improvement over the baseline is not itself a scoring criterion.

The full 350-query practice baseline has Recall@5 = 0.4213 and Recall@100 = 0.7699. A diagnostic found 101 queries with no labelled relevant document in the top five but at least one in the top 100, motivating candidate selection. Another 42 queries have no labelled positive in the top 100, which a reranker restricted to that pool cannot recover.

We have completed a fixed 30-query pilot comparing dense top-5, dense top-20, MiniLM reranking from 100 candidates to five documents, and dense/reranked selection under a 4096-token input budget. The chosen reranker did not show an advantage: top-five labelled recall fell from 0.2883 to 0.2717. Under the matched budget, recall was 0.4664 for dense ordering and 0.4164 for reranking. Answer-score comparisons remain uncertain, and case inspection found judge errors. These are exploratory results; dev remains unused.

Next we should investigate candidate depth and incorrectly promoted/demoted documents, and measure selection costs under matched resources. BM25 is an untested alternative for lexical matching; if it provides complementary evidence, dense+BM25 fusion is a possible extension. Other options include passage selection, query normalization and document ordering, but we should add them only when a diagnosed failure justifies the experiment. We will freeze the final method before dev validation and report quality, cost, negative results and evaluation limitations together.

We will connect this investigation to the ASR work through end-to-end latency and a small transcript-quality diagnostic. The proposed four conditions cross actual ASR versus audio-verified reference transcripts with baseline versus the selected RAG strategy. Supplementary answer evaluation must use the same original question intent across conditions, because the course evaluator otherwise uses each candidate's ASR transcript as its question. Reference transcripts are an oracle diagnostic, not deployed system output. These experiments are planned: the existing fixed-transcript RAG results do not establish ASR robustness or end-to-end speedup.
