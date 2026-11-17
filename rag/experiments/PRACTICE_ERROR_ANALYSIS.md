# Practice 错误分析：是否值得做 Snippet Selection？

## 结论

**暂不启动完整片段实验。** 已有证据支持“小范围研究证据适用范围混合”，
但不支持“长 context 是主要失败原因”或“裁剪能普遍改善答案”。
先人工复核养老金、股票估值、汽车付款三个候选案例及若干 judge 异常，
比扩大生成实验更有针对性。

本次为 AI 辅助、非盲、探索性离线分析，不是独立人工 adjudication。
未运行 ASR、embedding、reranker、generation 或 judge；未读取 dev 答案；
分析阶段未修改现有分数、报告正文或此前的盲审文件。随后按用户要求，
已将关键观察和停止标准整合进报告正文，并在附录补充方法、案例与局限；
原有分数、区间和盲审文件仍未改动。

## 设计与可复现性

- 数据：`rag/results/qwen4-answers-3724290/` 中已有 fresh practice 配对结果。
- 抽样框：Qwen 的 **121 道三维 judge 全零题**，而非所有错误或所有350题。
- 将 ID 排序后，用 Python `random.Random(20261118).sample(..., 20)`
  无放回均匀抽20题；不按案例是否有趣补换题目。这不是预注册实验。
- 检查问题、Qwen 回答、judge 理由、qrel-positive 原文，并追踪诊断涉及的
  实际 top-5 文档；同时保留 Dense5 配对答案供复核，不做新的配对显著性检验。
- 脚本对全部700份保存的 prompt 做精确重建及 SHA-256 检查，
  并对每个定性引文检查其确实出现在指定的答案、judge、gold 或 selected 文档中。
  引文匹配不等于解释或事实正确，仍需作者确认。
- 各标签非互斥，不应相加当作错误总数，也不外推总体失败机制比例。

从仓库根目录运行（stdlib-only，不导入模型/API 客户端）：

```bash
python3 rag/extensions/analyze_practice_errors.py
python3 -m unittest discover -s rag/tests -p test_practice_error_analysis.py -v
```

输出目录：`rag/results/practice-error-analysis-v1/`

- `summary.json`：抽样 ID、全量计数、输入/源码哈希与局限。
- `cases.json`：20题的两组答案、实际选中文档、gold 和 judge 理由。
- `REVIEW.md`：逐题诊断、适用局限及已验证原文引句。

可编辑定性标注：`rag/experiments/practice-error-analysis-v1.json`。
脚本重跑只重建以上三个诊断文件，不覆盖原始实验结果。
结果目录受仓库现有 ignore 规则影响；提交时如需要应精确指定文件，
不要顺带提交其他本地结果或盲审 key。

## 一、全量结果：judge 全零不是单一错误类型

以下是350题每个 arm 中的原有 judge 全零计数；不是从20题估计：

| Arm | 全零题 | 未命中任何 qrel ID | 命中部分 | 命中全部 |
|---|---:|---:|---:|---:|
| Dense5 | 138 | 90 | 29 | 19 |
| Qwen 4B | 121 | 62 | 29 | 30 |

注意分母：Qwen 的30/121是“全零题中标注ID全部命中”，不是
“全部命中时的失败率”；后者还需要全部命中题总数。
也不能据两组全零条件子集的数量比较推断 Qwen 更容易误用证据。

20题样本中13题无 qrel ID、5题全部命中、2题部分命中。
这些只是 **文档ID覆盖**：
例如 sabbatical/HELOC 问题即使没命中 gold ID，其他选中文档仍明确含 HELOC。
所以“Recall=0 → 没有正确证据”在具体案例上不成立。

## 二、没有发现本地 prefix 截断，但发现范围/条件处理问题

全部700个保存的用户 prompt 与 corpus 的完整 top-5 原文精确相等，
也与保存的 prompt 哈希一致。此次 fresh full-document 结果的错误
**不能解释成本地 context builder 切掉尾部**。
这不证明远程服务内部绝无截断；本次没有验证其内部处理。

最有价值的三个后续候选：

1. **Ireland pension — `q_371c6ca10bf1bf5e`**
   两个 gold 文档都已进入 prompt；同时混有明确 UK allowance、US
   retirement account 和公司税语境。回答把 carry-forward/offset against
   future profits 等合成爱尔兰养老金优点，没有保留适用范围。
   这是“范围混合”的具体证据，不只是 judge 说它不对。
2. **Stocks on sale — `q_1024d33a7dfa33ef`**
   回答把出售股票时的税基确认要求列成识别便宜股票的因素。
   对应离题文档可直接定位；适合研究“无关细节被纳入”，但尚无删除干预。
3. **Car financing — `q_ce474d8f8d548797`**
   第二篇 gold 明确 dealer 只关心及时无风险收到钱；回答整合其他融资建议，
   却遗漏该观点。不过问题本身很简略，也可能允许一般融资比较，不能绝对化。

反例同样重要：Tangerine 案例 gold 很短且已排第一，答案仍省略
到账计时从收件人完成操作之后开始这一条件。更短 context 不保证条件保留。
另有末日资产案例把 gold/silver 的 **after crisis** 混成 **during crisis**；
只抽关键词可能进一步恶化范围和条件丢失。

## 三、评估限制不能混同生成错误

本地 judge 实现明确要求 **只按 gold evidence** 评判，而生成器看到的是
实际检索的 top-5。两者证据范围不同：

- expense-ratio 回答的 prospectus、quarterly reports 有检索来源，
  但不在其 gold；这不是凭空编造的充分证据。
- technical-analysis 回答覆盖 gold 中的三个软件，也有其他检索支持的推荐，
  但仍获全零。不能把该分数直接理解为“完全没回答”。

样本中标记 **6/20 条 judge 文本冲突/误述待复核**，不是6条已纠正分数，
更不是总体 judge 错误率：

| Query | 可直接核对的冲突 |
|---|---|
| `q_32480a484879c3ec` | 理由说答案 grounded、证据不足判断正确，结果仍三维全零 |
| `q_3bd06eaf675d1a65` | 理由指责归责 credit bureaus，答案实际写 creditors |
| `q_66bb08d4774227f7` | 理由称答案说 after clicking the link，原回答没有该表述 |
| `q_c3dad6ee3fe9e058` | 理由把主要列个人扣除的答案描述成 business tax considerations |
| `q_e16177bbeb991130` | 理由对 Bloomberg 同一措辞一面说 unsupported，一面声称 gold 含它 |
| `q_e86dfaec0aa23a50` | 理由否认 helicopter、connections 等在 gold，原文实际包含 |

这些问题与真实生成错误可以并存。比如末日资产的危机时间范围确有误用；
不能因 judge 误读其他点，就宣布整个回答正确或直接补分。
本次没有重评，也没有用新增评审推翻原有配对区间。

## 四、是否继续做片段实验？

**目前不建议批量增加 generation/judge 调用。** 原因：

1. 样本显示证据缺失、题意/参考情境不匹配、范围混合、条件省略和评估问题并存。
2. 只找出三个值得进一步核实的片段候选，尚不足证明普遍收益。
3. 若优化只让回答贴近 gold-only judge，分数可能提高而真实质量未改善。

若作者确认案例并明确授权，可先对三个候选设计干预，再决定是否扩大：
保持问题、原始 top-5 集合、顺序、prompt 和生成设置固定；
预先定义抽取算法，并检查条件/例外保留。
长度与内容同时改变时不能把全部变化解释为“消除干扰”。
若用 gold 或人工答案知识挑片段，应明确称 **oracle diagnostic**，
不能声称它是可部署的 query-only selector。
本次没有实施该干预，也没有测量 snippets 的质量或速度收益。

## 对报告的意义

更稳妥的分析不是“正确文档已检索到，所以 LLM 推理弱”，而是：

> Document-ID recall does not establish that an answer uses the correct scope
> and conditions. Exploratory practice cases show cross-context scope blending
> and condition omission, while reference-only judging introduces a separate
> evidence-coverage limitation. These observations motivate targeted diagnostics,
> not a demonstrated benefit from snippet selection.

如纳入报告，应明确是 AI 辅助探索性案例分析，作者需要核对原文；
不能称作独立人工验证、显著性实验或新 dev 结论。
