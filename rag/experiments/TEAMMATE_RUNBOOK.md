# RAG 实验交接（队友负责）

ASR 四 kernel + baseline 已完成。你这边完成 **investigation 扩规模 + 写进 report**，jack 不再跑 RAG 实验。

## 研究问题（已定，勿换题）

检索已有 top-100，但生成默认只用 top-5 → **证据怎么选**（篇数 / token 预算 / cross-encoder 重排）对 **标注覆盖、答案分、输入 token 成本** 的影响？

试点结论见 `selection-pilot-results.md`（30 题）。你的任务是：**同方法、更可比的规模/耗时**，不是新开一个方向。

## 已完成（勿重复烧 ELM）

| 项目 | 位置 |
|------|------|
| Baseline 检索 350 + 转写 cache | `rag/results/icf-3713067/practice_transcripts.jsonl` |
| 试点 30×5 臂 + judge | `selection-pilot-metrics.json`，详情见 pilot md |
| 代码 | `rag/extensions/*`，`evidence_selection.py` |

## 你还要做（按顺序）

### 1. 全 practice 重排序（核心缺失）

- **目标**：350 题 × 100 候选 cross-encoder 分数；记录 **同节点** `reranking_ms`。
- **GPU**：`prepare` 加 `--rerank-device cuda`（已支持；此前 bug 是写死 CPU）。
- **不要**在这步跑 ASR；用冻结 transcript + `collect_retrieval_diagnostics.py` 的 rankings。
- 粗算：CPU 2 核 ~26 s/题 → 350 题 ~2.5 h；GPU 应快很多（需在 report 写清 device/batch）。

```bash
# 仓库根目录，GPU 节点，已 activate mls-rag-env
python rag/extensions/collect_retrieval_diagnostics.py \
  --transcript-cache rag/results/icf-3713067/practice_transcripts.jsonl \
  --output rag/results/selection-rankings-practice-full

# 全 350 题 sample：自建 jsonl（每行 query_id + transcript，与 cache 一致）
python rag/extensions/run_selection_experiment.py prepare \
  --sample rag/experiments/<practice350>.jsonl \
  --rankings rag/results/selection-rankings-practice-full/rankings.jsonl \
  --rerank-device cuda \
  --output rag/results/selection-run-full/plan
```

### 2. 候选池深度（试点建议的下一步）

从 **top-20 / 50 / 100** 池子里 rerank 再取 top-5，比 selected-ID recall。  
试点事后结果见 `selection-candidate-depth-diagnostic.json`；你要 **正式 prepare 配置** 或扩展脚本，不要只读 JSON。

### 3. 生成 + judge（按需花钱）

- 若只关心 **检索/重排质量**：rerankings + 覆盖指标即可，不必 350×5 次 ELM。
- 若正文要写 **答案分**：沿用 pilot 五臂，用 `--max-new-queries` 分批 + `--resume`；统一 `--json-mode` judge（见 `extensions/README.md`）。
- **dev 125**：方法定稿后再跑，practice 上迭代。

### 4. Report（6 页正文）

- 压缩 ASR（jack 侧），**RAG 半页 baseline + 1.5 页 investigation**。
- 把 pilot 数字写进 `report.tex`；全量 rerank 完成后替换耗时表。
- `python report/format_checker/check_report_format.py report.pdf`
- Group ID / 队友 UUN 填好。

## 资源约定

| 步骤 | 设备 |
|------|------|
| ASR | GPU（已完成，cache 已冻） |
| enc_a / 点积检索 | CPU |
| 重排序 | **建议 GPU**（`--rerank-device cuda`） |
| ELM | 远端 API |

## 联系人分工

- **jack (s2510156)**：ASR kernel、baseline 记录、report ASR 段、Task4 shuffle 修复、`--rerank-device`。
- **队友**：本 runbook 中 1–4 步、最终 PDF、dev（可选）。
