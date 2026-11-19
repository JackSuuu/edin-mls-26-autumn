# Audio RAG recording script

成片目标约 8:35，控制在 9:45 以内。下方时间从开录算起，最后两分钟留给 Question 1 和 Question 2。只念英文引用块。中文是画面和操作提示。

## 录制前

1. 在本机 Terminal 开隧道，保持窗口打开。

   ```powershell
   ssh -N -L 18765:127.0.0.1:18765 jacksu@fuji3
   ```

2. 打开 [本地 Visualizer](http://127.0.0.1:18765)，确认页首显示 **Audio RAG / Console** 和 **From audio to answer.**。如果 fuji3 上还没开服务，在单独的终端启动。隧道和服务都用 `18765`。

   ```bash
   cd ~/edin-mls-26-autumn
   python3 -u rag/demo/server.py --port 18765 \
     --events rag/results/video-demo/events.jsonl
   ```

3. 在 fuji3 的查询终端设置环境。使用已分配的 GPU，保留现有的 `CUDA_VISIBLE_DEVICES`。

   ```bash
   cd ~/edin-mls-26-autumn
   export RAG_ASR_PYTHON="$HOME/miniconda/envs/mls-asr/bin/python"
   export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
   export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
   PY="$HOME/mls-qwen-dev-env/bin/python"
   EVENTS=rag/results/video-demo/events.jsonl
   QUERY=q_7e9a0f2716c017d2
   ```

4. OBS 主画面放浏览器。终端裁到命令和进度，隐藏账号、主机提示符、个人路径和凭据。这一页包含整段演示。顶部的三个链接可以直接跳到 Evaluation results、Document chunking 和 ASR benchmark。排练时录下一次实际运行，保留命令、输入和完整输出，供正式录制时剪入。

## 0:00–1:55 Pipeline

画面停在页面顶部的 **Pipeline**。在终端开始请求，然后切回页面。先指五个步骤。转写出现后，往下指 **ASR transcript**，再回到 **Select evidence** 看进度。答案准备好后，指两栏排名和 **EuroLLM answer**。

```bash
"$PY" -u rag/demo/run.py --events "$EVENTS" --query-id "$QUERY" --method qwen --judge
```

> We start with a spoken question about sales tax in a retail store. Under ASR transcript, you can see the text that goes into the search. Dense search returns one hundred candidates. The counter here shows how many of those documents Qwen-4B has scored.
>
> The list on the left shows the first five from the original search. The one on the right shows the five we sent to EuroLLM after reranking. The tags tell us which documents kept a place in the top five and which came from further down the search results. You can scroll down to read a preview of each selected document. EuroLLM gets the full text and writes the answer beside the previews.

评分出现后，依次指 **Audio → answer**、**EuroLLM judge** 和计时卡里的 **Evaluate (separate)**。如果还在评分，先讲计时。可以指出屏幕上实际出现的一处排名变化。

> The clock stops when the complete answer comes back. Then the judge reads the answer against the course reference evidence. Here we can see its three scores and the explanation for this question. The timing card lists evaluation separately, so the answer time is the wait up to the point where the user can read the result.

## 1:55–3:15 Evaluation results

点页面顶部的 **Evaluation results**。先顺着表格的 **Recall@5** 一列往下看，再横向指 **Mean judge score** 和 **Selection + generation**。最后把鼠标移到表格下方的 **Separate end-to-end test · 10 audio queries**。

> This table sums up the results across three hundred and fifty practice questions. Recall at five is the share of labelled documents we found in the five we sent. That went from about forty-two percent with Dense5, to forty-seven with the smaller Qwen, to fifty-three with Qwen-4B.
>
> The score column tells a less clear story. The smaller Qwen found more labelled documents, and its average answer score was slightly lower than Dense5's. The four-billion-parameter model had the highest observed score, at zero point three seven. Its interval against Dense5 includes zero, so we're more confident about the gain in recall.
>
> The last column estimates the time spent choosing the evidence and generating an answer. The ten-audio test underneath measures the full wait from audio to answer. That went from about nine seconds with Dense5 to seventeen with Qwen-4B. Most of the extra time went into reranking.

## 3:15–5:25 Document chunking

点顶部 **Document chunking**。从卡片上的买房问题开始，把鼠标停在 **Document rank 8** 和 **Chunk-based rank 3**。先讲为什么在五篇内切分够不到这篇，再讲先切二十篇时怎么重排。

> The next example asks how to buy a home. We ran it offline with a limit of seven hundred and sixty-eight tokens for the question, instructions and evidence together.
>
> In our first approach, we picked five documents by document score. Then we split those five into short passages and chose which passages to fit into the prompt. The source on this card was ranked eighth, so that method had already left it out before we started splitting.
>
> In the second approach, we split all twenty candidates first. We gave each document the score of its best passage and then picked five. You can see what happened to the same source here. Its best passage moved it from eighth to third, which brought it into the group we could use.

指卡片上的引用，沿着首付、应急储蓄、还债依次划过。说到 **3 of 12** 时指卡片下方加粗数字。答案用了哪些内容，直接用卡片下方的说明承接。

> This is the passage EuroLLM received. It puts saving for a down payment next to building an emergency fund and paying down debt. The answer mentioned all three. We can trace those points back to the passage on this card.
>
> On questions where a labelled source was already in the prompt, cutting up the same five documents gave us no clear score gain. When the labelled sources started outside the five, scoring passages across all twenty brought labelled text into the prompt for three out of twelve questions. The house-buying example shows what one of those recoveries looked like.

## 5:25–6:35 ASR benchmark

点 **ASR benchmark**。先指 **4/4 test suites passed**，再指 **PyTorch 521 ms** 和 **All CUDA 1,986 ms**。讲到线性运算时，指下方加粗的 **84%**。

> At the start of the pipeline, we implemented four groups of CUDA kernels for speech recognition. Our attention kernel reads the input in blocks so it can avoid writing out the full attention matrix. On the screen you can see that all four test suites passed and that the CUDA version reproduced the reference transcription.
>
> We timed the two configurations on the same clip. GPU PyTorch took about half a second. Enabling all our custom kernels took almost two seconds, about three point eight times as long. Linear operations made up about eighty-four percent of the CUDA kernel time. That's where we'd start if we wanted to make this version faster.

## 6:35–7:35 Question 1 · Your main design choice

回到 Pipeline 的两栏文档排名。说 **Question one**。说到两种选法时，分别指 **Dense search · top 5** 和 **Selected for EuroLLM**。讲到 9 秒和 17 秒时，指 **Evaluation results** 表格下方的十条音频实测。

> Question one. We decided to send EuroLLM five documents and test two ways of choosing them. We wanted users to get an answer quickly, so we looked at how much text the model had to read. In our first experiment, sending twenty documents nearly quadrupled the input, and the average answer score barely moved.
>
> So we kept five and compared the original search order with Qwen reranking. Both methods chose from the same hundred candidates and used the same answer model. Qwen got more labelled documents into the top five. In the ten-audio test, the average wait went from about nine to seventeen seconds. We'd use Dense5 when response time is the priority.

## 7:35–8:35 Question 2 · Checking your work

回到 **EuroLLM judge**。说 **Question two** 后，先指三项分数和 **Normalised score**。后半段讲已经跑过的离线复核，画面停在评分卡即可。若本次请求的评分尚未返回，用彩排时保存的评分画面。

> Question two. We checked that the scores on this card matched what the judge returned. The card shows three marks out of four. We add them and divide by twelve to get the final score. We did that again from the raw judge responses for all seven hundred answers in one paired practice run. Every score matched the saved result.
>
> For example, we saw a lot of zeros and wanted to know if the code was misreading the judge's reply. We fed it a broken reply, and the parser raised an error. For the saved zero scores, we could go back to the original responses and see that the judge had returned zero for all three marks.

## 备用画面

如果录制当天 ELM 或 GPU 不可用，剪入彩排时录下的实际运行。存档可以补充展示转写、检索和答案。页面会显示 `Recorded replay`，回放时念这一句。

> This is a replay of a saved request.

```bash
RECORDING=rag/results/paired-audio-cost-20261118-233916/rows.jsonl
python3 -u rag/demo/run.py --events "$EVENTS" --method qwen --replay "$RECORDING"
```

存档使用另一段音频，回放时按页面上的实际问题介绍。两栏排名与单题评分用彩排画面。如果剪短等待，在画面上加 `Waiting time shortened`，保留原始的请求计时。
