# ICF RAG 基线运行指南

## 本次运行（2026-11-11）

本次已成功完成。作业编号 **3713067**，节点 **landonia11 / NVIDIA RTX A6000**，状态 **COMPLETED**，退出码 **0:0**，GPU 作业用时 **45 分 18 秒**（含首次环境加载、编译、测试及评估，不是纯推理时间），于英国时间 18:37 结束，GPU 已释放。

课程资源校验、3 项接口测试、4 项 CUDA 测试均通过。PyTorch 参考 ASR 和自写 CUDA ASR 均跑通第一条完整问答，转录及前五个检索文档 ID 一致。350 条 practice 查询已全部评估并缓存转录；另生成并评分了 5 条示例答案。

全集检索指标：Recall@5 **0.421337**，Recall@10 **0.479594**，Recall@20 **0.578524**，Recall@100 **0.769856**，MRR@10 **0.488746**，NDCG@10 **0.420461**。

5 条示例答案平均正确性 **1.6/4**、证据支持 **2.0/4**、完整性 **1.2/4**，归一化平均分 **0.40**。这是首五条的诊断性子集，评判器为课程默认 EuroLLM、每条只评一次，不能视为全集答案质量。

全部日志、版本记录、结果和缓存保存在[本地结果目录](results/icf-3713067/)。关键文件包括[检索日志](results/icf-3713067/retrieval-practice.log)、[转录缓存](results/icf-3713067/practice_transcripts.jsonl)、[答案评分汇总](results/icf-3713067/sample_judge/answer_judge_summary.json)、[Slurm 状态](results/icf-3713067/slurm-accounting.txt)和[实际提交脚本](results/icf-3713067/submitted-script.sbatch)。

```bash
squeue -j 3713067
tail -f ~/edin-mls-26-autumn/rag/results/icf-baseline-3713067.log
```

## 1. Windows → WSL → ICF

连接大学 VPN，在 PowerShell 进入 Ubuntu：

```powershell
wsl -d Ubuntu
```

在 **WSL Ubuntu** 中登录，手动输入 DICE 密码：

```bash
ssh -M -S /tmp/mls-icf-session -o ControlPersist=8h \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=6 \
  -J s2564099@student.ssh.inf.ed.ac.uk s2564099@icf
```

`-J` 指定学生跳板机。`-M -S` 建立可复用的 SSH 连接，供其他 WSL 终端使用；不申请 GPU。密码输入时不回显。

`ServerAliveInterval=30` 每 30 秒发送一次连接保活；连续六次没有响应才断开。VPN 中断时，重新运行这条命令建立连接。

`ControlPersist=8h` 控制最后一个客户端退出后共享连接的后台空闲保留时间，不是 SSH 会话或 GPU 作业的最长时长。

登录后在 **ICF 登录节点**执行：

```bash
hostname
pwd
sinfo -p Teaching
squeue -u s2564099
```

登录节点用于下载、准备环境和提交作业，模型推理通过 Slurm 在计算节点运行。

## 2. 本次目录与环境

- 代码：`/home/s2564099/edin-mls-26-autumn`，版本 `c0b7a7f`。
- ASR：`/home/s2564099/mls-asr-env`，Python 3.11、PyTorch 2.10.0/cu128、Transformers 5.0.0。
- RAG：`/home/s2564099/mls-rag-env`，独立虚拟环境，使用 CPU PyTorch。
- 解压资源：`/home/s2564099/mls-rag-assets`。
- 凭据：仓库内 `rag/.env`，Git 忽略，远程权限 `600`。
- 安装日志：仓库内 `rag/results/icf-prepare.log`。

原有 `~/venv` 不作修改。这里的版本是本次脚本指定值；最终实际版本以安装和作业日志为准。

## 3. 首次准备

本次已从 Jack 的仓库克隆代码，并上传本地完整 ZIP 和运行脚本。另建一份仓库时可参考：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/JackSuuu/edin-mls-26-autumn.git ~/edin-mls-26-autumn
```

`GIT_LFS_SKIP_SMUDGE=1` 跳过资源自动下载；随后需要通过 Git LFS 获取资源或上传完整 ZIP。

从另一个 **WSL 终端**上传本地资源：

```bash
cd /mnt/d/Desktop/uni_of_edinburgh/year4/machine_learning_systems/cw/edin-mls-26-autumn
scp -o ControlPath=/tmp/mls-icf-session rag-assets.zip \
  s2564099@icf:/home/s2564099/edin-mls-26-autumn/
```

在 **ICF 登录节点**启动准备脚本：

```bash
cd ~/edin-mls-26-autumn
mkdir -p rag/results
nohup bash rag/scripts/icf_prepare.sh > rag/results/icf-prepare.log 2>&1 < /dev/null &
tail -f rag/results/icf-prepare.log
```

`nohup` 使安装在 SSH 断开后继续；重定向保存正常输出与错误。按 `Ctrl+C` 只停止查看日志。安装结束前不要重复启动。成功标记是 `ICF_PREPARATION_COMPLETE`；脚本只下载和安装，不运行模型推理。

准备脚本的 pip 使用 `--no-compile`，跳过可选的 Python 字节码预编译，减少共享文件系统上的小文件读写。首次安装仍可能较慢；日志暂时没有新增不一定表示失败，可先检查进程是否还在。

需要配置 ELM 凭据时，输入不回显：

```bash
cd ~/edin-mls-26-autumn
umask 077
read -rsp 'ELM API key: ' elm_key
printf '\n'
printf 'ELM_API_KEY=%s\n' "$elm_key" > rag/.env
unset elm_key
chmod 600 rag/.env
git check-ignore rag/.env
```

最后一条只检查 Git 忽略，不输出密钥。本次已配置，无需重复。

可以用以下命令检查 ELM 认证和课程默认模型是否在列表中；它不会生成答案，也不打印密钥：

```bash
~/mls-asr-env/bin/python rag/scripts/icf_probe_elm.py
```

## 4. 提交 GPU 作业

环境准备成功后，在 **ICF 登录节点**运行：

```bash
cd ~/edin-mls-26-autumn
mkdir -p rag/results
sbatch rag/scripts/icf_baseline.sbatch
```

默认申请 Teaching 分区的一张 RTX A6000、2 个 CPU、24 GB 主存、最多 2 小时。`--mem` 是主存，不是 GPU 显存。Slurm 返回编号只表示提交成功。

```bash
squeue -u s2564099
scontrol show job JOB_ID
tail -f rag/results/icf-baseline-JOB_ID.log
```

`JOB_ID` 换成本次编号。`PD` 是排队，`R` 是运行。SSH 断开不会取消批作业。

脚本依次记录 GPU/软件/代码版本，编译并运行四项 CUDA 测试，检查 RAG 配置，用两种 ASR 运行同一条完整问答，评估 350 条 practice 查询的检索，最后生成并评分 5 条示例答案。

脚本关键步骤失败即停止，只有全部成功才打印 `ICF_BASELINE_COMPLETE`。输出位于 `rag/results/icf-JOB_ID/`。

## 5. 作业内关键命令及含义

以下由 GPU 批作业执行，不在登录节点直接运行：

```bash
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /home/htang2/toolchain-20251006/toolchain.rc
export RAG_ASR_PYTHON="$HOME/mls-asr-env/bin/python"
export PATH="$HOME/mls-rag-env/bin:$PATH"
export HF_HUB_OFFLINE=1
cd ~/edin-mls-26-autumn
make -C hw1-asr test
python rag/scripts/check_setup.py
ASR_CUDA=none python rag/scripts/run_audio_query.py --device cpu
ASR_CUDA=all python rag/scripts/run_audio_query.py --device cpu
```

`source` 加载 CUDA 编译器和库路径；`RAG_ASR_PYTHON` 选择 ASR worker 的环境；`PATH` 选择主进程的 RAG 环境。离线开关使用提前下载的模型。

`--device cpu` 只把查询编码器放到 CPU。ASR 始终在 GPU 上运行；`ASR_CUDA=none` 仅切换为 PyTorch 参考算子。

[context_builder.py](src/context_builder.py) 在 CPU 上把检索文档拼成提示词，本身不需要 GPU；答案由远程 ELM 生成。

## 6. 确认成功、取消与下载结果

在 **ICF 登录节点**检查：

```bash
sacct -j JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES
tail -n 30 rag/results/icf-baseline-JOB_ID.log
ls rag/results/icf-JOB_ID
```

同时确认 `COMPLETED`、`ExitCode 0:0` 与完成标记。任务从队列消失不等于成功。

取消自己的指定作业：

```bash
scancel JOB_ID
```

在 **WSL 本地项目目录**下载结果：

```bash
mkdir -p rag/results
scp -r -o ControlPath=/tmp/mls-icf-session \
  s2564099@icf:/home/s2564099/edin-mls-26-autumn/rag/results/icf-JOB_ID \
  rag/results/
```

保留日志、GPU 型号、版本和 ASR 转录缓存。后续固定转录比较 RAG；变更 ASR 时使用新缓存。单查询计时不含模型加载，但可能包含首条推理冷启动，不能单凭它推断稳态延迟。

缓存齐全后，检索评估可不再启动 ASR，因此不必申请 GPU。CPU 编码和批量评估仍应在本地 WSL 或获分配的计算资源中执行，不要在 ICF 登录节点上跑全集推理。

参考：[课程 RAG 指南](README.md)、[Hao Tang 的 ICF 教程](https://homepages.inf.ed.ac.uk/htang2/notes/icf/)。
