# ICF GPU：从 Windows / WSL2 到实际运行作业

更新：2026-11-12。供用户与后续 AI 会话复用。先读第 1–3 节恢复连接，再按需要读取环境准备、GPU 申请或故障排查；已有环境和作业不要重复创建。

## 1. 已验证的路径与证据边界

连接路径：Windows PowerShell → WSL2 Ubuntu 的 OpenSSH → 学生跳板机 → ICF 登录节点 → Slurm 分配的计算节点。

- 用户：`s2564099`；跳板：`student.ssh.inf.ed.ac.uk`；集群入口：`icf`；本次登录节点为 `hastings`。
- 2026-11-11，Teaching 作业 **3713067** 在 `landonia11` 的一张 **RTX A6000** 上完成 ASR CUDA 测试和 RAG baseline：`COMPLETED`，`ExitCode=0:0`，45 分 18 秒。该时长包括编译、初始化和评估，不是纯推理延迟。
- 先前作业 **3667131** 在 `saxa` 申请 `gpu:h200:1`，PyTorch 可见 **139.80 GiB** 显存，并完成 CUDA 张量计算。这只证明该次 H200 检测成功，不证明某个服务框架已跑通。
- GPU 型号、GRES、分区、权限和可用数量都可能变化。历史成功不等于当前空闲或永久访问资格。
- 本指南没有密码、API key 或私钥。用户在自己的终端交互输入密码；不从旧聊天提取凭据。

AI 接手时先检查当前项目、WSL 发行版、SSH 主连接、现有作业和环境。用户说“已登录”，可能只是在另一个终端拥有普通 SSH 会话；必须检查下面的控制 socket，不能假设能接管那个终端。

## 2. Windows → WSL2 → ICF

### 2.1 在 Windows PowerShell 中进入 WSL

先连接学校 VPN，然后执行：

```powershell
wsl --list --verbose
wsl -d Ubuntu
```

确认 Ubuntu 的 VERSION 为 2；若发行版名称不同，使用列表中的实际名称。这个流程使用 **WSL 内的 Linux ssh/scp**。不要把 PowerShell 的路径、换行规则和 Linux shell 混用，也不要为复用连接改回 Windows 原生 ssh。

本项目的 Windows 路径对应 WSL 路径：

```bash
cd /mnt/d/Desktop/uni_of_edinburgh/year4/machine_learning_systems/cw/edin-mls-26-autumn
```

### 2.2 在 WSL 中检查、建立可复用连接

先检查已有主连接：

```bash
ssh -S /tmp/mls-icf-session -O check s2564099@icf
```

出现 `Master running` 表示 socket 对应的 SSH 进程仍在；再用 `hostname` 检查网络确实可用。若没有主连接，用户在 WSL 中运行：

```bash
ssh -M -S /tmp/mls-icf-session \
  -o ControlPersist=8h \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=6 \
  -J s2564099@student.ssh.inf.ed.ac.uk s2564099@icf
```

可能分别要求跳板机和 ICF 的 DICE 密码；输入时不回显。首次主机指纹按学校可信信息核对，不使用关闭主机校验的选项。

- `-J`：通过学生跳板机访问 ICF。
- `-M -S`：创建本 WSL 用户可复用的 SSH 主连接，控制 socket 在 WSL 的 `/tmp`。
- `ControlPersist=8h`：最后一个客户端退出后，后台主连接可空闲保留 8 小时。它不是活动会话的最长时长，也不是 GPU 作业时限。
- `ServerAliveInterval=30` / `ServerAliveCountMax=6`：每 30 秒检查服务器响应，连续 6 次失败后断开；无法抵消 VPN 断开、休眠或 WSL 重启。

需要更长空闲复用时间，可在下次建立主连接时改为 `ControlPersist=24h`。后开的客户端不会把已经运行的主连接自动改成 24 小时。不要为修改时长关闭别人正在使用的主连接。

### 2.3 另一终端或 AI 会话复用连接

在 **同一个 Windows 用户、同一个 WSL 发行版、同一个 Linux 用户**下执行：

```bash
ssh -S /tmp/mls-icf-session -o BatchMode=yes s2564099@icf hostname
ssh -S /tmp/mls-icf-session s2564099@icf
```

第一条适合 AI 非交互检查；第二条打开交互会话。若检查失败，停止依赖此连接的操作，让用户重新执行上面的登录命令，不循环尝试密码。

AI 从 Windows PowerShell 执行简单远程检查时可用：

```powershell
wsl -d Ubuntu -- ssh -S /tmp/mls-icf-session -O check s2564099@icf
wsl -d Ubuntu -- ssh -S /tmp/mls-icf-session -o BatchMode=yes s2564099@icf hostname
```

复杂远程命令写成 LF 换行的 `.sh` / `.sbatch`，用 WSL 的 `scp` 上传，再远程执行文件。避免在 PowerShell → WSL → SSH 的多层字符串中嵌套 `$HOME`、`$!`、命令替换和凭据；不同 shell 的展开顺序容易改变命令。

## 3. 登录后先看节点、队列和环境

下面在 **ICF 登录节点**执行：

```bash
hostname
pwd
sinfo -p Teaching
sinfo -N -O partition:20,nodelist:24,gres:100
scontrol show partition Teaching
squeue -u s2564099
test -x /home/s2564099/mls-asr-env/bin/python
test -x /home/s2564099/mls-rag-env/bin/python
test -r /home/htang2/toolchain-20251006/toolchain.rc
```

`test` 成功时不输出，退出码为 0。先看 MOTD 的最新要求。登录节点只做轻量查询、文件传输、下载和环境准备，不直接跑模型、全集编码、训练或 CUDA 测试，也不递归扫描整个 `/home` 或 `/afs`。

GPU 由 Slurm 分配。SSH 到 `icf` 只获得登录节点，`nvidia-smi` 或 `torch.cuda.is_available()` 在登录节点的结果不能判断整个集群有没有 GPU。

## 4. 指定 GPU：按 GRES 申请，按实际输出确认

先查询当前 GRES，再选择。以下是历史上见过的资源名：

- **RTX A6000，48 GB**：`gpu:nvidia_rtx_a6000:1`，`landonia11`，本次 baseline 已验证。
- **H200 全卡**：`gpu:h200:1`，`saxa`，先前已验证；物理标称 141 GB，历史 PyTorch 可见 139.80 GiB。当前可能切成 MIG。
- **H200 MIG**：`gpu:h200_3g.71gb:1`、`gpu:h200_1g.18gb:1`，只在当前 `sinfo` 确实列出且分区允许时申请；`3g` 不是三张卡，也不能当完整 H200。
- **RTX 2080 Ti，11 GB**：`gpu:nvidia_geforce_rtx_2080_ti:1`，历史 Teaching 节点包括 `damnii`、部分 `landonia`、`opencast`。显存是否足够必须按任务检查。
- **其他历史资源**：`a40`、`nvidia_l40s`、`nvidia_h200` 等；分区和权限见当前查询，不能把另一个分区的资源名直接用于 Teaching。

下面是同一脚本的三种**替代提交方式**，选择一种，不要全部提交。文件需已在当前目录：

```bash
sbatch -p Teaching --gres=gpu:nvidia_rtx_a6000:1 gpu-check.sbatch
```

```bash
sbatch -p Teaching --gres=gpu:h200:1 gpu-check.sbatch
```

```bash
sbatch -p Teaching --gres=gpu:1 gpu-check.sbatch
```

最后一种允许 Slurm 选择任意满足条件的 GPU，不适合要求固定显存、架构或跨实验同卡比较的情况。通常申请型号即可；只有确有节点要求时才加 `--nodelist=landonia11`，否则会额外限制调度。

`sinfo` 的 GRES 数量是配置资源量，不是空闲卡数；`mix` 不等于没有空闲卡，`idle` 不保证当前用户有资格。`Teaching*` 的 `*` 代表默认分区，提交写 `Teaching`。

`--mem=24G` 是**主存**，不是显存；`--cpus-per-task=2` 是 CPU 核数；`--time=02:00:00` 才是作业墙钟时限。不要手动覆盖 Slurm 设置的 `CUDA_VISIBLE_DEVICES` 去选择物理卡号。进程里的 `cuda:0` 通常指分配给它的第一张可见卡。

## 5. 代码、资源与环境：优先复用这次已成功的安装

已验证的远程位置：

- 代码：`/home/s2564099/edin-mls-26-autumn`。
- ASR：`/home/s2564099/mls-asr-env`，Python 3.11、PyTorch 2.10.0+cu128、Transformers 5.0.0。
- RAG：`/home/s2564099/mls-rag-env`，CPU PyTorch 2.10.0、Transformers 5.0.0；baseline 其余包版本以该次 `pip freeze` 为准。
- 原始解压资源：`/home/s2564099/mls-rag-assets`。
- 已使用 CUDA 工具链：`/home/htang2/toolchain-20251006/toolchain.rc`，每次先检查存在。

ASR worker 使用 `RAG_ASR_PYTHON` 指定的解释器，主进程使用 RAG 环境。不要直接升级成功环境的 Torch 来迁就另一个框架。新框架另建环境，并核对 GPU 架构、驱动、Torch wheel 和扩展的兼容性。

### 5.1 新目录才需要克隆

在 **ICF 登录节点**，确认目标目录不存在后：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/JackSuuu/edin-mls-26-autumn.git /home/s2564099/edin-mls-26-autumn
```

这会跳过 LFS 自动下载，随后必须取得真正的资源 ZIP。已有仓库先 `git status --short` 和 `git rev-parse HEAD`，保留未提交修改，不自动 reset/覆盖。

在 **WSL 本地项目目录**上传完整资源与本地运行脚本；这些个人脚本未必在远程 Git 中：

```bash
cd /mnt/d/Desktop/uni_of_edinburgh/year4/machine_learning_systems/cw/edin-mls-26-autumn
scp -o ControlPath=/tmp/mls-icf-session rag-assets.zip s2564099@icf:/home/s2564099/edin-mls-26-autumn/
scp -o ControlPath=/tmp/mls-icf-session rag/scripts/icf_prepare.sh rag/scripts/icf_baseline.sbatch s2564099@icf:/home/s2564099/edin-mls-26-autumn/rag/scripts/
```

先查看文件是否存在，不能上传一个小型 LFS pointer 当作 ZIP；本次完整 ZIP 为 196521100 字节，可用 `unzip -t rag-assets.zip` 检查。后续版本大小变化时以资源校验为准。

### 5.2 首次安装才执行，不重复安装已经成功的环境

本次采用项目个人准备脚本。在 **ICF 登录节点**：

```bash
cd /home/s2564099/edin-mls-26-autumn
mkdir -p rag/results
nohup bash rag/scripts/icf_prepare.sh > rag/results/icf-prepare.log 2>&1 < /dev/null &
tail -f rag/results/icf-prepare.log
```

成功标记 `ICF_PREPARATION_COMPLETE`；安装结束前不启动第二份。`Ctrl+C` 只停止 `tail`。`nohup` 用于这段下载/安装流程，不能替代 Slurm 运行模型。

若脚本不存在，下面是关键安装步骤，供**确认要创建新环境**时参考。`/opt/conda/bin/conda` 必须实际存在；这些步骤针对该课程，不是所有 GPU 项目的通用依赖：

```bash
test -x /opt/conda/bin/conda
/opt/conda/bin/conda create -y -p /home/s2564099/mls-asr-env --override-channels -c conda-forge python=3.11
/home/s2564099/mls-asr-env/bin/python -m pip install --no-cache-dir --no-compile torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
/home/s2564099/mls-asr-env/bin/python -m pip install --no-cache-dir --no-compile transformers==5.0.0 numpy safetensors huggingface_hub
/home/s2564099/mls-asr-env/bin/python -c 'from huggingface_hub import snapshot_download; snapshot_download("zai-org/GLM-ASR-Nano-2512", revision="61ba4e0b3309b6656edea3e93e419f7bd5c61957")'
/home/s2564099/mls-asr-env/bin/python -m venv /home/s2564099/mls-rag-env
/home/s2564099/mls-rag-env/bin/python -m pip install --no-cache-dir --no-compile torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
cd /home/s2564099/edin-mls-26-autumn
printf 'torch==2.10.0\ntransformers==5.0.0\n' > /home/s2564099/mls-rag-env/course-constraints.txt
/home/s2564099/mls-rag-env/bin/python -m pip install --no-cache-dir --no-compile -c /home/s2564099/mls-rag-env/course-constraints.txt -r rag/requirements.txt
mkdir -p /home/s2564099/mls-rag-assets
unzip -q -n rag-assets.zip -d /home/s2564099/mls-rag-assets
/home/s2564099/mls-rag-env/bin/python rag/scripts/install_assets.py --from-dir /home/s2564099/mls-rag-assets
```

下载固定 revision 是为了复现 baseline 的模型；原准备脚本默认下载当时的最新 revision。其余未固定依赖可能随时间变化，严格复现应参照历史包清单选择兼容版本，而不是宣称以上命令能恢复完全相同的环境。

`--no-compile` 跳过 Python 字节码预编译，减少共享盘小文件操作；模型提前下载到共享 home 的 Hugging Face 缓存。作业内 `HF_HUB_OFFLINE=1` 只允许读现有缓存，不能帮忙下载缺失模型。

### 5.3 ELM 凭据仅在确实缺失时配置

用户在自己的 **ICF 终端**交互输入，不把 key 放在命令参数、技能、日志或 Git 中。以下只用于新建缺失的配置；已有 `.env` 不覆盖：

```bash
cd /home/s2564099/edin-mls-26-autumn
if [[ ! -e rag/.env ]]; then
  (
    umask 077
    read -rsp 'ELM API key: ' icf_elm_key
    printf '\n'
    printf 'ELM_API_KEY=%s\n' "$icf_elm_key" > rag/.env
    unset icf_elm_key
  )
fi
chmod 600 rag/.env
git check-ignore rag/.env
```

后续不 `cat .env`。如本地存在个人认证探针，可先上传再用 ASR Python 运行 `rag/scripts/icf_probe_elm.py`，仅检查模型列表，不生成答案、不打印 key。baseline 的 ELM 模型是 `utter-project/EuroLLM-22B-Instruct-2512`；远程生成不等于在本地 GPU 上运行这个大模型。

## 6. 最小 GPU 检测作业：先证明拿到了卡并能计算

下面在 **ICF 登录节点**创建检测脚本，实际计算只发生在提交后的计算节点。使用现有 CUDA PyTorch 环境；若换项目，修改 `GPU_CHECK_PYTHON`，不硬套 ASR 依赖。

```bash
cat > gpu-check.sbatch <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --job-name=gpu-check
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=gpu-check-%j.out
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through sbatch}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
if [[ -r /home/htang2/toolchain-20251006/toolchain.rc ]]; then
  source /home/htang2/toolchain-20251006/toolchain.rc
fi
hostname
date -Is
nvidia-smi -L
gpu_python="${GPU_CHECK_PYTHON:-/home/s2564099/mls-asr-env/bin/python}"
test -x "$gpu_python"
"$gpu_python" - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA unavailable in this allocation'
p = torch.cuda.get_device_properties(0)
print('Torch:', torch.__version__, 'Torch CUDA:', torch.version.cuda)
print('GPU:', p.name, 'Visible GiB:', p.total_memory / 1024**3)
print('Compute capability:', torch.cuda.get_device_capability(0))
x = torch.ones(3, device='cuda')
y = x + x
torch.cuda.synchronize()
assert torch.equal(y, torch.full_like(y, 2))
print('GPU calculation:', y)
PY
echo GPU_CHECK_COMPLETE
SBATCH
bash -n gpu-check.sbatch
sbatch -p Teaching --gres=gpu:nvidia_rtx_a6000:1 gpu-check.sbatch
```

记下新返回的 job ID，按第 8 节查看状态与输出。`4G` 是这个检测脚本的主存设置，不代表 ASR、vLLM 或其他实际任务只需要 4 GB 主存。检测通过也不等于模型推理或服务启动通过。

## 7. 跑课程项目，以及换 GPU 后重新编译

### 7.1 提交已验证的 baseline 脚本

先查看脚本内容和已有作业，确认没有同一实验在运行。在 **ICF 登录节点**：

```bash
cd /home/s2564099/edin-mls-26-autumn
mkdir -p rag/results
bash -n rag/scripts/icf_baseline.sbatch
sbatch rag/scripts/icf_baseline.sbatch
```

默认 Teaching / RTX A6000 / 2 CPU / 24 GB 主存 / 2 小时。脚本在开始运行前由 Slurm 打开输出文件，所以 `rag/results` 必须在提交前创建；提交目录也必须正确。

换 H200 时，用命令行覆盖该脚本的 GRES，前提是当前 Teaching 确有这个资源：

```bash
sbatch --gres=gpu:h200:1 rag/scripts/icf_baseline.sbatch
```

这是另一种提交选择，不应在前一个作业还运行时盲目再提交。`#SBATCH` 中的 `$HOME` 等 shell 变量不会像 Bash 代码那样展开，路径用相对提交目录或明确的绝对路径，作业编号使用 `%j`。

### 7.2 作业内的关键设置

以下命令属于 **GPU 作业脚本内部**，不要直接在登录节点执行模型或测试：

```bash
set -euo pipefail
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
test -r /home/htang2/toolchain-20251006/toolchain.rc
source /home/htang2/toolchain-20251006/toolchain.rc
export RAG_ASR_PYTHON=/home/s2564099/mls-asr-env/bin/python
export PATH="/home/s2564099/mls-rag-env/bin:$PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd /home/s2564099/edin-mls-26-autumn
nvcc --version
sm=$("$RAG_ASR_PYTHON" -c 'import torch; a,b=torch.cuda.get_device_capability(0); print(f"{a}{b}")')
make -C hw1-asr SM="$sm" test
ASR_CUDA=all python -u rag/scripts/run_audio_query.py --device cpu
```

`LD_LIBRARY_PATH` 先初始化，是为了兼容 `set -u` 与工具链脚本。工具链路径失效时应查询 MOTD、`module avail` 和课程要求；不能盲目加载最新 CUDA 或把最新版共享 PyTorch 链入这个已验证的旧环境。

本项目的 Makefile 按 `build/sm_XX` 分开编译：2080 Ti 为 sm_75，A6000 为 sm_86，H200 为 sm_90。上述命令从本次分配到的 CUDA 设备获取架构，再编译和运行测试。不要把 Windows/WSL 本地编译产物当作 ICF 可用产物，也不要把 A6000 的 `.so` 直接复用到 H200。

`--device cpu` 指 RAG 查询编码器在 CPU；ASR 由 `RAG_ASR_PYTHON` 启动的 worker 在 GPU 运行。`ASR_CUDA=none` 是 PyTorch 参考实现，仍用 GPU；`all` 切换自写 CUDA 算子。上下文拼接本身不需要 GPU。

正式 batch 脚本还会运行 practice 检索评估、生成和 judge。它将结果写入独立的 `rag/results/icf-$SLURM_JOB_ID/`，完成时打印 `ICF_BASELINE_COMPLETE`。后续实验保留 baseline 目录和转录缓存，使用新目录；已缓存转录的纯 RAG 实验不一定需要 GPU，但大批量 CPU 计算仍应使用分配到的计算资源。

### 7.3 交互调试只短时间占用

老师教程建议在 Interactive 分区做交互调试，Teaching 上的长任务用 batch。当前官方页面与旧教程在交互分区示例上有差异，以当前课程要求、MOTD 和权限为准。先查询：

```bash
sinfo -p Interactive -N -O partition:20,nodelist:24,gres:100
```

若该分区当前允许：

```bash
srun -p Interactive --gres=gpu:1 --cpus-per-task=2 --mem=8G --time=00:15:00 --pty bash
```

拿到 shell 后才是在分配的计算节点；用 `hostname`、`nvidia-smi -L` 检查。完成调试后 `exit` 释放。若要在 Teaching 的指定 A6000/H200 上诊断，优先提交短 batch 检测，而不是绕过调度直接 SSH 到节点。`tmux` 和 `nohup` 都不能延长 Slurm 分配时限。

## 8. 跟踪、判断成功、取消、下载

在 **ICF 登录节点**，把 `NEW_JOB_ID` 换成本次返回的数字，不能复制历史成功编号：

```bash
job_id=NEW_JOB_ID
squeue -u s2564099
scontrol show job "$job_id"
sacct -j "$job_id" --format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES,MaxRSS
tail -f "rag/results/icf-baseline-$job_id.log"
```

最小检测脚本的日志是 `gpu-check-JOB_ID.out`，不是 baseline 的路径。`PD` 是排队，`R` 是运行；查看 `Reason` 区分资源等待、优先级和配置/权限问题。不要因为排队就不断重复提交。

确认成功需综合 `COMPLETED`、`ExitCode=0:0`、预期输出及脚本完成标记。作业从 `squeue` 消失可能是完成，也可能失败、超时、内存不足或被取消。`sacct` 可能稍有延迟；`MaxRSS` 是主存，不是 GPU 峰值显存。

取消自己的指定作业：

```bash
scancel "$job_id"
```

SSH 断开一般不会取消已经提交的 `sbatch` 作业；重新登录后查询原 job ID。交互作业不要套用这一假设。

在 **WSL 本地项目目录**下载结果和顶层日志：

```bash
cd /mnt/d/Desktop/uni_of_edinburgh/year4/machine_learning_systems/cw/edin-mls-26-autumn
job_id=NEW_JOB_ID
mkdir -p rag/results
scp -r -o ControlPath=/tmp/mls-icf-session "s2564099@icf:/home/s2564099/edin-mls-26-autumn/rag/results/icf-$job_id" rag/results/
scp -o ControlPath=/tmp/mls-icf-session "s2564099@icf:/home/s2564099/edin-mls-26-autumn/rag/results/icf-baseline-$job_id.log" "rag/results/icf-$job_id/"
ssh -S /tmp/mls-icf-session -o BatchMode=yes s2564099@icf "sacct -j $job_id --parsable2 --format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES,MaxRSS" > "rag/results/icf-$job_id/slurm-accounting.txt"
```

保留代码 commit/未提交 diff、脚本、GPU、环境包清单、模型 revision、参数、输入/缓存、原始输出和 accounting。首次编译、模型加载、远程 API 网络时间与稳态推理分别解释；需要比较性能时同卡、预热、重复测量。不要把一次整作业的总时长当作模型加速比。

## 9. 存储和常见故障

- **连接 socket 失效**：VPN/休眠/WSL 重启后重新运行 `-O check` 与 `hostname`。只在确认没有活主连接后处理残留 socket；不用删除 `/tmp` 的宽泛命令。更简单的恢复办法是用新的明确 socket 路径，并在所有客户端统一替换。
- **AI 执行 WSL 返回 `E_ACCESSDENIED`**：可能是本地沙箱禁止访问 WSL 服务，不能据此判定 WSL/ICF 坏了。按宿主工具要求申请执行权限；仍不可用时让用户在 WSL 手工执行，不在 PowerShell 里反复尝试同样的失败连接。
- **报 CPU-only Torch / CUDA 不可用**：先确认实际处于 Slurm GPU 作业中，再检查使用的是 ASR 解释器还是 RAG CPU 解释器。不要靠给登录节点安装 CUDA 来解决。
- **`nvcc` 不存在或库加载失败**：在作业内检查工具链、`nvcc --version`、环境变量与依赖兼容；不是直接安装系统驱动。`nvidia-smi` 的 CUDA Version、`torch.version.cuda`、`nvcc --version` 含义不同，不要求字符串完全相同。
- **`invalid device function` / `no kernel image`**：核对当前 compute capability、扩展编译架构和 Torch 对新卡的支持；在分配到的 GPU 上重新编译适配的目标。
- **CUDA OOM 与 Slurm OOM**：前者检查显存/模型/批量/上下文，后者检查主存和 `--mem`；换卡或增大主存各自解决不同问题。
- **离线模型缺文件**：回登录节点按所需 revision 补齐共享缓存，再提交；不要在反复失败的 GPU 作业内重复下载。
- **API 认证失败**：检查是否加载正确 `.env`、文件权限和服务状态，只输出脱敏错误；不显示 key，不把凭据写进 `sbatch` 参数或日志。
- **Bash 出现 `\r` / bad interpreter**：Windows 上传脚本使用 UTF-8、LF 换行，并在提交前 `bash -n`。准备和运行脚本的逻辑失败要保留非零退出码，不让最后一条 `echo` 掩盖错误。
- **共享存储慢**：传大 ZIP 再解压比大量小文件传输更合适。当前官方文档称共享 home 使用 Lustre，旧教师教程称 NFS；不依赖旧称谓。计算节点不能直接读取 AFS 文件，先在登录节点搬到可访问的共享目录。
- **scratch**：在分配的节点上先 `df -hT`、检查 `/disk/scratch`，只用自己的子目录。按需把大数据复制到节点本地 scratch，结束前把结果复制回共享 home，再下载到本地；本次 baseline 成功不依赖 scratch。ICF home 不应视为可靠备份。

## 10. 新会话的使用约定

用户说“用 ICF 跑这个”时：先读取本指南，检查 WSL/SSH、当前作业与已安装环境；确认任务需要的 GPU 显存和架构，再选择当前确实存在且允许的 GRES。复用环境与缓存，先小规模检查，再运行用户授权的实验；记录 job ID 并跟进结果。

只要求整理指南不意味着要求提交 GPU 作业。个人指南、登录信息和本地辅助脚本默认不加入课程提交或 Git push；需要分享时另做不含个人信息的版本。更新技能只保存操作证据和必要配置，不保存密钥。新的成功记录优先于旧快照，同时保留日期与验证级别。

## 来源

- 本项目 baseline 实验记录：`rag/experiments/baseline-3713067.md`；原始作业、CUDA 测试、包清单、accounting：`rag/results/icf-3713067/`；个人准备与提交脚本：`rag/scripts/icf_prepare.sh`、`rag/scripts/icf_baseline.sbatch`。这些路径均相对本课程仓库；个人脚本可能只存在于用户本机。
- [ICF Teaching 官方说明](https://computing.help.inf.ed.ac.uk/teaching-cluster)：分区、访问与存储；2026-11-12 核对。
- [ICF 连接快速指南](https://computing.help.inf.ed.ac.uk/connecting-icf-quick-start)：跳板与 Slurm 查询；2026-11-12 核对。
- [Hao Tang 的 ICF 教程](https://homepages.inf.ed.ac.uk/htang2/notes/icf/)：登录、工具链、交互调试与 batch；2026-11-12 核对。旧教程与当前 MOTD/正式说明冲突时，保留来源并核对当前要求。
- [OpenSSH 配置手册](https://man.openbsd.org/ssh_config#ControlPersist)：连接复用和后台空闲时长。
- [Slurm sbatch](https://slurm.schedmd.com/sbatch.html)、[Slurm GRES](https://slurm.schedmd.com/gres.html)：提交、资源类型、路径与环境变量规则。
