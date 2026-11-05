# 🚀 Explore Colab

本项目致力于探索从**本地开发环境**高效联动 **Google Colab 云端算力**（免费/付费 GPU/TPU）的两种主流工作流：

1. **GitHub 双向联动（免克隆直达 + 反向提交）**：适合可视化交互式开发、教学演示与团队共享；
2. **Google Colab CLI（终端直连 + 智能体集成）**：适合自动化脚本、批处理训练任务与 AI Agent 无头（Headless）执行。

---

## 📊 两种核心联动方式对比

| 维度 | 方式一：GitHub 双向联动 (Web-first) | 方式二：Google Colab CLI (Terminal-first) |
| :--- | :--- | :--- |
| **核心机制** | 基于 GitHub 规范 URL 免克隆直接打开 Notebook；Colab 网页端编辑后反向 commit | 谷歌官方 CLI 工具 (`colab-cli`)，在本地终端通过 API/SSH 直接调用 Colab 运行时 |
| **交互界面** | 浏览器 Jupyter 交互界面（含图表、Widget、富文本渲染） | 本地终端 (CLI)、REPL 或远程脚本运行 |
| **硬件加速** | 右上角 Connect 下拉菜单 → Change runtime type，切换 GPU/TPU (T4, L4, A100, TPU) | 终端命令行直接指定参数（如 `colab new --gpu T4`） |
| **适用场景** | 数据可视化探索 (EDA)、交互式调试、团队 Notebook 评审与展示 | 自动化测试、长时间模型训练/微调、AI 编码智能体自动调度云算力 |
| **环境依赖** | 本地仅需 Git 客户端与浏览器 | 需要 Python / `uv` 安装 `google-colab-cli` 并完成 Google 账号鉴权 |

---

## 📌 方式一：GitHub 双向联动（极速实操）

### 1. 浏览器直达原理与 URL 格式

Google Colab 原生支持通过特定 URL 结构直接加载任何公开 GitHub 仓库中的 `.ipynb` 文件（私有仓库首次访问时授权 GitHub App 权限即可）：

```text
https://colab.research.google.com/github/<用户名>/<仓库名>/blob/<分支名>/<Notebook路径>.ipynb
```

例如本项目在 GitHub 上的直接打开链接：
```text
https://colab.research.google.com/github/yuler/explore-colab/blob/main/notebooks/01_colab_quickstart_benchmark.ipynb
```

### 2. 为项目添加「Open in Colab」徽章

在任何 Markdown 文档或 Notebook 第一行中，添加如下徽章代码：

```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yuler/explore-colab/blob/main/notebooks/01_colab_quickstart_benchmark.ipynb)
```

渲染效果：
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yuler/explore-colab/blob/main/notebooks/01_colab_quickstart_benchmark.ipynb)

### 3. 双向同步完整闭环流程 (Local ⇄ Cloud)

```mermaid
flowchart LR
    A["本地代码编辑<br/>(VS Code / Cursor)"] -->|"git push"| B["GitHub 远程仓库<br/>(yuler/explore-colab)"]
    B -->|"浏览器直达 URL<br/>/ GitHub 徽章"| C["Google Colab<br/>(连接 GPU/TPU 运算)"]
    C -->|"文件 -> 在 GitHub 中保存副本<br/>(自动 Commit)"| B
    B -->|"git pull"| A
```

#### 闭环操作步骤：
1. **本地编写与提交**：在本地编写好交互式 Notebook，通过 `git push origin main` 推送至 GitHub；
2. **云端直达运行**：浏览器点击对应直达链接或徽章，直接在 Colab 中挂载云端运行时，执行硬件加速任务；
3. **云端反向提交**：在 Colab 页面修改代码或运行产生新的输出后，点击菜单：
   > **「文件 (File)」** ➔ **「在 GitHub 中保存副本 (Save a copy in GitHub)」**
   > 选择目标仓库与分支，输入 Commit Message 即可直接提交到 GitHub；
4. **拉取回本地**：在本地执行 `git pull`，即可同步云端运行产生的修改与执行状态。

---

## 💻 方式二：Google Colab CLI（终端直连与自动化）

Google 官方推出的 **Google Colab CLI** (`google-colab-cli`，命令为 `colab`) 允许开发者与 AI Agent 无需打开浏览器，直接在终端中申请云端算力并执行脚本。

### 1. 安装与鉴权

```bash
# 使用 uv 安装 (推荐)
uv tool install google-colab-cli

# 或使用 pip
pip install google-colab-cli
```

**鉴权模式支持**：
- **OAuth 浏览器登录**：首次执行 `colab repl` 或 `colab new` 会自动唤起浏览器授权（凭证缓存在 `~/.config/colab-cli/`）；
- **ADC (Application Default Credentials)**（更适合 Agent / 无头服务器）：
  ```bash
  gcloud auth application-default login \
    --scopes=openid,\
  https://www.googleapis.com/auth/cloud-platform,\
  https://www.googleapis.com/auth/userinfo.email,\
  https://www.googleapis.com/auth/colaboratory
  ```
- 验证登录状态：
  ```bash
  colab sessions
  ```

### 2. 核心操作与实战命令

* **申请并创建云端运行时 (`colab new`)**：
  ```bash
  # 创建 CPU 会话
  colab new -s my-cpu

  # 创建指定 GPU 运行时 (支持 T4, L4, A100, H100 等)
  colab new -s my-gpu --gpu T4

  # 创建 TPU 运行时 (支持 v5e1, v6e1)
  colab new -s my-tpu --tpu v5e1
  ```

* **执行本地脚本 / 代码片段 (`colab exec`)**：
  ```bash
  # 远程执行单行 Python 代码（代码从标准输入读入，不能写成位置参数）
  echo "import torch; print('CUDA 可用:', torch.cuda.is_available())" | colab exec -s my-gpu

  # 远程执行本地脚本文件（内核状态跨命令持久保持）
  colab exec -s my-gpu -f scripts/colab_interactive_demo.py
  ```

* **独立临时运行（执行即释放，支持 Shebang！）(`colab run`)**：
  ```bash
  # 申请独立机器执行脚本，执行完成后自动释放销毁虚机，避免产生额外计费
  colab run --gpu T4 scripts/colab_interactive_demo.py
  ```
  > 💡 **黑科技：脚本 Shebang 声明**：  
  > 可以在 Python 脚本顶部加上：  
  > `#!/usr/bin/env -S colab run --gpu T4`  
  > 赋予执行权限后直接 `./script.py`，即可一键将该脚本分发至云端 GPU 运行！

* **文件上传与下载 (`colab upload` / `colab download`)**：
  ```bash
  # 上传本地数据到远程虚机 /content
  colab upload -s my-gpu ./data.csv /content/data.csv

  # 从远程虚机下载模型权重或产物到本地
  colab download -s my-gpu /content/model.pt ./models/model.pt
  ```

* **导出运行历史为 Jupyter Notebook (`colab log`)**：
  ```bash
  colab log -s my-gpu --output run_session.ipynb
  ```

* **释放与停止会话 (`colab stop`)**：
  ```bash
  colab stop -s my-gpu
  ```

---

## 🔌 进阶：REST API 与 MCP 集成

除 CLI 外，Google Colab 还提供两种更底层的编程接入方式，适合构建自定义工具链或将 Colab 集成到 AI Agent 工作流中：

### REST API

[Google Colab REST API](https://developers.google.com/colab/api/reference/rest) 允许通过 HTTP 直接管理 Colab 运行时（Runtime）的完整生命周期：创建、查询、执行代码、停止等。适合需要在自有后端或平台中以编程方式调度云端算力的场景。

### MCP（Model Context Protocol）

[Google Colab MCP](https://developers.google.com/colab/api/reference/mcp) 是 Google 官方提供的 MCP Server，遵循 Anthropic 的 Model Context Protocol 标准。AI 编码智能体（如 Claude、Cursor Agent、pi 等）可通过 MCP 直接调用 Colab 的运行时能力，实现：

- 让 Agent 自主申请云端 GPU/TPU 资源；
- 在 Agent 对话中直接执行代码并取回结果；
- 与本地开发环境无缝联动，无需人工切换浏览器。

> 💡 这正是让本地 AI Agent 「按需调用云端算力」的关键基础设施。

---

## 📂 仓库内容与示例体验

```text
.
├── README.md                               # 本项目说明指南
├── requirements.txt                        # 本地/云端依赖清单
├── notebooks/                              # 交互式 Notebook 目录（方式一）
│   ├── 01_colab_quickstart_benchmark.ipynb # 硬件检测、PyTorch CUDA 基准测试、Drive 挂载
│   └── 02_interactive_eda_and_plot.ipynb   # 交互式控件 (ipywidgets)、Colab 数据表、动态绘图
└── scripts/                                # 独立执行脚本目录（方式二）
    └── colab_interactive_demo.py           # 既支持 VS Code 交互式运行 (# %%)，也支持 colab exec
```

### 快速直达链接：
1. **[01_colab_quickstart_benchmark.ipynb](notebooks/01_colab_quickstart_benchmark.ipynb)**  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yuler/explore-colab/blob/main/notebooks/01_colab_quickstart_benchmark.ipynb)  
   *内容：CPU 核心/显存检测、CUDA 大矩阵乘法算力压测（对比 CPU vs GPU 耗时）、Google Drive 挂载模板。*

2. **[02_interactive_eda_and_plot.ipynb](notebooks/02_interactive_eda_and_plot.ipynb)**  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yuler/explore-colab/blob/main/notebooks/02_interactive_eda_and_plot.ipynb)  
   *内容：加载 Colab 数据表格增强插件、交互式滑动条与下拉菜单联动绘图、特征相关性热力图。*

---

## 🛠️ 下一步：同步测试建议

要测试 **方式一（GitHub 双向联动）**：
1. 在 GitHub 上创建名为 `explore-colab` 的公开仓库（以 `yuler` 账号）；
2. 执行以下命令推送当前代码：
   ```bash
   git branch -M main
   git remote add origin https://github.com/yuler/explore-colab.git
   git add .
   git commit -m "feat: initial colab exploration notebooks and readme"
   git push -u origin main
   ```
3. 打开浏览器访问：
   `https://colab.research.google.com/github/yuler/explore-colab/blob/main/notebooks/01_colab_quickstart_benchmark.ipynb`
4. 在页面中运行单元格，修改任意注释或代码，点击「文件」->「在 GitHub 中保存副本」；
5. 在本地执行 `git pull`，验证双向同步！
