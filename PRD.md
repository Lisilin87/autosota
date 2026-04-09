# AutoSOTA 复现项目 - 产品需求文档 (PRD)

## 1. 项目概述

### 1.1 项目名称
AutoSOTA - 端到端自动研究系统复现

### 1.2 项目目标
实现一个 AutoSOTA-like 端到端自动研究系统，能够：
- 输入：论文 PDF/HTML 和可选的代码仓库 URL
- 输出：可执行的 baseline、优化后的 best patch、scores.jsonl、code_analysis.md、idea_library.md、research_report.md 和 final_report.md

### 1.3 核心价值
自动化论文方法复现与优化流程，通过多 agent 协作、长周期执行、外部记忆和调度机制，避免长任务崩溃与失控。

### 1.4 技术栈
- 语言：Python 3.10+
- 数据验证：Pydantic v2
- 容器化：Docker
- 配置管理：YAML
- 版本控制：Git
- LLM：OpenAI API / Anthropic API

---

## 2. 系统架构

### 2.1 架构分层

```
[User / CLI / API]
      ↓
[Scheduler Service] ← → [Supervisor Service]
      ↓                      ↓
[Task Orchestrator] ← → [Monitor Service]
      ↓
[Resource] [Objective] [Init] [Fix] [Ideator] [Executor]
      ↓
[Workspace + Docker + Logs + Memory Files]
```

### 2.2 核心模块

| 模块 | 职责 |
|------|------|
| resource_service | 论文-仓库-依赖-数据-权重对齐 |
| objective_service | 构建 rubric、目标指标、baseline target |
| init_service | 初始化环境、发现命令、准备执行 |
| monitor_service | 流式监控、死锁检测、干预建议 |
| fix_service | 错误签名归一化、修复策略选择 |
| idea_service | idea library 生成与更新 |
| scheduler_service | 阶段调度、任务生命周期管理 |
| supervisor_service | 红线约束、合法性审计 |

---

## 3. 功能需求

### 3.1 Phase A：最小闭环版 (MVP)

#### 3.1.1 资源准备 (Resource Service)
- 从论文中提取 repo 链接
- 多 repo 候选排序与选择
- Shallow clone 仓库
- 抽取 README 和文件树
- Repo readiness 判断
- 解析外部依赖（dataset、base model、checkpoints）
- 生成下载计划并执行

#### 3.1.2 目标设定 (Objective Service)
- 识别论文主要指标
- 确定优化方向（higher/lower is better）
- 生成树状 rubric（BFS 递归展开）
- 为每个节点附 evidence
- 输出 pass/fail 条件

#### 3.1.3 环境初始化 (Init Service)
- 确定仓库 commit
- 检测 Dockerfile / conda env / requirements
- 生成 Docker 环境
- 校验 GPU / CUDA / Python / Torch
- 安装依赖
- 修补路径与配置
- 自动发现训练与评测命令
- 试跑 dry-run

#### 3.1.4 Baseline 运行
- 初始化 git
- Commit baseline
- 运行评测
- 记录到 scores.jsonl

#### 3.1.5 代码分析
- 扫描仓库结构
- 识别入口文件、训练脚本、评测脚本
- 生成 code_analysis.md

#### 3.1.6 Idea 生成
- 生成至少 10 条 idea
- 分类为 PARAM/CODE/ALGO
- 红线审计
- 输出 idea_library.md

#### 3.1.7 迭代优化
- 选 idea
- Git snapshot
- Patch 代码
- 执行评测
- 记录结果
- 更新 idea library

### 3.2 Phase B：增强稳定性

#### 3.2.1 监控服务
- 流式日志解析
- 死锁/停滞检测
- 干预建议（continue/resume/fallback/terminate/rollback）

#### 3.2.2 修复服务
- 错误签名归一化
- 修复策略选择
- 避免重复修复

#### 3.2.3 状态持久化
- Scheduler 持久化
- 崩溃恢复
- 资源回收

### 3.3 Phase C：批量跑论文

- 多任务队列
- 容器复用
- 统一结果面板

---

## 4. 数据模型

### 4.1 PaperTask
```python
class PaperTask(BaseModel):
    paper_id: str
    title: str
    paper_path: str
    repo_url: str | None = None
    conference: str | None = None
    domain: str | None = None
    target_metric: str | None = None
    target_direction: Literal["max", "min"] | None = None
    baseline_metric: float | None = None
    status: str
```

### 4.2 ResourceManifest
```python
class ResourceManifest(BaseModel):
    repo_candidates: list[str]
    selected_repo: str | None
    dataset_items: list[dict]
    model_items: list[dict]
    checkpoint_items: list[dict]
    readiness_signals: dict
    local_paths: dict
```

### 4.3 ObjectiveRubricNode
```python
class ObjectiveRubricNode(BaseModel):
    node_id: str
    name: str
    description: str
    depth: int
    weight: float
    pass_fail: bool | None = None
    evidence: list[str] = []
    children: list["ObjectiveRubricNode"] = []
```

### 4.4 IdeaItem
```python
class IdeaItem(BaseModel):
    idea_id: str
    title: str
    idea_type: Literal["PARAM", "CODE", "ALGO"]
    granularity: Literal["micro", "meso", "macro"]
    priority: int
    risk: Literal["low", "medium", "high"]
    rationale: str
    assumptions: list[str]
    status: Literal["PENDING", "CLEARED", "REJECTED", "DONE"]
    redline_audit: dict
    history: list[dict] = []
```

### 4.5 RunRecord
```python
class RunRecord(BaseModel):
    iteration: int
    git_commit: str
    idea_id: str | None
    command: str
    metrics: dict
    success: bool
    log_path: str
    start_time: str
    end_time: str
```

---

## 5. 红线约束 (Red Lines)

### R1: 评测参数不变
评测参数（如 recall@k 中的 k、history window、context window）不得修改。

### R2: 评测脚本不变
评测脚本、score aggregation、metric computation 不得修改。

### R3: 预测输出真实
预测输出必须来自真实模型推理，不得 hard-code。

### R4: 指标平衡
主指标提升不能以其他关键指标显著下降为代价。

### R5: 禁止数据泄露
不得将测试集信息引入训练。

### R6: 数据集不变
不得修改数据集分布（不能过滤、重采样、重标注）。

### R7: 论文特定约束
论文特定约束必须在 code_analysis.md 中显式列出。

---

## 6. 工作流

### 6.1 单论文主流程
```
Input paper
  → AgentResource
  → AgentObjective
  → AgentInit
  → Phase 0: baseline run
  → Phase 1: code analysis
  → Phase 2: idea library + redline audit
  → Phase 3: iterative optimization
  → best result export
```

### 6.2 状态流转
```
NEW
 → RESOURCE_READY
 → OBJECTIVE_READY
 → INIT_READY
 → BASELINE_DONE
 → CODE_ANALYZED
 → IDEA_LIBRARY_READY
 → OPTIMIZING
 → FINISHED / FAILED / NEED_REVIEW
```

---

## 7. 输出产物

### 7.1 必须产物
- `resource_manifest.json`
- `rubric.json`
- `target_metric.json`
- `init_report.json`
- `code_analysis.md`
- `idea_library.md`
- `research_report.md`
- `scores.jsonl`
- `state.json`
- `best_patch.diff`
- `final_report.md`

### 7.2 scores.jsonl 格式
```jsonl
{"iteration":0,"idea_id":null,"metrics":{"acc":0.823},"success":true}
{"iteration":1,"idea_id":"I003","metrics":{"acc":0.831},"success":true}
{"iteration":2,"idea_id":"I007","metrics":{"acc":0.819},"success":false}
```

---

## 8. 配置参数

### 8.1 默认配置
```yaml
workspace_root: ./workspaces
max_iterations: 8
monitor:
  stuck_timeout_sec: 1200
  repeated_error_threshold: 3
scheduler:
  max_parallel_tasks: 2
docker:
  enabled: true
  base_image: "pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime"
resource:
  shallow_clone: true
  max_download_gb: 50
objective:
  rubric_max_depth: 3
  rubric_weight_threshold: 0.1
supervisor:
  redline_policy: configs/policies/redlines.yaml
evaluation:
  primary_metric_required: true
  secondary_metrics_required: true
```

---

## 9. 实施计划

### Sprint 1：跑通 baseline
- paper task loader
- repo clone
- Docker build
- command discovery
- baseline baseline run
- metrics parser
- scores.jsonl

**验收标准**：能对 1 个 repo 成功跑出 baseline 分数

### Sprint 2：补齐分析与 idea
- code_analysis.md
- research_report.md
- idea_library.md
- redline audit

**验收标准**：至少自动生成 10 条 idea，能对每条 idea 给出 CLEAR/REJECT

### Sprint 3：迭代优化
- phase 3 loop
- git snapshot
- patch apply
- result compare
- best result export

**验收标准**：至少能自动跑 3 轮，能产出 best patch

### Sprint 4：增强稳定性
- monitor
- fix
- scheduler persistence
- recovery

**验收标准**：中途杀进程后可恢复，重复错误不会无限循环

---

## 10. 成功标准

1. 对 1 篇论文成功完成 repo 对齐与资源准备
2. 成功运行 baseline，并把指标写入 scores.jsonl
3. 自动生成 code_analysis.md、idea_library.md、research_report.md
4. 至少运行 3 轮合法优化
5. 最终导出 best patch 和 final report
6. 所有 idea 和 patch 都经过 redline 审计

---

## 11. 风险与注意事项

### 11.1 最大工程风险
把论文世界映射成可执行世界，包括：
- 资源碎片化
- 环境初始化
- 长周期调试
- 开放式创新

### 11.2 容易做歪的地方
- 误改 eval 脚本
- 指标 parser 和真实评测口径不一致
- 为了跑通偷偷缩小数据或改 test split
- 重复尝试同一修复路径
- idea library 沦为"调超参列表"

### 11.3 工程上最重要的三件事
1. 所有状态必须落盘（持久化与恢复）
2. 外部记忆必须成为正式产物
3. 红线规则必须硬编码成系统约束

---

## 12. 非功能性需求

### 12.1 可维护性
- 模块化设计，低耦合高内聚
- 清晰的接口定义
- 完善的日志系统

### 12.2 可扩展性
- 插件化的 agent 架构
- 可配置的 prompt 模板
- 支持多种 LLM 后端

### 12.3 可观测性
- 结构化日志
- 指标采集
- 状态追踪

### 12.4 安全性
- 容器隔离
- 资源限制
- 红线审计

---

## 13. 版本规划

### v0.1.0 (MVP)
- Sprint 1-2 功能
- 单论文支持
- Docker 隔离

### v0.2.0
- Sprint 3 功能
- 迭代优化

### v0.3.0
- Sprint 4 功能
- 监控与修复

### v1.0.0
- 完整功能
- 批量处理
-