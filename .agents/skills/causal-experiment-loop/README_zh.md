# Causal Experiment Loop — 可移植的科研实验 Agent Skill

这是基于 MiniFly / Fable 近期实验与审计方式**提炼**的通用 Agent Skill。它不是 Fable 官方发布的 skill，也不保证自动产生科研突破。

## 文件结构

- `SKILL.md`：代理主指令；严格执行“证据快照 → 定位瓶颈 → 预注册 → 资格检查 → 最小实验 → 敌意审计 → 下一步”。
- `references/minifly-evidence.md`：E16/E17/Fable 5.2 等七个具体案例和源码链接。
- `assets/experiment_contract.example.json`：可程序检查的预注册实验契约示例。
- `assets/evidence_ledger.template.md`：事实、失败与撤回结论的证据账本。
- `assets/result_report.template.md`：包含 seed 级数值、对照是否真正匹配、证据来源的结论模板。
- `scripts/lint_contract.py`：检查契约必要字段、角色/流是否匹配、单变量差异、预注册时间与异常配置。

## 使用

把整个目录放到使用的代理支持的技能目录下（例如 Codex 项目中的 `.agents/skills/causal-experiment-loop/`），不要只复制 `SKILL.md`，否则案例、模板和检查脚本无法按相对路径读取。

使用示例：

> 使用 `causal-experiment-loop` 对 MiniFly 最新 E17 实验进行敌意审计。只列出新增事实和可能推翻结论的对照，给出一个低成本、可证伪的下一项实验，不要直接启动。

> 使用 `causal-experiment-loop` 将一个失败的强化学习实验转成预注册实验契约，检查同数据流的参考组、梯度可达性和更新规则。

> 使用 `causal-experiment-loop` 对照仓库的 prediction、receipt 和 weight，标记哪些结果被实现缺陷 invalidated，保留历史记录。

本技能**不会**自行获得 GitHub、终端或后台任务权限，需由运行它的代理具有相应工具；可能昂贵的训练与向仓库推送需明确授权。建议首先让它完成只读审计，再授权最小实验。

## 检查实验契约

```bash
python3 scripts/lint_contract.py assets/experiment_contract.example.json
```

示例文件带有占位 SHA，命令会给出警告；用于正式实验时必须替换为实际 hash，并锁定预注册版本。

源资料：`references/minifly-evidence.md`。此包适配通用的 Agent Skills `SKILL.md` 格式，但不同宿主的安装目录和支持能力可能不同。