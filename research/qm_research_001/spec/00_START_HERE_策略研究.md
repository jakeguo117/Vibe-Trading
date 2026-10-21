# 策略研究｜START HERE
更新：2026-10-21（Asia/Hong_Kong）｜入口版本：v1.7
状态：MOM-XASSET-001 已按统一口径完整重跑（v1.2）并归档；研究结论与交付完整性见新 run。旧正式结果与旧归因包保留审计，旧归因数字不再作为最终解释。

## 1. 固定入口与资料分工
- 策略研究总入口：https://drive.google.com/drive/folders/1WvLQ5vOLcaFC3LIFN6SHdx1GCpXSTRVZ
- Notion 项目控制台：https://app.notion.com/p/3f5474970e4081578c89e7de18c0760f
- 已有正式回测结果正本：https://drive.google.com/drive/folders/1kRID4peUueFv10wFC2wrkrICL5hw40eI
- 已有研究方法与设计：https://drive.google.com/drive/folders/1zinFx0L6kyZq9TLp5yDSS_SHVb9fdcoB
- 研究代码仓库：https://github.com/jakeguo117/Vibe-Trading
- 正式系统仓库：https://github.com/jakeguo117/trading-system

Drive 存研究定义、报告、结果和证据；Notion 存任务负责人、状态、下一步和 Drive 链接；GitHub 存代码及版本。研究通过且 Jake 拍板后才可进入正式系统。
本入口覆盖整个策略研究，非仅 Momentum。已有资料保留原文件 ID 和位置，通过链接统一访问；新研究方案与交接资料放本入口的策略子目录，正式回测输出继续遵守已有“回测结果”归档约定。
本文件是持续更新的入口，维护时更新原文件，避免反复创建同名“最新版”。下载副本只是当时快照，后续以 Drive 原文件为准。

## 2. 接手顺序
1. 读取本文件，确认更新时间和研究范围。
2. 读取对应研究的最新摘要、result、acceptance、corrections；需复验时读取配置、数据指纹、代码版本与逐笔结果。
3. 查 Notion 当前任务与负责人；先回复：当前版本、已完成、已决定、未决问题、准备做的唯一下一步。
4. 有冲突以具体 run 的原始结果为数值依据，以 Jake 的明确决策为批准依据。更新晚不自动等于权威。
5. 没权限或文件缺失时列出缺口；不得声称已读取，也不得仅凭聊天记忆重建结论。

## 3. 当前研究清单
| 研究 | 当前证据与状态 | 下一步 |
|---|---|---|
| MOM-GLD-001 | R1/R2/R3 已完成；保留为历史证据 | 不继续原版单纯提仓位修补，新研究见下一行 |
| MOM-XASSET-001 | 首轮NO_PASS；Jake已授权统一口径完整重跑，状态：已设计待测试 | Hermes按v1.2规范执行原180组开发、三资产验证及归因，保存新run |
| Fib / 周期 / D-H | 根据既有项目记录，Jake 已决定暂停 D/H，不做下一轮 | 保留证据；不自动重启 |
| GTO 情景瘦表 | 既有接手盘点记录为暂停；本次未独立复验数值 | 如重启先查原报告与 Jake 决策 |
| 单标的 / 金银油均线 / 早期因子等 | 历史材料已存在，尚未在本轮逐项复核 | 从历史结果与研究仓库定位，禁止把历史探索当生产批准 |
| Mean Reversion 等替代方向 | 尚未在本次检索中找到已冻结且验收的正式方案 | 待讨论；不能从 Momentum 失败直接推导其有效 |

### Momentum 正本
- 设计：https://drive.google.com/file/d/10CH3KJlLhw7o5JBHephyVA7DWdUGCcNO/view
- R1 结果及后续 revisions：https://drive.google.com/drive/folders/1mxAparZnsdjTYI-RPIy5k3pa8u5_Qzrg
- R2 run：https://drive.google.com/drive/folders/1k5N2_g41VrPxuGcvEY8Bneruvq5yHAkX
- R2 摘要：https://drive.google.com/file/d/17bv55F8Oi8lNdTnJiPn_Qd2WbfQ8veIW/view
- R3 run：https://drive.google.com/drive/folders/1rkUTEsz639b0n3YcdhTfpbggjy7w90GZ
- R3 摘要：https://drive.google.com/file/d/19VNAS5CM-r1tSdTViLRQI2dJhDzWdPEQ/view
- R3 result：https://drive.google.com/file/d/1kD729hCstoUra8jUf5DTG7V0AnrwTO3v/view
- 既有综合接手盘点（含推论，须与原始结果交叉核对）：https://docs.google.com/document/d/1dPyUNHCYEl_lAvQlAMH5LniXBMu8fvKYJJ6nkVB90uU/edit

### Momentum 已核对的报告数值
2016-01-04 至 2026-10-19，初始 16,000 美元，5bps，现金利息为零；以下为报告读回，未在本会话重新运行代码。

| 路径 | 年化收益 | 最大回撤 | Sharpe |
|---|---:|---:|---:|
| A 原版风险仓位 | 2.71% | -6.11% | 0.79 |
| A-50 入场半仓 | 2.75% | -6.23% | 0.64 |
| A-100 入场满仓 | 5.49% | -11.82% | 0.66 |
| BH-100 买入持有 | 12.83% | -26.40% | 0.82 |

- R3：A 持仓期平均名义仓位已为 43.60%；在场 703 / 2714 个交易日。满仓仍沿用原进出场时点，不是全程满仓。
- R2：取消移动止损提高收益，也提高波动与回撤，Sharpe 降到 0.66；直接允许重入使收益、回撤、Sharpe 和成本均恶化。
- R2：简单低权重 GLD/现金月度组合 Sharpe 约 0.79，与 A 相近，但实际风险匹配不充分，不能宣布严格同风险胜负。
- 可支持的判断：目前具体规则缺乏足够择时优势证据；单纯提仓位不能解决问题。
- 不可支持的判断：Momentum 整个策略家族被证伪、GLD 必然适合均值回归、移除止损必然更优。

### 已发现的口径问题
1. R2 D1 的 75.2% 空仓归因使用更长诊断范围，不要与 2016 起净值主表混用。2016 起在场比例可由 703/2714 单独计算。
2. R3 摘要“现金收益 0”段把 A-100 写成“长期满仓、现金少”，与其全程仓位 25.78% 不一致；BH-50 平均仓位 62.03%，也不等于“长期六成现金”。现金影响应按每日实际现金余额与当期可实现利率重新核算，不能沿用这句文字判断。
3. 止损后继续上涨是事后路径描述；缺少无条件上涨概率及匹配对照，不能独立证明止损错误或均值回归优势。
4. 历史段已被多次观察。新改动在同段的漂亮结果属于探索性结果，不能重新标为干净样本外。
5. 旧总索引与 Notion 历史文字主要写到 R1。本入口已列出 R2/R3；历史记录不删除，接手须读取 revisions。

## 4. MOM-XASSET-001 当前任务与首轮历史
### 当前任务：统一口径完整重跑（已完成）
- Jake 决定：“直接按照最新的标准重新做一次回测”。任务 ID：MOM-XASSET-001-RERUN-01。
- 权威文件：[重跑规范 v1.2](https://drive.google.com/file/d/1FQdvOwdHRuWygzj6I8RTfFYqITjabaRf/view) +
  [原计划 v1.0](https://drive.google.com/file/d/1pkuVqUplWBc5ruOiec1Xy4lWzT4VSwga/view)，冲突以 v1.2 为准。
- 本轮 run：`mom-xasset-001-rerun-20261021-2313`｜结果正本：https://drive.google.com/drive/folders/1kRID4peUueFv10wFC2wrkrICL5hw40eI → `2026-10-21_mom-xasset-001-rerun/mom-xasset-001-rerun-20261021-2313/`
- 一句话结论：见 `00_summary_zh.md`；逐条门槛：`acceptance.md`；新旧差异：`correction_impact.csv`。
- 本轮修正：固定权重基准首日建仓、首日成本/收益纳入指标、Sharpe 与 Sharpe0 分离、
  日期精确对齐（不一致即报错）、连续切片与独立分段分表、隔夜/日内归属、含现金日的捕获率、期末清算唯一性。
- 可靠性：`test_evidence.md`（旧实现失败/新实现通过对照）；自动接续：`continuation_check.md`。
- 旧归因包已被本轮 `attribution.md` 取代（仅作审计保留）：https://drive.google.com/drive/folders/1wOs3TSC2fVEdtuoA5DgRljfE7_MdxN5C

### 首轮历史（2026-10-21 完成；数字已被本轮 rerun 取代，仅作差异审计）
- 研究文件夹：https://drive.google.com/drive/folders/1tBa54vyB67CUxOehI4RYcDBjKGSSUaSY
- 完整计划 v1.0：https://drive.google.com/file/d/1pkuVqUplWBc5ruOiec1Xy4lWzT4VSwga/view
- 用户已选方向：MacBook 本地 Hermes 自动搜索入场、退出、仓位参数；资产不只 GLD，可在 GLD 开发后到 USO 等检验。
- 状态：**已执行，NO_PASS**。主候选 `A-N40_XA-M10_F100`（40 日突破 / 10 日通道低点 / 100% 净值）。
- 结果正本：https://drive.google.com/drive/folders/1BeWl-WTiqpT_WO4m0aEqei1pLJlPIU-R
- 一句话：GLD 与 SPY 的收益和 Sharpe 明显低于买入持有（GLD CAGR 3.89% vs 12.77%），USO 相对 BH100 三项全改善但绝对回撤 31.80% 超过 30% 上限；三段一致性、20bps、延迟、扰动多数不通过；DSR 0.47。
- 归档交付：已完成（rclone check 0 差异，5 个关键文件 SHA256 读回一致）。研究结论与归档状态分开记录。
- 修复记录：BH50-M/VT10 卖出现金流 bug 影响两个竞争基线，不影响主引擎/开发评分/候选锁定；见结果目录 `fix_impact.md` 与 `superseded/`。
- **旧归因包：待纠错，不能作为最终解释**：https://drive.google.com/drive/folders/1wOs3TSC2fVEdtuoA5DgRljfE7_MdxN5C
  - ChatGPT读取报告及代码并最小复现，确认固定权重sleeve首月未建仓；持仓条件捕获率不能证明择时无效；归因Sharpe与正式口径混用。分段及日期对齐亦需按v1.1统一。
  - 原“同暴露择时−0.41/−2.87/+2.95pp”等数值待重算；“整族无效”“仅H1/H2值得做”超出证据范围，当前不采纳。
  - 旧归因文件保留供审计；完整重跑后以新版数值、差异说明和证据限制为准。旧正式NO_PASS仍为历史结论，不能预先当作新run结果。
- 首轮：GLD 2007–2015 开发，锁定最多3个候选且主候选唯一；2016–2026-10-20 GLD/USO/SPY 验证。各资产独立账户，不是轮动或多资产组合。
- 范围：180个主配置，最多36个预先冻结的局部扰动；最多3000次路径评估、正式计算累计8小时；无新增数据采购、云算力或自动下单。
- 目标：年化收益更高、最大回撤更小、Sharpe更高；门槛/排序/成本/现金/复权/失败停止规则详见计划。没有合格候选也是有效交付。
- 数据（本轮已核实）：GLD/USO/SPY 调整日线均由免费源重下至 2026-10-20（GLD 5514 行、USO 5165 行、SPY 5527 行），DGS3MO 至 2026-10-19；与既有本机快照重叠段一致；USO 1:8 反向拆股未产生假收益；无缺失交易日；零采购。详见结果目录 `data_sources.md` / `data_audit.json`。
- 数据核对报告：https://drive.google.com/file/d/1bTzN2cKWt0pdsGJ5T5LTQ8hcT6LpXEkz/view
- 历史已经多轮观察，本轮只能称算法隔离的历史迁移验证；未来模拟才提供新行情证据。
- 本轮 run_id：`mom-xasset-001-20261021-1719`；引擎评估 1024 次（开发 720 / 验证 304），墙钟约 1 分钟级；未修改生产系统、未下单、未新增采购。状态由 Hermes 写入并在归档后读回。

## 5. 多 agent 接续约定
- 触发：Jake 指定任务，或负责人明确交接。建立资料入口不等于自动启动任务。
- 输入：本文件 + 策略定义 + 最新已验收 run + 当前任务。
- 每项任务一个负责人；多人并行时各写独立产物，不同时改同一个状态文档。角色按当前任务认领，不绑定某模型名称。
- 新 run 使用唯一 strategy_id / run_id；完成后先保存并读回结果，再更新本入口的最新 run 链接，最后同步 Notion 状态。
- 状态只能用：待讨论、已设计待测试、测试中、待审查、暂停、待前瞻验证、已批准。必须区分 agent 建议与 Jake 决策。
- Notion 只保存短状态和正本链接，不再复制一套完整研究报告。
- 写回前读取当前版本；发生并发冲突时保留各自产物，交负责人合并，不覆盖别人的更新。
- 验收：链接能读取；版本/日期明确；配置、代码提交、数据快照、成本和现金口径可追溯；摘要与底表一致。
- 失败：明确写“未完成”与缺失项，保存已完成产物及唯一下一步，不把尝试记成完成。
- 每个 agent 的接入需要分别验证：读取本入口、复述最新 run、在授权测试任务中写回并读回成功。ChatGPT连接读写已验证；Hermes已有Drive归档与读回记录，但其Notion连接器仍缺失；Gemini/Grok需在各自会话核验。文件夹存在不等于自动同步、自动唤醒或自动拥有权限。

## 6. 通用交接提示词
你正在接手 Jake 的整个策略研究项目。固定入口：
https://drive.google.com/drive/folders/1WvLQ5vOLcaFC3LIFN6SHdx1GCpXSTRVZ
先读取 00_START_HERE_策略研究.md，再读取相关策略最新 run 的摘要、结果、验收与更正，核对 Notion 当前任务。
请先用五点回复：最新版本、已完成、已决定、未决问题、你准备做的下一步，并列出已读取的正本链接。
不要将建议当批准，不要重跑已结束的问题，不要修改生产系统。不能访问的资料请明确说明。
完成任务后保存可复现证据，读回验证，更新同一份入口及 Notion 状态。新研究必须在执行前固定假设和停止条件。

## 7. 更新记录
- 2026-10-21：按 Jake 要求建立全策略研究共享入口；核对 MOM R2/R3 报告与 Notion 项目；新增证据限制和通用接续约定。


- 2026-10-21（v1.1）：用户选定跨资产参数搜索与 Hermes 执行；新增 MOM-XASSET-001 v1.0 计划及 USO 数据证据；替换原待定下一步，实际训练待 Hermes 接手。
- 2026-10-21（v1.3）：Hermes 运行时预算 `agent.max_turns` 60→200、`budget_warning_ratio` 0.75，断点写入 CHECKPOINT.md；研究结论不变。
- 2026-10-21（v1.4）：完成 MOM-XASSET-001 失败归因（只读，未新搜索）：差距主因是暴露不足，同等暴露下择时中性偏负；USO 优势集中 2020–2022、最大回撤实为 2022–2023；该规则族不建议继续。
- 2026-10-21（v1.2）：Hermes 执行 MOM-XASSET-001 首轮并交付：180 组开发搜索、锁定 1 主 2 敏感候选、2016–2026-10-20 三资产验证、四场景 + 20 组扰动、bootstrap/DSR；结论 NO_PASS；修复并披露 BH50-M/VT10 基准现金流 bug；结果归档 Drive 并读回。

- 2026-10-21（v1.5）：Jake授权完整重跑；新增统一口径v1.1规范，原180组及三资产全流程复算。旧归因关键数字标待修正，停止把“规则族无效”当作已证结论；Hermes待启动，未伪报运行。


- 2026-10-21（v1.6）：按Jake“确保跑完”新增重跑规范v1.2第8.1节：独立运行、有限重试、自动接续、完成与阻塞判据及临时守护清理；当前仅规范已交付，待Mac部署验收。
- 2026-10-21（v1.7）：MOM-XASSET-001 按 v1.2 完成统一口径重跑 （run `mom-xasset-001-rerun-20261021-2313`）；修正首日/基准/日期对齐/指标口径；新结果归档并读回。
