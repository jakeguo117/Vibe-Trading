# GPT v1.1 开跑前核对：回踩分钟线版（PB-AV）vs 日线版（PB）· 2023+

**日期**：2026-10-25（Asia/Hong_Kong）  
**角色**：策略研究执行助手  
**约束**：不下数据、不付费、不开 Cloud Agent；仅 metadata.get_cost + 规格/代码/旧账阅读。  
**区间口径**：EQUS.MINI 可用末日 = **2026-10-23**（API：`data available up to 2026-10-24 00:00:00+00:00`，`end` 为开区间；2026-10-24 美股虽已收盘，数据集尚未覆盖该日）。研究窗口写 **2023-03-28 → 2026-10-23**。

---

## 1. 核对结论（一句话）

**PASS 可开跑**——规则可用 EQUS.MINI `ohlcv-1m` 真实表达（AVWAP 过滤 + 可选同 bar 分钟复核）；须走「先日线 PB 预筛 → 再按 (symbol,h..t) 累加 get_cost/下载」路径，**禁止全市场分钟**；下载前必须门户确认 Credit 余额；Credit 不够即停；结果只能标 **EXPLORATORY**。

---

## 2. 逐项核对表

| # | 项 | 判定 | 证据 |
|---|---|---|---|
| 1 | **AVWAP 可表达** | **PASS**（附强制口径） | 规格 §3 / D09：锚点=h 日开盘；`AVWAP=sum(typical×vol)/sum(vol)`，`typical=(H+L+C)/3`，从 h 到 t **常规时段**；`C(t)>AVWAP`；不含 t+1。`metadata.list_fields(EQUS.MINI, ohlcv-1m, dbn)` → `ts_event, open, high, low, close, volume`，字段够算 typical×vol。单价 ohlcv-1m = **$12/GB**。**未复权**：官方 OHLCV 为成交聚合价，无拆股调整字段（旧 SUMMARY 亦写明）。**强制处理**：(a) 与 `C(t)` 比较须同一价格空间——用引擎已有原始价 `C_raw=C×F` 对 Databento 未复权 AVWAP，或本对比两臂都改用 EQUS 日线未复权；(b) `[h,t]` 内若发生拆股 → 该信号 **排除或按拆股比回推** 后再累加，禁止混尺度求和；(c) 常规时段客户端按 `ts_event` 切 **09:30–16:00 America/New_York**（含半日市日历）；盘前/盘后 bar 不入 AVWAP；(d) h 日早于 2023-03-28 的信号无法从锚点完整累计 → **DATA_BLOCKED/排除**（见下：约 95 个）。 |
| 2 | **盘中触发含义** | **PASS**（范围写死） | 规格 §3：「PB-AV **唯一扩展**」= 追加 `C(t)>AVWAP`，「**其余完全一致**」。日线版执行（§7）：信号日 t 收盘后，**t+1 挂 B=H(t)+0.01 的 stop-market**（跳空/风险取消规则不变）。**分钟线不用于改写入场触发或主路径止损**；入场仍是次日日线挂单仿真。**「盘中意思」实际覆盖**：(i) 信号形成层用 h→t 常规分钟算 AVWAP 过滤；(ii) §7 要求的同日入场且 L≤S 歧义笔，可用 1m **复核**（见项 3）。**不覆盖**：盘中即时触发、盘中动态止损替换日线规则、Martin 本人盘中裁量。若 PM 期望「分钟入场/分钟止损主路径」→ 超出 v1.1 规格，本对比不实现。 |
| 3 | **同 bar 入场+止损** | **PASS** | 规格 §7：同日入场且日线 L≤S → **保守计当日止损**，单列歧义，并用分钟复核；引擎 `engine.py` 已实现 `stop_same_bar` / `ambiguous=1`。**1m 判定算法（建议写入 Cloud Agent）**：对入场日 RTH 1m bars 按时间序；找第一根满足 `H≥B` 的 bar 为入场时刻（若开盘 `O≥B` 则视为 bar 开盘入场）；入场后找第一根 `L≤S` 为止损。若入场与止损落在**不同分钟**→ 可消歧并记录实际顺序；若落在**同一根 1m**（该 bar 同时 `H≥B` 且 `L≤S`）→ 仍无法知先后 → **保持保守止损**，标记 `ambiguous_1m=1`。主结果仍以日线保守为准；分钟只出复核表，不把乐观顺序当主结果。 |
| 4 | **数据覆盖** | **PASS（不阻断对比；可信度降级）** | `get_dataset_range(EQUS.MINI)`：ohlcv-1m/1d 自 **2023-03-28** 起，可用至 **2026-10-23**（见上）。未复权；无退市清算现金；现有 Yahoo 池仍有存活者偏差（`data_audit.md`）。PM 已接受 2023+ 窗口 → **不阻断 PB vs PB-AV 对比本身**；全部格子必须标 **EXPLORATORY**，不得 RESEARCH_PASS。 |
| 5 | **密钥** | **PASS** | 本机环境 `DATABENTO_API_KEY` 已注入（len=32）。Cloud Agent 只用 `secret_names`，**禁止写入仓库/报告明文**。本文件无 key。 |

**任一项无法用数据真实表达则 STOP**：本轮五项均可表达（在强制口径与 Credit 护栏下）→ **不 STOP**。  
**硬停条件（开跑后仍适用）**：累计 get_cost/实际账单将超剩余 Credit；试图拉 ALL_SYMBOLS 分钟；余额未门户确认仍下载；用日线近似 AVWAP。

---

## 3. 费用估算表 + 建议下载顺序

### 3.1 本次 get_cost 原始数字（2026-10-25 HK，metadata only，无 download）

| label | dataset | schema | symbols | start | end | cost_usd |
|---|---|---|---|---|---|---|
| D_EQUS_ohlcv1d_ALL_20230328_to_20261024 | EQUS.MINI | ohlcv-1d | ALL_SYMBOLS | 2023-03-28 | 2026-10-24 | **12.987280413508** |
| D_EQUS_ohlcv1d_ALL_end_20261023 | EQUS.MINI | ohlcv-1d | ALL_SYMBOLS | 2023-03-28 | 2026-10-23 | **12.971793785691** |
| D_EQUS_ohlcv1d_20sym_full | EQUS.MINI | ohlcv-1d | n=20 | 2023-03-28 | 2026-10-24 | **0.028100609779** |
| M_EQUS_ohlcv1m_ALL_20230328_to_20261024 | EQUS.MINI | ohlcv-1m | ALL_SYMBOLS | 2023-03-28 | 2026-10-24 | **575.756096184254** |
| M_EQUS_ohlcv1m_ALL_end_20261023 | EQUS.MINI | ohlcv-1m | ALL_SYMBOLS | 2023-03-28 | 2026-10-23 | **575.045466810465** |
| M50_1day_full | EQUS.MINI | ohlcv-1m | n=50 | 2024-06-03 | 2024-06-04 | **0.012613981962** |
| M50_4days≈200td | EQUS.MINI | ohlcv-1m | n=50 | 2024-06-03 | 2024-06-07 | **0.050405859947** |
| M50_10cal≈350td | EQUS.MINI | ohlcv-1m | n=50 | 2024-06-03 | 2024-06-17 | **0.126479029655** |
| M50_1day_RTH_0930_1600ET | EQUS.MINI | ohlcv-1m | n=50 | 2024-06-03T13:30:00 | 2024-06-03T20:00:00 | **0.011668324471** |
| M_AAPL_10cal | EQUS.MINI | ohlcv-1m | AAPL | 2024-06-03 | 2024-06-17 | **0.003117978573** |
| M_AAPL_1day_RTH | EQUS.MINI | ohlcv-1m | AAPL | 2024-06-03T13:30 | 2024-06-03T20:00 | **0.00024408102** |
| M_20sym_10cal | EQUS.MINI | ohlcv-1m | n=20 | 2024-06-03 | 2024-06-17 | **0.055703043938** |

派生：**每 ticker-day（全时段，50 液态样本）≈ $0.00025228**（= 0.012613981962/50）。液态股偏密 → 作 **上界单价**。RTH 相对全时段约 **92.5%**（0.011668/0.012614）。

`end=2026-10-25` → API **422 data_end_after_available_end**（证实可用上界为 2026-10-24 00:00 UTC）。

### 3.2 必须先有日线 PB 信号列表才能精确 quote 分钟

AVWAP 窗口是每个信号自己的 **[h,t]**（h∈[t−10,t−2]），失败信号也要拉（否则无法判 `C(t)>AVWAP` 失败）。**不能**在无预筛时用 ALL_SYMBOLS 分钟硬跑。

用现有 Yahoo 矩阵 + `sig_pb`（与引擎一致，**未下载任何新数据**）在 2023-03-28..2026-10-23 统计：

| 量 | 值 |
|---|---|
| 日线 PB 信号格数 (sym,t) | **12,227** |
| Σ(h..t) 未去重 ticker-days | **83,401**（均窗 ≈6.82 日） |
| **去重后可拉取 unique (sym,day)** | **48,812** |
| h 早于 EQUS 起点的信号 | **95**（完整 AVWAP 不可算 → 排除/DATA_BLOCKED） |
| 涉及股票数 | **1,598** |

### 3.3 两档费用

| 档 | 含义 | 估算 USD | 备注 |
|---|---|---|---|
| **禁止上界** | EQUS.MINI ohlcv-1m ALL_SYMBOLS 全区间 | **≈ $575–576** | 远超赠送/剩余 Credit → **硬停路径，禁止** |
| **样本外推上界** | 无预筛时用 50/200/500 ticker-day × $0.000252 | 50→**$0.013**；200→**$0.050**；500→**$0.126**；2000→**$0.50**；5000→**$1.26** | 仅作量级；真实需求 ~5 万 ticker-day |
| **计划路径（推荐）** | 先日线 PB → 去重 (sym,day) get_cost 累加再下 | 48,812 × $0.000252 ≈ **$12.3**（全时段液态上界）；RTH 请求约 **$11.4**；薄票可能更低 → 实务区间约 **$8–$15** 分钟 | 另加同 bar 复核入场日（百级 ticker-day，<$0.1） |
| **日线（可选）** | EQUS.MINI ohlcv-1d ALL | **≈ $13.0** | **非必须**：盒子已有 Yahoo 日线可跑 PB 预筛；若为与分钟同库一致可下，但占 Credit |
| **推荐合计** | Yahoo 日线（已有，$0）+ 预筛分钟 | **约 $8–$15** | 若加 Databento 日线 → **约 $21–$28** |

### 3.4 对照旧账 / Credit

来源：`/workspace/qullamaggie/databento_orh_quote_20261022/SUMMARY.json`（2026-10-22）：

- 文档注册赠送 **$125**；先验花费重建约 **$72.7** → 若仍有效约剩 **$52.3**
- **门户未登录，余额未核实证**（billing API 曾 401；本轮仍未登录门户）
- 全市场分钟 ≈$576 ≫ 任何合理剩余 → 再证必须预筛

**下载前必须先确认门户余额。** 累计 get_cost 将超余额 → **停，一分钱都不额外付，不绑卡不充值。**

### 3.5 建议下载顺序（Credit 护栏）

1. **门户确认**剩余 Credit；记入结果文件。不够覆盖计划路径下界（建议预留 ≥$20 缓冲）→ STOP。  
2. **不下** ALL_SYMBOLS `ohlcv-1m`。  
3. 用现有 Yahoo 日线跑 PB（2023-03-28..2026-10-23）→ 产出 `pb_signals_2023plus.csv`：每行 `symbol,t,h,B,S,...`（含失败于 AVWAP 前的全部日线合格信号）。  
4. 展开去重 `(symbol, date)` 且 `date∈[h,t]`（及可选 entry day）；对每个批次 `metadata.get_cost` **累加**；若累计 > 剩余 Credit−缓冲 → **停**。  
5. 仅对通过 quote 的批次 batch download；失败信号与成功信号同样保留。  
6. 算 AVWAP → 滤信号 → 与纯 PB 同池同时段同成本 10/30/50bp 对比。  
7. 同 bar 歧义子集用 1m 复核表。  
8. 全程 EXPLORATORY；不写 PASS。

---

## 4. Cloud Agent 任务草稿（中文要点，供主 agent 粘贴）

**标题**：COMP v1.1 · PB vs PB-AV（2023-03-28→2026-10-23）· Credit 护栏下载与回测  

**做什么**  
1. 注入 secret `DATABENTO_API_KEY`（`secret_names`），禁止写入仓库。  
2. 先读本 preflight 与 `spec_v1.1.md` §3/§7、D08/D09；确认门户余额（若 Agent 无法登录门户，则要求人工把余额数字写入环境/文件后再继续）。  
3. **不要**下 EQUS.MINI ALL_SYMBOLS 分钟。  
4. 用盒子已有 Yahoo 矩阵跑 `sig_pb`，窗口 2023-03-28..2026-10-23，导出全部日线 PB 信号（含后来会因 AVWAP 失败者）及 h。  
5. 对每个信号展开 `[h,t]` 交易日，去重 `(symbol,date)`；按批 `get_cost` 累加；超 Credit 即停并留痕。  
6. 下载 EQUS.MINI `ohlcv-1m` 仅针对上述集合；客户端滤 09:30–16:00 ET；算 AVWAP（锚 h 开盘，typical=(H+L+C)/3）；`C_raw(t)>AVWAP`；拆股穿越窗口则排除该信号。  
7. 跑 PB 与 PB-AV 三成本情景（10/30/50bp），同池同时段同资金规则；分钟复核同 bar 歧义。  
8. 全部标记 EXPLORATORY。  

**成功标准**  
- 产出信号表、分钟 manifest（含每批 get_cost 与实际花费）、AVWAP 中间值抽检、trades/equity/metrics、PB vs PB-AV 对比表、同 bar 复核表、排除清单（h 早于 2023-03-28、拆股窗口、缺 bar）。  
- 零 ALL_SYMBOLS 分钟下载；零绑卡/充值；花费 ≤ 确认余额。  

**Credit 护栏**  
- 累计 quote > 余额−$5 缓冲 → 立即 STOP。  
- 单笔 get_cost 若异常接近全市场量级 → 人工复核后再下。  

**结果文件清单（写回共享路径，不要只存在 Agent 本地）**  
- `pb_av_vs_pb_manifest.json`（花费、余额声明、批次）  
- `pb_signals_2023plus.csv` / `pb_av_signals.csv`  
- `trades_*.csv` `equity_*.csv` `metrics_compare.csv`  
- `same_bar_minute_review.csv`  
- `README_结论_EXPLORATORY.md`  

**不要退回本机下载大数据**；大文件留在约定的 workspace/对象存储路径，本机只取摘要与小表。

---

## 5. 给 PM 的一句话回执草稿

核对结论：**PASS 可开跑**（PB-AV 的 AVWAP 可用 EQUS.MINI 1m 真实表达；入场仍是次日日线挂单，分钟只做 AVWAP 过滤+同 bar 复核）；预估花费约 **$8–$15**（预筛去重分钟，液态上界约 $12；全市场分钟 ≈$576 禁止）；Credit 门户余额未核实（旧账重建约剩 $52），**下载前必须先确认余额，不够就停**；结果仅 EXPLORATORY。详情：`/workspace/gpt-candidates/pb_av_vs_pb_preflight_20261025.md`。

---

## 附录 A · 规格锚点摘录

- §3 PB-AV：锚点 h 开盘；AVWAP 公式；`C(t)>AVWAP`；无分钟则 DATA_BLOCKED；禁止日线近似。  
- §7：次日 stop-market；同 bar 保守止损 + 分钟复核。  
- D08/D09：执行默认；AVWAP 扩展数据不足则阻断该格。  

## 附录 B · 引擎现状

- `sig_pb` 无 AVWAP；`E08_PB_AV` 现为 DATA_BLOCKED（`run_all.py`）。  
- `simulate` 已实现同 bar 保守 `stop_same_bar`。  
- 本对比落地需新增分钟 AVWAP 过滤层，**不**改日线入场主路径。
