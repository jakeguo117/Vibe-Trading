# MOM-XASSET-001 跨资产动量研究执行计划 v1.0

> For agentic workers: 若具备 superpowers:executing-plans，按该技能逐任务执行；Hermes 未安装时直接按本计划检查项原生执行。执行方式已由 Jake 指定为 MacBook 本地 Hermes，不需要另选执行模式或组建多 agent 团队。

**Goal：** 在限定预算内自动搜索入场、退出、仓位规则；用 GLD 开发并冻结候选，检验其在后续 GLD、USO 和 SPY 上能否同时改善年化收益、最大回撤、Sharpe。允许结论为没有合格候选。
**Architecture：** 一个本地可复现的日线回测程序，配置驱动的有限网格搜索，先开发、后锁定、再统一评估。Drive 保存研究计划和证据；Notion 保存进度；研究代码只在 Vibe-Trading。
**Tech Stack：** 优先复用现有 Python 环境、pandas/numpy/pytest、现有行情缓存；图表用 matplotlib。无需 GPU、LLM 逐轮推理、模型训练平台或新付费服务；“训练”指参数搜索。
**Spec：** 本文件第 1–10 节是冻结研究规格，第 11 节是执行任务；本文同时携带规格与实施计划。
**日期：** 2026-10-21，Asia/Hong_Kong。市场日使用 America/New_York。
**状态：** 已设计待测试；Jake 已授权编写计划、由 Hermes 接手开始研究。本文具体网格与阈值由研究设计者设定为首轮可执行默认值，不宣称 Jake 逐项审定；Hermes 收到启动 prompt 后按此版执行，不能根据结果改阈值。本 ChatGPT 会话未启动 Hermes 或实际训练。
**研究 ID：** MOM-XASSET-001；这是新分支，不覆盖 MOM-GLD-001 R1/R2/R3。

## 1. 固定入口、权限和范围

- 本研究文件夹：https://drive.google.com/drive/folders/1tBa54vyB67CUxOehI4RYcDBjKGSSUaSY
- 策略研究总入口：https://drive.google.com/drive/folders/1WvLQ5vOLcaFC3LIFN6SHdx1GCpXSTRVZ
- START HERE：https://drive.google.com/file/d/1bZhPOT58enw7wAfCmdlUxNf4NM_UnShj/view
- 结果正本根目录：https://drive.google.com/drive/folders/1kRID4peUueFv10wFC2wrkrICL5hw40eI
- Notion：https://app.notion.com/p/3f5474970e4081578c89e7de18c0760f
- 研究仓库：https://github.com/jakeguo117/Vibe-Trading
- 正式系统：https://github.com/jakeguo117/trading-system（本任务只读，禁止修改生产参数、调度或下单）。

首轮仅做 GLD、USO、SPY 日线 ETF 正股，只做多或现金，不杠杆、不期权、不做空、不轮动、不将三个账户组合成一个多资产收益曲线。每个资产独立模拟 16,000 美元账户；“跨资产”是相同规则的迁移测试。
SPY 是预先指定的第三市场检验，不能看了结果再替换成表现更好的资产。SLV/QQQ/TLT 等留待另轮，不在本轮扩大资产池。

本任务允许：本地研究代码、现有或免费数据获取、完整研究运算、Drive 归档与 Notion 状态更新。不得购买行情、订阅、云算力、调用收费行情 API 或新增付费服务。现有工具套餐内正常使用不另作采购。不得触发其他 agent、向第三方发消息或提交交易。

已知 Mac 路径线索来自旧复现报告，执行前必须核实：
- /Users/jake/projects/trading/Vibe-Trading
- /Users/jake/projects/trading/Vibe-Trading-mom
- /Users/jake/projects/trading/Vibe-Trading/.venv/bin/python
目录、Python 版本不是本次已核验事实。读取仓库 AGENTS.md，检查工作树状态；在研究仓库创建独立分支/工作树，保留未提交改动；不覆盖旧 run，不编辑规则文件。常规研究代码可本地提交留痕，不自动推送或合并远端。

## 2. USO 数据调查：现在先不买

### 本次已读证据
A. TRD-DATA-COMPARE-0001：
https://drive.google.com/file/d/1bTzN2cKWt0pdsGJ5T5LTQ8hcT6LpXEkz/view
报告记录 Yahoo 调整日线已存在：
- data/adjusted/USO.csv：5,158 行，2006-04-10 至 2026-10-09；
- SHA256：3c85b0cc15459b937ec39ef9a76abd4f76c501d2b5f772663328806993ca8da0；
- 验收快照 USO_yahoo_adj.csv：SHA256 70ba7c2d584639da07585ef1536c22a0439f198167ccff288e30778dcd6aa5eb；
- 2006-04-10 至 2026-09-30 的共同 5,151 行 OHLCV 比对无差异；
- 数据目录不提交 Git，报告中存在文件路径不代表 GitHub 能下载原文件，也不代表 Hermes 当前机器上文件必然仍在。

B. 2026-10-18 单标的研究报告：
https://drive.google.com/file/d/1-C8d7IxrAmrys1nwjkhXuKPIkShSob0P/view
报告记录 GLD、SLV、USO、SPY、QQQ、TLT 调整日线已使用至 2026-10-16，来源 Yahoo/yfinance。
这支持“历史上已取得并使用 USO 和 SPY 日线”，并非本次重新验收其原始字节。

C. 本次 Drive 按精确名称查 USO.csv / USO_yahoo_adj.csv / acceptance_v1.0.zip 未找到匹配；GitHub 默认分支 USO 搜索也未返回结果。不能据此认定数据不存在，可能保存在 Mac 缓存、未合并分支或压缩包中。
本轮 ChatGPT 没有直接读取 MacBook，也没有重新下载或比对完整 USO 数据。当前判断：不需要先购买，Hermes 先盘点和验收，缺少的尾部或字段优先免费补齐。

### 获取顺序
1. 在已核实的交易项目目录里用 rg --files 定位 USO/GLD/SPY 数据、数据 manifest、旧 run/config 和抓取脚本。优先复用不可变快照，记录实际绝对路径与 SHA256。
2. 找不到时，复用已有授权行情管道或 Yahoo 日线来源，保存请求参数、抓取时间、依赖版本、原始响应和规范化文件。核对数据源使用条件；不规避限流/登录/付费墙。
3. 请求日线 OHLCV、调整因子和公司行动。每个网络请求最多 3 次合理重试；不可用时报告数据缺口，不擅自换付费源。
4. 截止日固定为 2026-10-20 的已完成美股交易日，不因执行日推迟而自动向后延长。先补齐至该日；补齐失败可以生成较早截止的诊断，但全量验收标 INCOMPLETE。
5. 原油现货、CL 连续期货不能替代 USO；不得把 2006 年上市前替身序列接入主样本。

### 公司行动和质量要求
- USO 官方文件确认 2020-04-28 收盘后 1:8 反向拆股，2020-04-29 起拆股后交易。来源：
https://www.sec.gov/Archives/edgar/data/1327068/000117120020000271/i20263_ex99-1.htm
- 因此旧报告“未报告公司行动”不等于 USO 没有拆股；actions=False 的调用也不能用来证明没有行动。
- 必须核对 2020-04-27 至 2020-04-30：原始价、调整价、拆股因子、调整前后收益连续性。不得出现人为约八倍涨幅。
- USO 是期货敞口产品，换月和历史持仓结构变化可能影响回测；只解释为 USO 的可交易表现，不能外推为无换月影响的现货油策略。ETF 价格已包含基金内部换月与费用影响，不再次扣一份同样费用。
- 检查唯一/递增交易日、缺失 NYSE/NYSE Arca 会话、OHLC 合法性、非正值/NaN、时区、重复、数据修订；不填充缺失交易日行情，不因极端涨跌删除真实观测。
- 对 USO 拆股窗口、2020 极端行情及每资产首尾各 5 个交易日进行源数据核查。两份 Yahoo 快照一致只证明可复现，不算独立数据源验证；第二来源不可得必须记录。
- 所有调整 OHLC 必须同一尺度。SPY 含分红总回报处理，调整价收益不得再加一遍分红。
- 数据门槛：GLD、USO、SPY 从 2006 年可用历史至截止日，2007 起测试所需窗口完整；缺核心 GLD/USO 时停止主研究；SPY 缺失可以完成两资产部分，但三资产结论 INCOMPLETE。
- data_audit.json 输出 source、path、sha256、rows、first/last、adjustment_method、actions_check、缺失日、warnings、verdict。

无需本轮购买分钟、逐笔、报价盘口、期权链或专业期货数据。若免费日线最终达不到要求，交付 gap_report.md，列缺哪些字段/日期、试过哪些来源及其错误；采购是后续单独决策。

## 3. 时间分割和防泄漏

目标历史：2006 年可用日起至 2026-10-20；2006 年仅用于 2007 年起指标预热。
训练/开发仅使用 GLD，2007-01-01 至 2015-12-31（边界映射到真实交易日）。
开发期分为三个评分块：2007–2009、2010–2012、2013–2015。每块从 16,000 现金重置账户，指标可使用此前已有数据；另外跑 2007–2015 连续曲线。
候选锁定后统一评估：
- GLD 2016-01-01 至 2026-10-20：时间迁移。
- USO 同期：核心跨资产迁移。
- SPY 同期：不同市场补充检验。
- 每资产主曲线连续跑 2016 起；同时报告 2016–2019、2020–2022、2023–截止日各自从现金启动的分段结果及逐年主曲线收益，明确两种口径，不能混算。

本轮采用一次训练、固定参数，不年度重选参数。这样先回答“同一套规则能否迁移”。年度滚动重训属于未来独立研究，不混入本轮。
选择参数时不能读取任何 USO/SPY 策略绩效，不能读取 GLD 2016 后绩效。数据工程可检查全段质量，但不得据之调整交易规则。
训练函数输入必须物理截断至 2015-12-31，不仅靠文档声明。冻结文件记录候选、数据 hash、代码 commit、参数网格 hash、排序公式与时间戳。
本项目此前已看过 GLD、USO、SPY 的历史研究，因此这里属于“算法隔离的历史时间/跨资产迁移检验”，不是完全未知的独立历史样本，也不是统计独立资产。不得称干净样本外、保证泛化或实盘验证。
本轮失败后不能以“修正”为由继续看着测试成绩改参数；只允许修复可复现的代码/数据错误，须记录影响范围并重跑所有受影响路径，不能只重跑赢家。

## 4. 冻结策略空间：180 组

两个入场家族，不增加宏观、Fib、机器学习、资产专属例外。
所有信号在交易日 t 收盘后计算，最早 t+1 开盘执行。

### 入场 9 个定义
A. 收盘突破：C_t > max(H_{t-N},...,H_{t-1})，N = 20、40、60，共 3 个。
B. 均线趋势状态：C_t > SMA_slow(t) 且 SMA_fast(t) > SMA_slow(t)，fast = 10、20、40，slow = 60、120，共 6 个。SMA 包含 t 收盘，不是必须“金叉当天”。
空仓且条件成立即可产生买单；两家族均用同样再入场规则，不能给某资产加额外条件。

### 退出 5 个定义
A. 收盘跌破通道：C_t < min(L_{t-M},...,L_{t-1})，M = 10、20，共 2 个。
B. 收盘跟踪阈值：C_t <= peak_close_since_entry - k*ATR14_t，k = 2、3、4，共 3 个。peak 仅用持仓期间已完成收盘，包括当前 t。
所有路径另有固定基础退出：C_t <= entry_reference_open - 4*ATR14_signal_day。
此基础阈值用于统一风险尺度；它仍是收盘触发、下一开盘成交，不是保证亏损上限的盘中止损。
ATR14 用 TR=max(H-L,abs(H-C_prev),abs(L-C_prev)) 的 14 日简单均值，固定定义，禁止库默认 Wilder 平滑悄悄替换。

### 仓位 4 个定义
A. 固定投入净值 50% 或 100%，共 2 个。
B. 风险预算为净值 0.5% 或 1.0%，共 2 个。入场数量=q_risk=r*E/(4*ATR14_signal_day)，同时受现金可负担数量限制，不杠杆。
入场后持有数量固定，不每日再平衡、不加仓、不分批止盈。风险预算是按信号日波动计算的模型预算，跳空、下一开盘执行可使实际损失超过预算。

### 再入场及优先级
平仓优先；退出当日收盘不生成新入场单，至少等下一个交易日收盘重新评估，最早再下一开盘买入。不能同日卖出后立即买回。每路径最多一个待执行指令；延迟场景中的已生成指令不因等待期间新信号取消/替换，执行前不再生成同方向重复指令。测试窗口首日从现金启动，只在首日收盘起产生策略信号，不继承窗口前持仓或挂单；基准按定义首日开盘买入，该差异固定披露。
组合数=9*5*4=180，全部登记唯一 param_id。不得边看结果边加入新规则或超出范围搜索。

## 5. 成交、复权、现金和成本

### 主研究引擎
主搜索允许使用同尺度的拆股/分红调整 OHLC + 小数“调整后份额”，这是总回报归一化研究账户，份额不是历史真实股票股数。
理由：现有数据已是调整日线；此模型便于公平比较而不会因 USO 拆股产生假收益。不得将这些份额直接当可下单股数。
所有信号/阈值/ATR/成交参考在同一调整尺度；主引擎不重复派发分红或再改变份额处理拆股。
每条测试区间初始 16,000 美元，无外部入金提款。
入场目标美元或风险数量只由 t 收盘账户权益、t 指标决定。t+1 的开盘价仅作实际成交和现金可负担上限，不用于择时或重新选参数：
- fixed：计划数量=w*E_t/C_t；
- risk：计划数量=r*E_t/(4*ATR_t)；
- 实际数量=min(计划数量, available_cash/[O_next*(1+cost)])；
- 买入现金扣除 q*O_next*(1+cost)；卖出获得 q*O_next*(1-cost)。
entry_reference_open=不含成本的成交开盘参考，止损阈值不随成本变动；成本仅改变现金/权益/数量，避免 R3 中成本改变止损触发时点的问题。
持仓信号不能引用权益或成本，因此同参数不同成本的交易时点原则上相同（仅现金不足的极端情况例外并解释）。

### 成本场景
单边 all-in 5bps 为主场景，20bps 为压力场景，均作用于实际成交额，涵盖研究假设中的佣金/点差/滑点，不额外叠加另一份旧佣金。
第三场景：5bps、所有信号额外延迟一个交易日开盘执行，退出等待期间忽略新入场，不能在缺失价格时任意填价。
成本是统一研究假设，不声称匹配用户真实券商账单。未来实盘须按券商规则复核。
区间最后收盘按该收盘价格作研究结算并扣卖出成本，各策略和基准统一。此为会计期末结算，不是用未来信息产生的常规交易信号，单独标记并从普通交易次数中区分。

### 现金与 Sharpe
主账户现金利息=0：当前未核实用户券商对 16,000 美元账户的真实付息规则，不假设能够拿到全部国债收益率。
另做 cash_proxy 场景：未投资现金按滞后两次美国交易会话的 DGS3MO 日度年率近似计息，按实际日历天数 accrual=y/100*days/365，余额用上一收盘现金；称“国债利率代理情景”，不称可获得的券商收益。
Sharpe 主定义：sqrt(252)*mean(r_day-rf_day)/std(r_day-rf_day,ddof=1)，rf_day 同上述滞后国债率与日历天数；另报 rf=0 的 Sharpe0 便于对比旧报告。
年率为负时保留，不擅自截零。DGS3MO 缺值只允许用已发布过去值向前填充不超过 7 个日历日，不向后填；缺更长则数据门槛不通过。
DGS3MO 是收益率代理，非真实滚动短债组合总回报。若无法免费获取，可先跑零利率探索，但主验收标 INCOMPLETE，不能靠替换 Sharpe 定义达标。
来源：https://fred.stlouisfed.org/series/DGS3MO
不存在“未投资现金多所以必然是择时 alpha”的归因。

### 真实股数执行复核
历史迁移达标的第一候选还需在 GLD/USO/SPY 对应原始历史价格与公司行动账本上做整股复核，再称 PAPER_READY：
- 不用调整价直接 floor 成所谓真实股数；拆股按事件日期调整股数/价格/阈值，SPY 分红按除息资格、可确认支付日期计现金；
- 支付日期不可得可记录应收分红到估值，但不能假设提前可用来再投资；无法可靠实现则标 EXECUTION_NOT_VERIFIED。
- 使用相同经济信号、同一目标美元/风险预算和成交成本，缺少原始数据不停止探索，但不得宣称可执行验收通过。
- 整股后核心 GLD/USO 三指标仍须严格优于对应整股买入持有，最大回撤不超过 30%；未通过则标 EXECUTION_FAIL。
原始账本的实现方式可按现有可靠引擎适配，但不能修改经济规则。主目标不是在本轮开发复杂券商系统。

## 6. 比较基准与指标

每资产分别对比，使用同一日期、数据、成本与初始资金：
1. BH100：首个测试日开盘一次买入至现金允许的最大数量，持有至末日结算。
2. BH50-M：首日 50% 资产/50%现金，之后每个月最后交易日收盘确定 50% 目标，下一交易日开盘调仓；每次真实换手计成本。
3. CASH：全现金，在主场景收益 0；在 cash_proxy 场景按同一代理率计息。
4. 可实现的风险预算基准 VT10：过去 60 个收盘收益的样本年化波动，目标权重 min(1,0.10/vol60)，每月末信号、次日开盘执行；预热不足或零波动不持仓。它是固定规则参照，不是“事后精确匹配风险”。每段披露实际波动比，偏离 25% 不宣称同风险比较。

CAGR=(末权益/初权益)^(365.25/实际日历天数)-1。
最大回撤 MDD 为负数；DrawdownMagnitude D=-MDD（非负）。
收益日包含所有可交易日和现金日；不只统计持仓日。末日结算成本入净值。
同时输出 Sharpe/Sharpe0、波动、逐年收益、暴露天数、时间加权仓位、换手、总成本、完成交易数（不含行政期末强平）、持仓时长、最差单笔。
USO 若买入持有长期亏损，空仓也可能轻易超过它，所以必须同时列现金与简单风险预算基准；超过 USO 买入持有不自动说明高质量策略。

## 7. 开发期选参：确定性，不看验证结果

对每组参数在 GLD 开发连续段及三个分块计算：
dc=CAGR_strategy-CAGR_BH100；
ds=Sharpe_strategy-Sharpe_BH100；
dd=D_BH100-D_strategy（正数表示回撤更小）；
S=min(dc/0.02, ds/0.20, dd/0.05)。
这是工程排序尺度，不是显著性或未来收益概率。采用 min 是为了防止只靠一个指标极好掩盖另一个很差。

基础资格：开发连续段 CAGR>0、Sharpe>0、D<=30%，已完成交易>=20，每开发分块>=3。无资格候选就输出 NO_CANDIDATE，不放宽门槛。
原始稳健分= min(S_连续段, median(S_三个分块))。
邻居=同入场家族、同退出家族、同仓位模型下，只把一个数值参数换为网格内相邻值；其余不变。MA fast/slow 分别作为独立数值参数。邻居参数也必须来自登记的 180 组。
稳定分=min(自身原始稳健分, median(全部相邻参数原始稳健分))；邻居因无交易/数据异常没有分数时记 -infinity，不能跳过差邻居。
排序：稳定分降序 → 自身原始稳健分降序 → 换手升序 → param_id 字典序。
输出最多 3 个基础合格候选；第一名为唯一 primary，第二/三名为预先冻结的 sensitivity，不得看 USO 成绩后将它们升为主策略。
即使最高 S<=0，可把它作为“未达开发三胜目标的诊断候选”继续迁移测试；报告必须写清开发目标未达，不许称训练成功。
保存全部 180 组包括失败/无交易组合；固定随机种子 20261021（网格本身不随机），不使用贝叶斯优化/遗传搜索/LLM 自创迭代。

## 8. 验证和通过标准：事先固定

### 每个资产的三项经济目标（5bps、主现金口径，2016 至截止）
- CAGR 比该资产 BH100 高至少 1 个百分点；
- 最大回撤幅度比 BH100 小至少 2 个百分点；
- Sharpe 高至少 0.10；
- 同时 CAGR>0、Sharpe>0、最大回撤幅度<=30%，完成交易>=30，每个验证分块>=5。
以上是本轮预设研究门槛，不是统计显著性证明，也不是历史最大回撤的未来保证。

### 稳健性门槛（对 primary，GLD 与 USO 必须分别满足）
- 三个验证分块中至少 2 个同时 dc>0、ds>0、dd>0；任何分块 D 不超过 30%。
- 20bps 场景整段仍 dc>0、ds>0、dd>0，D<=30%。
- 额外一日延迟场景 CAGR>0、Sharpe>0、D<=30%，ds>=0。
- cash_proxy 场景同口径与 BH100 比较，dc、ds、dd 都>0，不能只在现金零利息下表现好。
- 局部参数压力：在查看任何验证绩效之前，冻结候选每个数值参数单独上下 20% 的版本，整数窗口四舍五入且>=2，仓位权重限制<=1，fast<slow，不生成非法重复项。其余不变。最多每候选 12 个、合计 36 个新增参数配置。只作压力测试，不能用于选新的赢家。
- primary 的合法扰动中至少 60% 在 GLD、USO 各自满足 dc>0、ds>0、dd>0；各自扰动结果 ds 中位数>0。固定仓位 100% 的上扰动裁到 100% 后去重并披露分母，禁止补找其他有利扰动。
- BH50-M、VT10 作为竞争基线：各自报告；若 primary 在核心资产上相对任何一个基准的 CAGR、Sharpe、D 全面更差，则不得升为 PAPER_READY。

### 证据强度
对 primary 与 BH100 的逐日收益做配对移动区块 bootstrap，区块长度 20 交易日、2,000 次、固定种子，重新计算 dc/ds/dd，报告各自 95% 区间；不得称 bootstrap 是新市场历史。
开发阶段以全部 180 组试验数报告 Deflated Sharpe Ratio（DSR），使用日频单位、样本长度、偏度峰度，说明相依试验及近似假设；不把 DSR 当“未来赚钱概率”。若现有可信实现不可用，不自造概率，写未计算；经济结果可以交付但统计确认不得通过。
即使统计结果漂亮，既往历史暴露也限制“独立验证”说法。本轮最多给历史迁移候选，不批准实盘。
资料：https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf

### 结论标签
- NO_CANDIDATE：开发期无基础合格候选。
- NO_PASS：primary 未达到 GLD+USO 任一核心目标；不得挑 sensitivity 改名宣布胜利。
- PAIR_PASS：primary 在 GLD+USO 的经济目标、稳健性均通过；仅支持这两资产的历史迁移。
- BROAD_PASS：PAIR_PASS 且 SPY 也达到三项经济目标和分段/成本/延迟/cash_proxy/扰动标准。SPY 不通过不允许称跨市场普遍有效。
- INCOMPLETE：必要数据、测试或产物不完整；必须列具体缺口，可以附已完成部分。
- PAPER_READY：PAIR_PASS 或 BROAD_PASS，数据与执行账本验收通过，DSR>=0.95，核心 GLD/USO 三项差值 bootstrap 各自下界>0。其含义仅为值得进入未来模拟观察，非已获实盘批准。
规则失败、样本太少、统计不确定必须区分。若只改善部分指标，以 PARTIAL_TRADEOFF 附注呈现，不降低用户三目标。

## 9. 预算、断点和停止

- 参数配置上限：180 主网格 + 最多 36 个冻结扰动；3 个候选最多，不自动新一轮。
- 一个 engine evaluation 定义为“一个参数/基准 × 一个资产 × 一个日期窗口 × 一个成本/现金/延迟场景”；总上限 3,000 次，manifest 分开记录参数数与评估次数；bootstrap 不算策略重跑。
- 首轮正式计算累计墙钟上限 8 小时，从数据通过且引擎自检完成后计时。时间是预算，不是耗时承诺；达到任一预算保存 BUDGET_EXHAUSTED，不延长或随意抽掉失败任务。
- 实施与排错时间单列；同一技术问题最多 3 次有证据的修复尝试，仍失败就交阻塞包。
- 无需逐参数请示；常规环境/兼容问题可自主修复。不得为预算扩大或采购寻求后台自动批准。
- 每完成阶段更新本地 STATUS.md，计算期间至少每 15 分钟或每 25 个评估写 checkpoint。保存已完成 evaluation_id、配置 hash、数据 hash、代码 hash，重启仅续跑未完成任务。
- CPU 工作者最多 min(4,可用核心数-1)，资源紧张降到 1。只是进程并行，不新增模型 agent；遵守本机其他任务负载。
- 任何未来数据影响历史信号、错误拆股、净值不守恒、负现金/隐含杠杆、验证结果反馈选参的情况立即停止相关计算，先修正并留痕。
- 前瞻模拟本轮仅输出配置和建议的 60 个交易日操作核对方案；不自行创建 cron、自动交易或长期任务。60 日只用于运行核对，不等于收益优势已被证明。

## 10. 交付结构和验收

研究代码建议新目录 research/momentum_cross_asset_v1/（如已有同名目录先检查，不能覆盖）：
- PLAN.md：本规格镜像；config.json：冻结网格/窗口/门槛；
- data_audit.py：数据发现、规范化、指纹和质量；
- engine.py：信号/账户/成交；search.py：GLD 开发排序；validate.py：锁定候选迁移/压力；
- report.py：从底表生成摘要/图；run.py：阶段入口；tests/test_integrity.py：关键失败模式；
- results/<run_id>/：所有本次产物；不改生产模块，可只读复用已核验函数。

必要输出：
1. 00_summary_zh.md：一句话结果、是否有合格策略、GLD/USO/SPY 分别三指标表、基准、问题、唯一下一步。
2. data_audit.json + data_sources.md + contamination_log.md；
3. frozen_config.json + candidates_locked.json（含锁定时刻、代码/data/config hash、候选与扰动集合）；
4. all_trials.csv（180 行全量配置与状态；不得只导出赢家）；
5. dev_metrics.csv、validation_metrics.csv、yearly_metrics.csv、robustness.csv；
6. primary 和 sensitivity 每个资产/场景的 daily_equity.csv、trade_ledger.csv、signals.csv；每条路径唯一 ID；无成交价也须记录未执行原因。
7. benchmark_metrics.csv、bootstrap_summary.json、dsr_report.md、execution_check.md；
8. acceptance.md（每个门槛 PASS/FAIL/INCOMPLETE 和证据文件，不用一句“通过”代替）；
9. reproduce.md（真实环境/命令/依赖版本/代码提交/数据获取及复跑方式）；
10. manifest.json（文件字节数与 SHA256）、STATUS.md、失败时 gap_report.md；
11. 最多 3 张核心图：逐资产净值与回撤、开发参数稳定性、分段三指标差值。参数热图只能用开发数据展示，验证数据不用于继续挑参数。

Drive 写入：
- 在“回测结果”根目录建立 2026-10-21_mom-xasset-001/<run_id>/，日期若执行日变化可按真实启动日期，但研究数据截止仍固定 2026-10-20。
- 本研究文件夹保留计划，并新增 result_link.md 指向结果唯一正本，不复制第二套结果。
- 体积大产物可压缩，另保留可直接阅读的中文摘要、验收、复现说明、manifest。
- 原行情再分发需遵守来源许可；不宜上传原字节时上传来源、hash、获取脚本和本机留存路径，不能声称 Drive 已有原始数据。
- 上传后读回摘要/验收，核验关键数字与底表、链接可读、必要文件存在。
- 成功后更新 START HERE 同一个文件的研究状态/最新 run 链接，再更新 Notion 的研究入口段；不覆盖日常交易、生产或其他项目内容。
- Drive 不可写但计划可读时，可继续本地研究并写 pending_sync.md，最后状态注明“计算完成，归档未完成”；主入口和 Notion 未同步不得声称全流程完成。
- 整个任务真正完成的定义是结果已验证并可取用，或明确 NO_CANDIDATE/NO_PASS/INCOMPLETE 的证据包已交付。找到赢家不是完成的必要条件。

## 11. Hermes 分步执行任务

### Task 1：定位环境、数据和任务身份
Files：PLAN.md、config.json、data_audit.py、tests/test_integrity.py。
Interfaces：audit_data(config:dict)->dict；写 data_audit.json，返回 PASS/FAIL 和规范化数据路径。
- [ ] 读取本计划、START HERE、已有 R2/R3 摘要和两个数据证据报告；确认项目范围与生产边界。
- [ ] 核实 Mac 仓库、工作树、Python；建立隔离研究目录；把本计划 hash 写入 run manifest。
- [ ] 写数据测试：USO 1:8 拆股不创造约 700% 收益；重复日期/缺交易日/混合尺度必须 FAIL；SPY 不重复加分红。
- [ ] 先运行测试确认会检测故意损坏的样例，再实施数据规范化和质量检查。
- [ ] 核实 GLD、USO、SPY 原文件；缺失时按第 2 节补取；保存 2006 起数据和 DGS3MO。
- [ ] 生成数据结论；PASS 才进 Task 2；失败给 gap_report，不买数据。代码本地提交并记录 commit。

### Task 2：构建可审计回测账户
Files：engine.py、tests/test_integrity.py。
Interfaces：simulate(bars,params,scenario,start,end,initial_cash=16000)->{equity,trades,signals,metrics}。
- [ ] 写合成样例测试：t 收盘信号只能 t+1 开盘成交；截断未来数据历史信号不变；通道不包含当前高低点。
- [ ] 写账户测试：买卖现金守恒、费用恰好扣一次、不负现金、不杠杆；跳空按下一开盘成交，不按前收盘或阈值假成交。
- [ ] 写退出优先/冷却期/期末行政结算测试；ATR14 用规定简单均值；成本变动不能改变阈值。
- [ ] 写现金与 Sharpe 测试：现金为零收益时不能生成利息；国债代理严格滞后；周末天数不漏计；全现金零波动 Sharpe 返回 NA。
- [ ] 实现引擎和 BH100/BH50-M/CASH/VT10；常量/单调/拆股合成路径用手算核对。
- [ ] 仅使用开发数据和合成数据排错；测试通过后本地提交。旧 R3 可用于理解，不要求新规则复现旧数值。

### Task 3：GLD 有界开发与锁定
Files：search.py、config.json、results/<run_id>/all_trials.csv。
Interfaces：search(dev_bars,grid,config)->ranked_candidates；freeze(candidates,hashes)->candidates_locked.json。
- [ ] 枚举 180 个唯一 param_id，assert count==180；训练输入末日<=2015-12-31。
- [ ] 实现第 7 节确定性排序并测试同分顺序、差邻居不被跳过、无候选分支。
- [ ] 跑完开发连续段和三个块，保留每组失败状态与成本。
- [ ] 生成最多 3 个候选，固定 primary；登记最多 36 个 ±20% 扰动，不读取验证指标。
- [ ] 保存并校验锁定文件/hash，记录此前历史已被研究的 contamination_log；无候选则直接交付 NO_CANDIDATE。

### Task 4：一次性跨时间/资产验证
Files：validate.py、results/<run_id>/validation_metrics.csv 等。
Interfaces：validate(locked,all_bars,config)->evaluation_table；任何参数不在锁定集则拒绝。
- [ ] 测试锁定后任何参数修改会触发 hash 失败；验证函数不能返回修改后的候选给 search。
- [ ] 按计划跑 GLD/USO/SPY 后期连续段、三个分段及规定场景/扰动，记录失败资产，不择优隐去。
- [ ] 输出 paired bootstrap、DSR 和预算实际使用；必要统计缺失如实标明。
- [ ] primary 达历史门槛则做原始价格/公司行动整股账本复核；缺数据不购买，输出 EXECUTION_NOT_VERIFIED。
- [ ] 分类 NO_PASS/PAIR_PASS/BROAD_PASS/INCOMPLETE/PAPER_READY；不能在评估后升级第二候选。

### Task 5：报告、归档和交接
Files：report.py、acceptance.md、reproduce.md、manifest.json。
Interfaces：render_report(result_tables,config)->artifact_paths；verify_delivery(paths)->acceptance_record。
- [ ] 直接从底表生成摘要与图，复核 CAGR/MDD/Sharpe 和同口径基准，逐条填写验收。
- [ ] 使用已保存配置重跑一条 primary 路径，逐日净值误差<=1e-8 美元、信号与交易日期完全相同（允许输出文本舍入，不放宽业务差异）。
- [ ] 归档 Drive 并读回；回写 START HERE 和 Notion；无法同步则给本地路径与 pending_sync。
- [ ] 向 Jake 回复：是否找到合格策略、三资产各自结果、实际尝试数、数据是否复用/补齐、任何成本支出（应为0新增采购）、主要限制、结果链接和下一步。不要只回复“训练完成”。

## Review Focus（已对应任务测试）
1. USO 反向拆股/调整与原始价混用 → Task 1 数据损坏样例及 Task 4 原始账本复核。
2. 当前 K 线/未来收益泄漏入场或排序 → Task 2 截断等价、Task 3 输入末日、Task 4 hash 锁定。
3. 跳空/现金不足/成本重复/负现金 → Task 2 合成账户测试。
4. SPY 分红与国债利率重复或提前到账 → Task 1 与 Task 2 现金测试、Task 4 账本。
5. 选择最优验证资产或改 primary、失败产物丢失 → Task 3 确定性排名、Task 4 参数白名单、Task 5 全量验收。

## 12. 既有研究参考和事实边界

MOM-GLD-001 R2：
https://drive.google.com/file/d/17bv55F8Oi8lNdTnJiPn_Qd2WbfQ8veIW/view
R3：
https://drive.google.com/file/d/19VNAS5CM-r1tSdTViLRQI2dJhDzWdPEQ/view
旧 Mac 复现路径：
https://drive.google.com/file/d/1UyIwXeD1wMxuqxiLQv9BdzEr2i5LDuAV/view

旧结果只说明其特定 GLD 规则的证据不足。此轮是新的有限候选比较，不是“无限调到赢”为止。
USO 免费历史页面存在：https://finance.yahoo.com/quote/USO/history/
这不保证当前环境批量下载无障碍。优先现有缓存，遇到限制如实报告。
USO 官方产品与期货换月说明：https://www.uscfinvestments.com/uso
上述外部来源于 2026-10-21 检索；本计划没有宣称重新验证过整个行情数据库。

## 13. 启动 prompt（配合此计划链接使用）

你是 MacBook 本地 Hermes，本任务执行负责人。Jake 已授权本轮跨资产动量研究，请读取本计划和 START HERE 后直接实施，不要停在复述方案或重新询问是否开始。
工作范围是 MOM-XASSET-001 v1.0：先核实已有 GLD/USO/SPY 日线，免费补缺；在 GLD 2007–2015 搜索 180 组，冻结主候选和扰动后，跑 GLD/USO/SPY 2016–2026-10-20 的迁移验证。按文档口径处理成本、现金、拆股、分红与样本污染。
先给简短接手回执并继续执行。无需逐参数询问。严守预算，不购买数据，不改生产，不下单，不以验证成绩反向调参。
完成后归档结果正本并读回验证，更新 START HERE 与 Notion。没有合格候选也交完整证据包。只有数据、权限或不可恢复技术问题阻塞时报告具体缺口；不要把执行尝试写成完成。

## 14. 设计自查
- 范围：多资产验证、单资产独立账户，与本会话已选方向一致。
- 决策：网格、时段、排序、primary、门槛、预算与失败路线均已固定。
- 数据：已有证据与本机待核验明确分开；拆股与旧“无公司行动”说法的冲突已列出。
- 报告：通过计算、通过经济目标、统计证据、可执行性、前瞻/实盘批准均分开。
- 未执行：本文件不包含编造的回测结果，未远程启动 Hermes。

