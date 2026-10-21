# QM-RESEARCH-001｜关键经济行为自检（test_evidence）

执行：`python tests/test_engine.py` → **13/13 PASS**（2026-10-22）
性质：合成数据单元测试，验证账户内核的会计与经济行为；**不验证**与原方案盘中规则的等价性（无分钟数据，无法验证）。

| # | 检查 | 结果 | 断言要点 |
|---|---|---|---|
| 1 | planned_loss_within_budget | PASS | 计划损失（含止损滑点与双边佣金）不超过 0.5% 权益 |
| 2 | gap_through_stop_fills_at_open | PASS | 跳空穿过止损时按开盘价成交，不按止损价理想成交 |
| 3 | single_position_and_ranking | PASS | 同日多信号只开 1 仓，按排序键取第一名 |
| 4 | liquidity_cap_no_zero_share | PASS | 流动性上限生效时不产生 0 股或负股 |
| 5 | long_cash_never_negative | PASS | 多头现金不为负；建仓金额不超权益 |
| 6 | short_proceeds_do_not_add_power | PASS | 卖空所得不增加购买力，初始名义 ≤30% |
| 7 | stop_priority_over_target | PASS | 同 bar 同时触及止损与目标时按止损（不利事件优先） |
| 8 | commission_minimum_1usd | PASS | 每笔最低 1 美元佣金生效 |
| 9 | final_liquidation_once | PASS | 期末只清算一次，收盘后持股归零 |
| 10 | future_prices_do_not_retroact | PASS | 改动未来价格不改变已有入场的价格与日期 |
| 11 | split_no_fake_return | PASS | 拆股调整后不产生假收益 |
| 12 | segment_reset_flat | PASS | 独立分段从空仓、初始权益开始 |
| 13 | metrics_no_nan | PASS | 无交易时指标不崩溃、交易统计返回 0 笔 |

## 已知未覆盖

- 分钟级 OR5、09:35 可见性、下一分钟开盘成交、5% 分钟量上限：**无数据，未测**。
- 停牌恢复成交、退市最后价格、借券召回：**无数据，未测**。
- 同日多股票“同一分钟”竞争：EOD 代理只能在日级排序，未测分钟竞争。
- 权重压降与部分止盈的交互：只在合成用例中覆盖了部分止盈，未构造同日全平+止盈+压降的三重冲突用例。

结论：内核会计行为可复现、无前视；但**执行证据等级低**，C 层维持 EXECUTION_INCOMPLETE。
