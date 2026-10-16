# A股战法12 每日纸面订单表

纸面交易：不连接券商、不下单。每个交易日生成一张订单表（默认发到邮箱）。规则与 `research/ashare_tactics` 回测一致，本目录不改策略参数：

- 初始资金 10 万元，最多 10 只、等权（单只目标为权益的 10%）
- T+1；买入按 100 股整数倍
- 退出：收盘跌破 MA20 则下一交易日开盘卖出，或持有满 40 个交易日
- 费用沿用回测：佣金万 2.5（最低 5 元）、卖出印花税 0.05%、滑点 0.1%；剔除 ST

状态、订单表和日线缓存只写在 `STATE_DIR` / `CACHE_DIR`（GitHub Actions 上是工作区里的 `.t12/`，已加入 `.gitignore`）。**不要把订单表、持仓或 `state.json` 提交进仓库。** 本仓库是公开的。

## 在 GitHub Actions 上跑（Jake）

工作流文件：

- `.github/workflows/ashare-tactic12-daily.yml`：每个交易日的订单表。备用 cron 为 UTC 00:30（北京时间 08:30），周一到周五，超时 60 分钟。
- `.github/workflows/ashare-tactic12-build-cache.yml`：一次性建全历史缓存，手动触发，超时 6 小时。与每日任务共用同一份缓存。

两个工作流都限制在 `jakeguo117/Vibe-Trading`。GitHub 托管 runner 的 IP 在美国，东财/新浪会被限流，所以抓取带最小间隔、指数退避，并在东财失败后改用新浪、再失败改用腾讯。每只股票写入一个 parquet；超时或失败也会把已经下好的缓存存进 `actions/cache`，下次接着补。

### 1. 仓库 Secrets

在 GitHub 仓库 **Settings → Secrets and variables → Actions → New repository secret** 添加：

| Name | Value |
| --- | --- |
| `GMAIL_APP_PASSWORD` | Gmail 应用专用密码（16 位，不要用登录密码） |
| `GMAIL_USER` | `jakeguo117@gmail.com` |
| `MAIL_TO` | `394592848@qq.com` |

应用专用密码在 Google 账号的两步验证里生成。Actions 日志不会打印这三项；演练（`dry_run`）不发信。

### 2. 打开 Actions

Fork 默认可能关掉 Actions。打开 **Settings → Actions → General → Actions permissions**，选 Allow，并保存。然后到 **Actions** 页启用本仓库的工作流。

第一次全市场约 5400 只、从 2015 年起的日线，在美国 IP 上不一定能在 60 分钟内下完。建议先手动跑 **ashare-tactic12-build-cache**，跑完（或失败但缓存已保存）再让每日任务接手。覆盖率不足时当日不会出单，避免发错单。

### 3. 隐私

订单表里有纸面持仓和买卖数量。GitHub 公开仓库的 Actions 日志、工件（工件保留 7 天）对能看仓库的人可见。更稳妥的做法是把这份 fork **改成私有仓库**，再打开 Actions。不要把 `.t12/`、订单表或密码文件 `git add` 进来。

### 4. 准时触发（cron 会晚）

GitHub 的 `schedule` 在高峰时可能晚几个小时才开始，不能当成北京时间 08:30 的闹钟。要在北京时间 08:30 左右准时跑，用仓库外的定时器调用 `workflow_dispatch`（每日工作流的 `dry_run` 留空或 false 才会发信）。

需要一个 **fine-grained PAT**，权限只给这个仓库 `Vibe-Trading`，Actions 为 Read and write。不要用 classic PAT，也不要给其他仓库权限。调用示例（把令牌放在定时器那台机器上，不要写进仓库）：

```bash
curl -sS -X POST \
  -H "Authorization: Bearer $GH_PAT" \
  -H "Accept: application/vnd.github+json" \
  https://api.github.com/repos/jakeguo117/Vibe-Trading/actions/workflows/ashare-tactic12-daily.yml/dispatches \
  -d '{"ref":"research/tactic12-daily","inputs":{"dry_run":false}}'
```

`ref` 用工作流所在的分支。合并前可把 `dry_run` 设为 true 做演练。

## 本地演练

不发邮件，状态和缓存放在临时目录：

```bash
STATE_DIR=/tmp/t12-state CACHE_DIR=/tmp/t12-cache DRY_RUN=1 \
  ./run_daily.sh --dry-run
```
