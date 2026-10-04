# What to add to gold-ai-trader — ideas from 1,689 GitHub repos

_Reviewed 2026-10-04. Source list: `all_repos.csv`. Starting point: BTC H1 `momentum_rsi_mtf` is live
at 0.01 lot, XAU is on paper after failing its forward test, and the validation pipeline (walk-forward,
cost stress, parameter robustness, Monte Carlo, live gate) is the strongest part of the project._

## The one rule that comes out of the reading

Across the repos, the LLM trading systems that publish real records point the same way:

- **DXRG's six-month record of 600 live LLM agents** (`ProjectDXAI/continuous-record-llm-trading-agents`):
  their sizing ignored volatility (5× leverage whatever the regime), they gave back almost all open profit
  (49 % of trades that reached +3 % still closed negative), and **a fixed 2 %/4 % bracket beat every
  discretionary LLM exit**. Behaviour followed the risk controls far more than the strategy text.
- **TradeTrap** (`Yanlewen/TradeTrap`): small prompt injections, poisoned news feeds or tampered memory
  flip LLM agents into panic selling. Any text the agent reads is an attack surface.
- **RakshaQuant** (`HimanshuMohanty-Git24/RakshaQuant`): the honest design. **The AI can only veto**,
  never size or add orders. Every veto gets a counterfactual trade, and the AI must beat the no-AI book
  *after its API cost* in a pre-registered test.
- Your own Kronos filter experiment (`backtest/kronos_filter_experiment.py`) already found no benefit.

**So: the LLM never generates signals and never touches order placement.** The frozen rule trades.
LLMs earn their place in research, operations, review and veto, each one measured by the pipeline you
already have.

---

## A. LLM / agent ideas, ranked by fit

### 1. Research agent with an autoresearch loop ⭐ best fit
*From:* `chrisworsey55/atlas-gic` (prompt changes kept or reverted by git commit), `ZhuLinsen/alphaevo`,
`CamusGIT/EvoQuant`, `tarsyang/quantevolve` (an OpenEvolve fork), `Miasyster/QuantGPT`, `OnePunchMonk/AgentQuant`.

An agent proposes **one** strategy or feature change at a time, writes it to `strategy/btc_strategies.py`,
runs `btc_h1_validation.py` + `xau_h1_validation.py`, and keeps the change only if it passes **the
existing live gate on both instruments**. Everything else is reverted and recorded.
- Your pipeline is the judge. That is what these repos lack: most of them score on backtest Sharpe and overfit.
- Add a **trial counter** and deflated-Sharpe correction (B7), because 200 agent attempts will find
  something by luck.
- Keep a `research/negative_results.md` log so the agent doesn't retry dead ideas (M5 has no edge;
  mean reversion fails; the Kronos filter adds nothing).
- Build it with Claude Code headless or the Agent SDK (it needs bash and file editing), run weekly.

### 2. Veto advisor run as a paired-book experiment (not a filter you switch on)
*From:* RakshaQuant (paired books, counterfactuals, pre-registration), `0xethanq/astra-quant-agent`
(LLM analysis + deterministic Python risk), TradingAgents' Bull/Bear/Risk debate.

Rewrite `claude/analyzer.py` as a **veto-only** advisor and run it **in shadow** next to the live BTC trader:
- Book A = live, unfiltered. Book B = the same signals minus the ones Claude vetoes (paper only).
- Input: the signal, ATR, regime features, the economic calendar (B1), recent results. No free-text news
  at first (TradeTrap risk).
- Decide the pass rule before starting: e.g. after ≥ 60 signals, B's expectancy beats A's by
  ≥ 0.05 R **after API cost**, and veto precision > 50 %. Otherwise delete it.
- Expect it to fail, as Kronos did. The value is getting a clear answer cheaply.

### 3. Read-only MCP server over the project
*From:* `ariadng/metatrader-mcp-server`, `alpacahq/alpaca-mcp-server`, `kukapay/freqtrade-mcp`,
`QuantConnect/mcp-server`, `LuxAlgo/edge-stats`.

Expose `reports/btc_live_status.json`, `logs/btc_live.log`, trade history, validation reports and
backtest queries as **read-only** MCP tools. Then you can ask Claude "why no trade since Tuesday?" or
"how are this week's trades doing against the backtest distribution?" from your phone.
- **No order tools.** `metatrader-mcp-server` lets the LLM place orders — don't copy that part.
  Order placement stays in `execution/live.py` behind the kill switch.

### 4. Daily ops / watchdog agent
*From:* `TNT-Likely/PanWatch`, `JingHao-Leon/dsh-alpha-desk` (cron + risk gate + memory review),
`SilentFleetKK/riskguard`, `6alaile/MetaTrader5-to-Telegram`.

A scheduled job reads the logs and status JSON and sends a Telegram message: trades opened/closed,
drawdown against the floor, MT5 disconnected, Algo Trading button off, missed hourly bar, spread above
backtest assumption. Most of it is plain Python. Use Claude only to summarise unusual cases
("3 errors in the log — here is what they mean").

### 5. Trade journal with memory, plus a weekly review
*From:* `mnemox-ai/tradememory-protocol` (losing-pattern recall, tamper-evident audit),
`hugodemenez/deltalytix`, `pipiku915/FinMem-LLM-StockTrading`.

Store each closed trade with its features. Every week Claude writes a review: actual vs expected
R-distribution, streaks against the Monte Carlo bands, whether the slow decline in edge
(+0.41 R in 2014 → +0.13 R in 2025) is continuing. This is **descriptive only** — it feeds decisions
you make, never the bot.

### 6. Paper-to-strategy agent
*From:* `SL-Mar/quantcoder` (research paper → QuantConnect code), `adam-s/alphadidactic`.
Feed it momentum papers (C1) and have it produce candidates for idea #1's queue.

### Skip
- **"AI hedge fund" persona swarms** (`AutoHedge`, `swarm-trader`, ai-hedge-fund clones, Buffett/Munger
  agents): built for equity stock-picking, untested for an H1 rule, expensive per decision.
- **LLM signal generation of any kind** (nof1 clones, `Hyper-Alpha-Arena`, `LLM_trader`, vision chart
  readers): DXRG's record says no.
- **Social sentiment bots** (Elon/Trump tweet bots, LunarCrush): unproven and easily manipulated.

### Which Claude models (checked against current Anthropic docs)
| Use | Model | Price per 1M tokens (in / out) |
|---|---|---|
| Research agent (#1), paper-to-code (#6) | `claude-opus-5-5` (adaptive thinking, effort `high`) | $4 / $20 |
| Veto advisor (#2), weekly review (#5) | `claude-opus-5-5` at effort `low`–`medium`; try `claude-sonnet-5-5` ($2 / $10) or `claude-haiku-4-5` ($1 / $5) only if the cost-adjusted test needs it cheaper | — |
| Any JSON reply | Use **structured outputs** (`output_config.format` / `client.messages.parse()`) instead of prompt-and-`json.loads` | — |

`claude/analyzer.py` still uses `claude-sonnet-4-5` with free-text JSON parsing. It needs a model update
and structured outputs before it is used for anything.

---

## B. Non-LLM additions (probably worth more than the agents)

| # | Add | From | Why it matters here |
|---|---|---|---|
| B1 | **Economic calendar in `news/filter.py`** (it currently returns `[]`) | `fizahkhalid/forex_factory_calendar_news_scraper`, `EarnForex/News-Trader` | CPI/NFP/FOMC moves gold more than anything else. Backtest "skip entries ±N min around high-impact USD events" through the pipeline before using it live. |
| B2 | **MFE/MAE exit study** on the 4,330 Bitstamp + 1,257 Valetax trades | DXRG capture-gap finding, `EarnForex/Trailing-Stop-on-Profit` | How much open profit is given back before the 2 ATR stop is hit? Test breakeven-at-1R or a partial close at 1.5R as **frozen** variants through the full gate. Your 2/3 ATR bracket may already be near optimal (DXRG found fixed brackets win) — this confirms it either way. |
| B3 | **Regime detector as a research label, not a filter** | `QuhiQuhihi/regime_model`, HMM in `quantium-ai/research`, `rafa-rod/pytrendseries` | Label each trade bull/bear/chop and check where the edge lives. It matters because the edge is shrinking and BTC hasn't had a long bear phase in your Valetax data. |
| B4 | **Telegram alerts + equity guard** | `6alaile/MetaTrader5-to-Telegram`, `EarnForex/Account-Protector`, `codedpro/mt5-trade-split-manager` | You have the kill switch and drawdown floor. Add a push when either fires, plus a daily heartbeat so silence means something is broken. |
| B5 | **Longer XAU history from Dukascopy ticks** | `Leo4815162342/dukascopy-node` | XAU failed its 4-week forward test (31 trades, −0.14 R). Testing the same frozen rule on 2003–2014 gold (never seen) tells you whether that failure was noise or decay — the same test that made BTC credible. |
| B6 | **Same frozen rule on more instruments** (ETH, NAS100, EURUSD, oil on Valetax) | `kieranjwood/trading-momentum-transformer`, `x-trend` papers | Edge on unrelated markets was your strongest evidence. More uncorrelated markets also means more trades per month with the same 0.01-lot risk. |
| B7 | **Deflated Sharpe, PBO, purged/combinatorial CV** | `hudson-and-thames/mlfinlab`, `baobach/mlfinpy`, `WenjieZ/TSCV`, `quantscious/finmlkit` | Adds a multiple-testing check to the live gate. Required before running idea #1. |
| B8 | **QuantStats tear sheet from live trades** | `ranaroussi/quantstats` | One-line HTML report for live vs backtest; feeds idea #5. |
| B9 | **Conditional edge queries** ("expectancy when ATR is in the top decile and it's the Asian session") | `LuxAlgo/edge-stats` | Each answer comes with a sample size and confidence interval, so you don't fool yourself. Fast way to find out where the momentum edge really sits. |
| B10 | **Headless MT5 on the Oracle VM** | `gmag11/MetaTrader5-Docker`, `ejtraderLabs/Metatrader5-Docker` | Removes "home PC asleep / Windows Update" as a failure mode (see GO_LIVE.md and ORACLE_DEPLOY.md). |

### C. Reading for the momentum edge
1. `kieranjwood/slow-momentum-fast-reversion` — momentum with change-point detection to cut losses at turns.
2. `kieranjwood/trading-momentum-transformer` — attention-based momentum with regime awareness.
3. `kieranjwood/x-trend` — few-shot trend following that carries over to new markets (relevant to B6).
4. `stefan-jansen/machine-learning-for-trading` (3rd ed.) and `hudson-and-thames/mlfinlab` — triple-barrier labels, meta-labelling. Meta-labelling ("should I take this signal?") is the principled version of idea #2.

---

## Watch out for these repos
- **Fake "2026 MT5 …" repos**: `PropGuard-Trailing-Equity-Armor`, `Risk-Nexus-Command`,
  `MT5-Exposure-Mesh-Analyzer`, `Matrix-Confluence-Scorer`, `mt5-quiet-automata` ("run MT5 as stealth
  service") and many similar. Keyword-stuffed descriptions, ~120 stars and 0 forks each; the five
  checked were all created within about an hour of each other on 2026-06-28 from different accounts.
  This is how malware gets spread on GitHub. **Never download their releases.**
- **Solana sniper/MEV/copy-trading bots** and **repos whose description is just the name repeated**
  (`polymarket trading bot polymarket trading bot…`): same pattern, often wallet-key stealers.
- **`openbq-org/OpenBB`**: the official project is `OpenBB-finance/OpenBB`. Check before cloning.
- Repos claiming fixed returns ("72 % ROI", "$150k+ trader") without a live record.

## Added after reviewing tradesdontlie (2026-10-04)

Full read of `tradesdontlie/tradingview-mcp` @ `c05b8f5` plus `prop-firm-monte-carlo`. Not malware;
clean code, 2 dependencies, no third-party network calls. What carries over:

1. **Regime-split, price-path Monte Carlo** (from `prop-firm-monte-carlo`). Today
   `backtest/monte_carlo*.py` resamples finished trade R. Instead, bucket each trade by ATR-percentile
   regime (low ≤25 / normal / high 75–90 / extreme >90), record its MFE/MAE path, and simulate by
   drawing regime sequences and paths. That answers "what if the next 6 months are mostly high-vol?"
   and gives B2 (exit study) its data. Uses the 4,330 Bitstamp + 1,257 Valetax trades already on disk.
2. **MCP design lessons for A3** (our read-only project MCP): compact output by default with `verbose`
   opt-in, a decision tree in the server `instructions` field, a `health_check` tool called first, and
   capped payloads. They found 84 small tools worked fine with good instructions. **Don't copy**:
   `ui_evaluate`-style "run anything" tools, `tools: "*"` on sub-agents, or anything that can reach
   orders.
3. **Optional: TradingView as a chart-viewing tool, never in the trading path.** Generate a Pine
   indicator from `logs/btc_live.log` / backtest trades (entries, exits, SL/TP as `label.new`/`line.new`)
   and look at the bot's trades on a TradingView chart, or prototype a strategy in Pine before porting
   it into the validation pipeline. Plain copy-paste into the Pine editor is enough; tradingview-mcp only
   automates it. If you install it: block `ui_evaluate`, connect no broker in TradingView, don't
   `pine_check` private strategies (it uploads source as Guest), launch TradingView yourself
   (`tv_launch` force-kills it), and close port 9222 afterwards.
4. **Not useful**: TradingView data for research (500-bar / 20-trade caps), its alerts (Telegram from
   our own logs is simpler and doesn't need TradingView open), replay practice.

## Combined plan

| Phase | Do | Built from | Effort | Pays off because |
|---|---|---|---|---|
| **0. Check the live verdict** | Add swap to the cost model and re-run cost sensitivity; write the retirement rule and scaling rule; add `.claude/settings.json` deny rules (see "Gaps" below) | own code + MT5 `symbol_info` | 1 day | The live decision was made without swap costs or a stop rule |
| **1. Protect the live bot** | B4 Telegram alerts + daily heartbeat; B1 economic calendar in `news/filter.py` (backtest it first) | EarnForex, ForexFactory scraper | 1–2 days | BTC is live with real money now; silent failures and CPI/FOMC are the biggest near-term risks |
| **2. Know the edge better** | Regime-split price-path Monte Carlo (item 1 above) → B2 exit study → B3 regime labels | prop-firm-monte-carlo, DXRG capture gap, regime_model | 3–5 days | The edge is shrinking (+0.41 R → +0.13 R); this shows which regimes still pay and whether the 2/3 ATR bracket is right |
| **3. Strengthen or kill strategies** | B5 Dukascopy XAU 2003–2014; B6 same frozen rule on ETH/NAS100/EURUSD/oil; B7 deflated Sharpe in the live gate | dukascopy-node, momentum papers, mlfinlab | 1 week | Cross-market evidence is the project's strongest proof; XAU's failed forward test needs an answer |
| **4. Claude becomes useful (no trading risk)** | A3 read-only project MCP (using the design lessons above); A5 weekly review; optional TradingView trade-overlay Pine | tradingview-mcp design, tradememory, quantstats | 3–4 days | Ask "how's the bot doing vs backtest?" from anywhere; see trades on a chart |
| **5. Automated research** | A1 research agent judged by the existing live gate (needs B7 first) | atlas-gic, alphaevo, quantevolve | 1–2 weeks | Turns the validation pipeline into a strategy-search engine without overfitting |
| **6. Only if still curious** | A2 Claude veto advisor in shadow, pre-registered pass rule | RakshaQuant | ongoing | Kronos already failed; this is a cheap, clear yes/no on LLM vetoes |

**Never**: LLM signal generation, LLM order placement, or any tool (MCP, TradingView, agent) that can
reach `execution/live.py` or a broker.

## Gaps found on review (2026-10-04)

**In the project (more important than anything above):**
1. **Swap/financing is not in the backtest cost model.** Costs are spread $29.76 + slippage only.
   `reports/btc_broker_spec.json` doesn't record `swap_long`/`swap_short`, and the 4-week forward test
   was paper, so it never paid swap. Live P&L does include it (`execution/live.py:182`). Avg hold is
   14.3 h (max 96 h), so most trades cross a rollover, and weekends are usually charged triple. Pull
   `symbol_info("BTCUSD.vx").swap_long/swap_short/swap_mode` and re-run cost sensitivity with it. **Do
   this first; it could change the live verdict.**
2. **No rule for retiring the strategy.** The only stop is the equity floor (20 % of start). The edge has
   shrunk from +0.41 R to +0.13 R; decide now, in writing, when to stop (e.g. live expectancy after N
   trades falls below the backtest 5th percentile, or a CUSUM/SPRT test against the backtest
   distribution fires). Without it you'll decide under drawdown stress.
3. **No live-vs-backtest execution tracking.** Log expected entry/exit vs actual fill for every trade
   (slippage, spread at fill, swap) and compare against the $31.76 assumption weekly.
4. **No portfolio risk cap.** Phase 3 adds instruments; BTC, NAS100 and gold momentum can line up. Cap
   total open risk across symbols before adding any.
5. **Scaling rule.** When does 0.01 lot become 0.02? Write the equity thresholds down now (GO_LIVE.md
   gives $300 → 2–4.4 % per trade, already above the usual 1 %).
6. **Guard the live machine from agents.** No `.claude/settings.json` exists. Before any agent work
   (Phases 4–5), deny reads/edits of `.env`, `state/`, `execution/live.py` and `LIVE_TRADING` changes,
   so no agent can flip the bot live or touch credentials.
7. **Weekend/gap handling for BTC CFD** isn't explicit in `strategy/btc_h1_signal.py` / `execution/live.py`:
   check Valetax's weekend spread and whether SL can gap through.

**In the research itself:**
- **The scrape missed major repos**: `nautilus_trader`, `virattt/ai-hedge-fund` (~59k★), `shiyu-coder/Kronos`,
  `AI4Finance-Foundation/FinRobot`, `google-research/timesfm`, `amazon-science/chronos-forecasting`,
  the official `OpenBB-finance/OpenBB`. Unauthenticated search with rate-limit retries left holes.
  (I also recommended nautilus_trader earlier although it wasn't in the CSV — it was from memory.)
  Fix: `gh auth login`, then re-run with more queries.
- **Sources outside GitHub not covered**: MQL5.com Code Base / Market (the biggest MT5 EA ecosystem),
  arXiv/SSRN momentum and crypto papers, Hugging Face time-series models (TimesFM, Chronos, Moirai —
  same caution as Kronos: test as a filter through the pipeline, expect nothing), r/algotrading,
  QuantConnect community strategies.
- **Paid platforms** (3Commas, Cryptohopper, Gainium, TradingView Premium strategy tester) not compared.

## Suggested order (original)
1. **B1** economic calendar → **B4** alerts → **B2** exit study (cheap, directly useful, no LLM risk).
2. **B5** Dukascopy XAU history and **B6** more instruments (strengthen or kill the edge).
3. **A3** read-only MCP + **A5** weekly review (Claude becomes useful with no trading risk).
4. **B7** multiple-testing correction, then **A1** research agent.
5. **A2** veto advisor in shadow, with a pre-registered pass rule — only if you still want to know.
