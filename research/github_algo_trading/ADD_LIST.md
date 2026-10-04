# Everything we can add to gold-ai-trader

_One list, built from 1,952 GitHub repos (`all_repos.csv`), the tradesdontlie review, and a check of
the project itself. Reasoning and evidence: `IDEAS.md`. Status: ☐ not started, ✅ built 2026-10-04 (see RULES.md)._

**Hard rule:** no LLM generates signals or places orders. Nothing below may reach `execution/live.py`
or the broker except the existing frozen strategy.

## Phase 0 — Check the live decision (do first)
| ☐ | Add | Why | Use |
|---|---|---|---|
| ✅ | **Swap/financing in the cost model**, re-run BTC cost sensitivity | Backtest has no swap; live P&L pays it; trades average 14 h, so most are held past the daily rollover | MT5 `symbol_info().swap_long/swap_short/swap_mode` |
| ✅ | **Written retirement rule** (when to switch the strategy off) | Edge fell +0.41 R → +0.13 R; only stop today is the 20 % equity floor | CUSUM/SPRT vs backtest distribution |
| ✅ | **Written scaling rule** (equity → lot size) | $300 already means 2–4.4 % risk per trade | `risk/sizing.py` |
| ✅ | **`.claude/settings.json` deny rules** for `.env`, `state/`, `execution/live.py`, `LIVE_TRADING` | Must exist before any agent work | Claude Code permissions |
| ☐ | **Weekend gap / spread check** for BTCUSD.vx | Stop-loss can gap through; weekend spread unknown | live logs + broker spec |

## Phase 1 — Protect the live bot
| ☐ | Add | Use |
|---|---|---|
| ✅ | **Telegram alerts + daily heartbeat** (trade open/close, drawdown, MT5 down, Algo Trading off, missed bar) | `6alaile/MetaTrader5-to-Telegram`, `sholafalana/MT5-MT4-Telegram-API-Bot` (pattern only) |
| ✅ | **Economic calendar in `news/filter.py`** (CPI/NFP/FOMC block, backtested first) | `lcsrodriguez/ecocal` (Python), `fizahkhalid/forex_factory_calendar_news_scraper`, `andrevlima/economic-calendar-api`, `freenetwork/investing.com.economic-calendar` |
| ☐ | **Fill-quality log**: expected vs actual price, spread at fill, swap — weekly vs $31.76 assumption | own code (`execution/live.py` already reads deal swap) |
| ✅ | **Equity guard push** when kill switch / drawdown floor fires | `EarnForex/Account-Protector` (pattern) |

## Phase 2 — Understand the edge
| ☐ | Add | Use |
|---|---|---|
| ☐ | **Regime-split, price-path Monte Carlo** (ATR-percentile regimes, MFE/MAE paths) | `tradesdontlie/prop-firm-monte-carlo` (method) |
| ☐ | **Exit study**: open profit given back; test breakeven/partial-close as frozen variants | DXRG capture-gap finding, `EarnForex/Trailing-Stop-on-Profit` |
| ☐ | **Regime labels** per trade (bull/bear/chop, vol regime) | `QuhiQuhihi/regime_model`, `jasonstrimpel/volatility-trading` (Yang-Zhang, Garman-Klass estimators), `chibui191/bitcoin_volatility_forecasting` (GARCH) |
| ☐ | **Conditional edge queries** with sample size + CI | `LuxAlgo/edge-stats` |
| ☐ | **Live tear sheet** vs backtest | `ranaroussi/quantstats` |

## Phase 3 — Strengthen or drop strategies
| ☐ | Add | Use |
|---|---|---|
| ☐ | **Older gold history** (2003/2009–2014, never seen) for XAU | `philipperemy/FX-1-Minute-Data` (HistData 1-min XAUUSD), `Leo4815162342/dukascopy-node` |
| ☐ | **More BTC/crypto history & venues** | `ccxt/ccxt`, `tardis-dev/tardis-node` |
| ✅ | **Same frozen rule on ETH, NAS100, EURUSD, oil** | Valetax symbols; papers: `rkohli3/TSMOM` (Moskowitz-Ooi-Pedersen replication), `maxlamberti/time-series-momentum`, `kieranjwood/slow-momentum-fast-reversion`, `kieranjwood/x-trend`, `jshellen/CTA-strategies` |
| ✅ | **Portfolio risk cap** across instruments before adding any | `2023ai/ftmo-risk-control` (fail-closed design, MT5 adapter — pattern only) |
| ✅ | **Multiple-testing correction** (deflated Sharpe, PBO, purged CV) in the live gate | `hudson-and-thames/mlfinlab`, `baobach/mlfinpy`, `WenjieZ/TSCV`, `quantscious/finmlkit` |

## Phase 4 — Put Claude to work (no trading risk)
| ☐ | Add | Use |
|---|---|---|
| ✅ | **Read-only project MCP** (status, logs, reports, backtest queries; no order tools) | design lessons from `tradesdontlie/tradingview-mcp`; MT5 read examples in `Qoyyuum/mcp-metatrader5-server`, `ariadng/metatrader-mcp-server` (strip their order tools) |
| ☐ | **Weekly trade review** (Claude, structured outputs, descriptive only) | `mnemox-ai/tradememory-protocol`, `Eleven-Trading/TradeNote`, `GeneBO98/tradetally` |
| ☐ | **Update `claude/analyzer.py`**: `claude-sonnet-4-5` → `claude-opus-5-5`, structured outputs | Anthropic SDK |
| ☐ | Optional: **TradingView trade overlay** (Pine script generated from trade logs) | `tradesdontlie/tradingview-mcp` (locked down) or copy-paste |

## Phase 5 — Automated research
| ☐ | Add | Use |
|---|---|---|
| ✅ | **Research agent**: one change at a time, kept only if it passes the live gate on BTC + XAU (needs Phase 3 multiple-testing first) | `chrisworsey55/atlas-gic`, `ZhuLinsen/alphaevo`, `tarsyang/quantevolve`, `SL-Mar/quantcoder` |
| ✅ | **Negative-results log** so it doesn't retry dead ideas | own file |

## Phase 6 — Optional experiments
| ☐ | Add | Use |
|---|---|---|
| ☐ | **Claude veto advisor** in shadow, pre-registered pass rule, net of API cost | `HimanshuMohanty-Git24/RakshaQuant` |
| ☐ | **One batch test of time-series foundation models** as a filter (expect nothing — Kronos already failed) | `google-research/timesfm`, `amazon-science/chronos-forecasting`, `Time-MoE/Time-MoE`, `moment-timeseries-foundation-model/moment`, `thuml/Sundial`, `ibm-granite/granite-tsfm` |

## Infrastructure
| ☐ | Add | Use |
|---|---|---|
| ☐ | **MT5 headless on the Oracle VM** (no home-PC sleep/updates) | `gmag11/MetaTrader5-Docker`, `ejtraderLabs/Metatrader5-Docker`, `lucas-campagna/mt5linux` |
| ☐ | Commit `research/` to the branch | git |

## Second tier (reasonable, lower priority)
| ☐ | Add | Why | Use |
|---|---|---|---|
| ☐ | **Prop-firm route** around the $100 capital problem: simulate challenge pass probability for the frozen BTC/XAU rule | H1 stops need ~$1,000+ for 1 % risk; a funded account solves sizing | `tradesdontlie/prop-firm-monte-carlo`, `2023ai/ftmo-risk-control` |
| ☐ | **BTC derivatives features** (funding rate, open interest, liquidations) tested through the pipeline | Crowding/leverage often precedes momentum failure | `ccxt` funding history, `tardis-dev/tardis-node`, `CryptoGnome/LickHunterPRO` (idea) |
| ☐ | **Gold macro features** (DXY, US real yields, CFTC COT positioning) | Main drivers of gold; may explain XAU forward-test failure | FRED, `rsvp/fecon235`, `macrosynergy/macrosynergy` |
| ☐ | **Session / hour / weekday breakdown** of the edge | Cheap; may show hours to skip | `LuxAlgo/edge-stats` |
| ☐ | **Ensemble with the other positive strategies** (donchian_breakout PF 1.18, ema_trend 1.09) | Several weak, partly uncorrelated edges can smooth equity | own `btc_strategies.py` |
| ☐ | **Second, uncorrelated strategy**: gold/silver or BTC/ETH pairs | Momentum-only portfolio has one failure mode | `KidQuant/Pairs-Trading-With-Python`, `bradleyboyuyang/Statistical-Arbitrage` |
| ☐ | **Block-bootstrap / synthetic bear-market stress test** | Valetax BTC data has no long bear phase | `stefan-jansen/synthetic-data-for-finance` |
| ☐ | **Volatility-targeted sizing + Kelly ceiling** once capital allows | Fixed 0.01 lot ignores volatility (DXRG's main failure) | `deltaray-io/kelly-criterion`, `risk/sizing.py` |
| ☐ | **Telegram command bot**: `/status`, `/kill` only | Control from phone; no trade commands | `python-telegram-bot` |
| ☐ | **Smart-money-concepts features** (order blocks, FVG) as a filter test | Popular; test once and record the result | `joshyattridge/smart-money-concepts` |
| ☐ | **CI**: run tests + validation on every push | Keeps the live engine and backtest identical | GitHub Actions |

## Looked at, not adding
| Repo | Why not |
|---|---|
| `virattt/ai-hedge-fund`, `FinRobot`, `NoFxAiOS/nofx`, `TradingAgents`, AutoHedge, persona swarms | LLM signal generation / equity stock-picking |
| `nautilus_trader`, Lean, backtrader, vectorbt | You already have a faster numba engine with realistic costs |
| freqtrade, hummingbot, jesse, OctoBot | Exchange bots; you trade MT5 CFDs |
| `claude-trading-skills`, `finance-skills` | Equity-investor workflows |
| `jackson-video-resources/claude-tradingview-mcp-trading`, MCPs with order tools (Bybit, Binance, Alpaca, Robinhood) | Let an LLM place orders |
| Solana sniper/MEV, "2026 MT5 …" keyword-spam repos | Malware/scam pattern |
