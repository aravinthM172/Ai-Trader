"""
Final rule set for a FundedNext Stellar 2-Step $15K account, backtested end to end (2026-10-13).

    python -m backtest.fundednext_build          writes FUNDEDNEXT_BUILD.md

Rules = the live multi trader (H1 momentum, stop 2 ATR, target 6 ATR, 96 bars at most; daily EMA200 side for gold,
USDJPY and NDX100; buys only on USDJPY and NDX100; RSI >= 65 on gold and NDX100) plus the two additions of this build:
  calm    BTCUSD / ETHUSD signals are skipped when ATR % of price >= 1.2 x its 30-day median (strategy/calm_filter.py)
  cap     the BTCUSD + ETHUSD position value is held to 60 % of the account (risk/margin_cap.py; FundedNext: crypto at
          1:1 leverage, "70 % or more" margin use counts as gambling)
  exit    a BTCUSD / ETHUSD trade at least 2 ATR in profit is closed at the next open after an H1 candle of 1.5 ATR
          or more against it (strategy/reversal_exit.py)
  equal   0.35 % risk on all five markets (today ETHUSD and USDJPY run at half)
GER40 is not traded (no edge in any earlier study).

The challenge replay starts a new attempt every month: phase 1 +8 %, phase 2 +5 %, 5 trading days each, fail at
-5 % of the initial balance on one day or -10 % in total.  It uses closed trades only (a floating loss inside a day
is not seen), planned position sizes and the cost model of backtest/fp_strategy_build.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest import exit_rules_study as ex            # noqa: E402
from backtest import long_history_backtest as lh       # noqa: E402
from strategy import calm_filter, reversal_exit        # noqa: E402

OUT = ROOT / "FUNDEDNEXT_BUILD.md"
SYMS = ["BTCUSD", "ETHUSD", "XAUUSD", "USDJPY", "NDX100"]
CRYPTO = ("BTCUSD", "ETHUSD")
BAL = 15000.0
RISK = {"BTCUSD": .0035, "ETHUSD": .00175, "XAUUSD": .0035, "USDJPY": .00175, "NDX100": .0035}     # live today
EQUAL = {s: .0035 for s in SYMS}
CAP = 0.60
T0 = pd.Timestamp("2018-03-01", tz="UTC")              # all five markets have data from here
DAILY, TOTAL, TARGETS, MIN_DAYS = 0.05, 0.10, (0.08, 0.05), 5


def simulate_reversal(df: pd.DataFrame, cost: dict) -> pd.DataFrame:
    """ex.simulate (stop 2 ATR, target lh.TP) plus the reversal exit: once a completed bar closes at least
    MIN_PROFIT_ATR x the entry ATR in profit, is against the trade and CANDLE_ATR x ATR tall, leave at the next open."""
    fr = ex.fr
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    t = df["time"]
    ent, _ = ex.build_momentum(c, h, l, **ex.PARAMS)
    A = ex.base_atr(h, l, c, 14)
    big = (h - l) >= reversal_exit.CANDLE_ATR * A
    rows, i, n = [], 1, len(c)
    while i < n - 1:
        d, a = int(ent[i]), A[i - 1]
        if d == 0 or not np.isfinite(a) or a <= 0:
            i += 1
            continue
        ep = o[i]
        sl_d = fr.SL_ATR * a
        stop, tgt = ep - d * sl_d, ep + d * lh.TP * a
        x = xp = None
        at_open, leave, why = False, False, "time"
        for k in range(i, min(n, i + fr.HOLD)):
            if leave:
                x, xp, at_open, why = k, o[k], True, "reversal"
                break
            if k > i and d * (o[k] - stop) <= 0:
                x, xp, at_open, why = k, o[k], True, "stop"
                break
            if (l[k] <= stop) if d > 0 else (h[k] >= stop):
                x, xp, why = k, stop, "stop"
                break
            if k > i and d * (o[k] - tgt) >= 0:
                x, xp, at_open, why = k, o[k], True, "target"
                break
            if (h[k] >= tgt) if d > 0 else (l[k] <= tgt):
                x, xp, why = k, tgt, "target"
                break
            leave = bool(d * (c[k] - ep) >= reversal_exit.MIN_PROFIT_ATR * a and d * (c[k] - o[k]) < 0 and big[k])
        if x is None:
            x = min(n - 1, i + fr.HOLD)
            xp, at_open = o[x], True
        nt = fr.nights(t.iloc[i], t.iloc[x])
        rate = cost["swap_long"] if d > 0 else cost["swap_short"]
        rows.append((t.iloc[i], t.iloc[x], d, why, nt, (d * (xp - ep) - cost["cost_frac"] * ep + ep * rate / 365 * nt) / sl_d))
        i = x if at_open else x + 1
    return pd.DataFrame(rows, columns=["entry", "exit", "dir", "reason", "nights", "R"])


def trades(sym: str, reversal: bool = False) -> pd.DataFrame:
    """Live-rule trades of one market with the live filters applied and the calm ratio / stop size attached."""
    df = lh.h1(sym)[0].reset_index(drop=True)
    cost = lh.fb.cost_of(lh.SPECS[sym])
    one = (lambda p: simulate_reversal(p, cost)) if reversal else (lambda p: ex.simulate(p, lh.TP, cost))
    t = pd.concat([one(p) for p in ex.fr.segments(df)], ignore_index=True)
    h, l, c = (df[k].to_numpy(float) for k in ("high", "low", "close"))
    rel = pd.Series(ex.base_atr(h, l, c, calm_filter.ATR_LEN) / c)
    ratio = (rel / rel.rolling(calm_filter.MEDIAN_BARS, min_periods=calm_filter.MIN_BARS).median()).to_numpy()
    rsi = ex._rsi(c, 14)
    j = pd.Index(df.time).get_indexer(t.entry) - 1                      # signal bar
    t["rs"] = np.where(t.dir > 0, rsi[j], 100 - rsi[j])
    t["calm"], t["stop_frac"] = ratio[j], ex.fr.SL_ATR * rel.to_numpy()[j]
    day = (t.entry + pd.Timedelta(hours=3)).dt.tz_localize(None).dt.normalize()
    t["trend_ok"] = lh.daily_regime(df, True).side.reindex(day, method="ffill").to_numpy() == t.dir.to_numpy()
    if sym in ("XAUUSD", "NDX100"):
        t = t[t.trend_ok & (t.rs >= 65)]
    if sym in ("NDX100", "USDJPY"):
        t = t[t.trend_ok & (t.dir > 0)]
    return t.assign(sym=sym).sort_values("entry").reset_index(drop=True)


def book(tr: dict, *, calm: bool, cap: bool, haircut: float = 0.0, risk: dict | None = None) -> pd.DataFrame:
    """One dollar ledger for the account: every trade with its size after the cap, P/L booked at the exit."""
    risk = risk or RISK
    rows = []
    for s in SYMS:
        t = tr[s]
        if calm and s in CRYPTO:
            t = t[t.calm < calm_filter.MAX_RATIO]
        rows.append(t.assign(want=BAL * risk[s] / t.stop_frac if s in CRYPTO else 0.0))
    b = pd.concat(rows).sort_values("entry").reset_index(drop=True)
    b = b[b.entry >= T0].reset_index(drop=True)
    scale, open_pos = np.ones(len(b)), []
    if cap:
        for k, r in enumerate(b.itertuples()):
            if r.sym not in CRYPTO:
                continue
            open_pos = [(x, v) for x, v in open_pos if x > r.entry]
            room = CAP * BAL - sum(v for _, v in open_pos)
            scale[k] = float(np.clip(room / r.want, 0.0, 1.0)) if r.want > 0 else 1.0
            if scale[k] > 0:
                open_pos.append((r.exit, scale[k] * r.want))
    b["scale"] = scale
    b["pnl"] = (b.R - haircut) * b.sym.map(risk) * BAL * b.scale
    return b[b.scale > 0].sort_values("exit").reset_index(drop=True)


def phase(b: pd.DataFrame, start: pd.Timestamp, target: float):
    x = b[(b.exit >= start) & (b.exit < start + pd.Timedelta(days=540))]
    day = (x.exit + pd.Timedelta(hours=3)).dt.date.to_numpy()
    pnl, when = x.pnl.to_numpy(), list(x.exit)
    cum, dsum, days = 0.0, {}, set()
    for k in range(len(pnl)):
        cum += pnl[k]
        dsum[day[k]] = dsum.get(day[k], 0.0) + pnl[k]
        days.add(day[k])
        if dsum[day[k]] <= -DAILY * BAL or cum <= -TOTAL * BAL:
            return "fail", when[k]
        if cum >= target * BAL and len(days) >= MIN_DAYS:
            return "pass", when[k]
    return "open", None


def challenge(b: pd.DataFrame) -> dict:
    res = []
    for s0 in pd.date_range(T0, b.exit.max() - pd.Timedelta(days=400), freq="MS"):
        r1, t1 = phase(b, s0, TARGETS[0])
        if r1 != "pass":
            res.append((r1, None, None))
            continue
        r2, t2 = phase(b, t1, TARGETS[1])
        res.append((r2, (t1 - s0).days, (t2 - s0).days if r2 == "pass" else None))
    r = pd.DataFrame(res, columns=["res", "p1", "both"])
    yrs = (b.exit.max() - b.exit.min()).days / 365.25
    eq = b.pnl.cumsum()
    last3 = b[b.exit >= b.exit.max() - pd.Timedelta(days=1095)]
    daily = b.groupby((b.exit + pd.Timedelta(hours=3)).dt.date).pnl.sum()
    return {"attempts": len(r), "pass %": round(100 * (r.res == "pass").mean()), "fail %": round(100 * (r.res == "fail").mean()),
            "not finished %": round(100 * (r.res == "open").mean()), "phase 1 days": r.p1.median(), "both days": r.both.median(),
            "slowest pass days": r.both.max(), "$ / yr": round(b.pnl.sum() / yrs), "last 3 yrs $ / yr": round(last3.pnl.sum() / 3),
            "max DD $": round(float((eq.cummax() - eq).max())), "worst day $": round(float(daily.min())),
            "trades / yr": round(len(b) / yrs)}


def market_row(t: pd.DataFrame) -> dict:
    yrs = max((t.entry.max() - t.entry.min()).days / 365.25, 0.1)
    eq = t.R.cumsum()
    last3 = t[t.entry >= t.entry.max() - pd.Timedelta(days=1095)]
    yr = t.groupby(t.entry.dt.year).R.sum()
    run = longest = 0
    for lost in (t.R <= 0):
        run = run + 1 if lost else 0
        longest = max(longest, run)
    return {"from": str(t.entry.min().date()), "trades / yr": round(len(t) / yrs), "win %": round(100 * (t.R > 0).mean(), 1),
            "avg R": round(t.R.mean(), 3), "R / yr": round(t.R.sum() / yrs, 1),
            "t": round(t.R.mean() / t.R.std() * np.sqrt(len(t)), 1), "max DD R": round(float((eq.cummax() - eq).max()), 1),
            "last 3 yrs avg R": round(last3.R.mean(), 3), "years +": f"{(yr > 0).sum()}/{len(yr)}", "longest losing run": longest}


def _md(rows: dict) -> str:
    d = pd.DataFrame(rows).T
    head = "| | " + " | ".join(map(str, d.columns)) + " |\n|---|" + "--:|" * len(d.columns) + "\n"
    return head + "\n".join(f"| {i} | " + " | ".join("" if pd.isna(v) else f"{v:g}" if isinstance(v, float) else str(v)
                                                    for v in r) + " |" for i, r in d.iterrows()) + "\n"


def main() -> int:
    tr = {s: trades(s) for s in SYMS}
    trx = {s: (trades(s, reversal=True) if s in CRYPTO else tr[s]) for s in SYMS}       # reversal exit on BTC / ETH
    final = {s: (t[t.calm < calm_filter.MAX_RATIO] if s in CRYPTO else t) for s, t in trx.items()}
    markets = {s: market_row(t) for s, t in final.items()}
    rev = pd.concat([final[s] for s in CRYPTO])
    rev_n, rev_R = int((rev.reason == "reversal").sum()), float(rev[rev.reason == "reversal"].R.mean())
    FINAL = "+ equal 0.35 % on all five = FINAL"
    steps = {"live rules today": challenge(book(tr, calm=False, cap=False)),
             "+ calm filter on BTC / ETH": challenge(book(tr, calm=True, cap=False)),
             "+ 60 % position cap": challenge(book(tr, calm=True, cap=True)),
             "+ reversal exit on BTC / ETH": challenge(book(trx, calm=True, cap=True)),
             FINAL: challenge(book(trx, calm=True, cap=True, risk=EQUAL))}
    stress = {f"every trade {h:.2f}R worse": challenge(book(trx, calm=True, cap=True, risk=EQUAL, haircut=h))
              for h in (0.05, 0.10, 0.15)}
    fb = book(trx, calm=True, cap=True, risk=EQUAL)
    yearly = fb.groupby(fb.exit.dt.year).pnl.sum().round().astype(int)
    by_sym = fb.groupby("sym").pnl.sum() / ((fb.exit.max() - fb.exit.min()).days / 365.25)
    capped = fb[(fb.scale < 1) & fb.sym.isin(CRYPTO)]
    crypto_n = int(fb.sym.isin(CRYPTO).sum())
    text = f"""# FundedNext $15K build -- final rules, backtested end to end

Generated by `python -m backtest.fundednext_build`.  Account ${BAL:,.0f}.  Rules: the live H1 momentum trader (stop
2 ATR, target 6 ATR) with its filters, plus on BTCUSD / ETHUSD a calm-market filter, a {CAP:.0%} cap on their combined
position value and a reversal exit, and 0.35 % risk on all five markets.  Challenge: +8 % then +5 %, 5 trading days per
phase, -5 % of the initial balance in a day or -10 % in total fails.  Closed trades only; planned sizes.

## Each market under the final rules (R = risk of one trade)

{_md(markets)}
## What each addition does to the account (portfolio from {T0.date()})

{_md(steps)}
The position cap cut the size of {len(capped)} of {crypto_n} BTC / ETH trades (median size kept:
{capped.scale.median():.0%}).  The reversal exit closed {rev_n} of {len(rev)} BTC / ETH trades, at {rev_R:+.2f}R on
average.  FundedNext's maximum loss on this account is ${TOTAL * BAL:,.0f} and its daily limit ${DAILY * BAL:,.0f}.

## If the edge is weaker than the backtest (final rules)

{_md(stress)}
The live record on 2026-10-13 was 25 trades at -0.16R, so these lines matter more than the table above.

## Final rules, dollars per year by market

{_md({s: {"$ / yr": round(v)} for s, v in by_sym.items()})}
## Final rules, dollars per calendar year

{_md({str(y): {"$": int(v)} for y, v in yearly.items()})}
## Limits of this test

- Found on one week of live losses: the calm filter is a strong lead, not a proven edge, and only for BTC / ETH.
- Closed trades only: a loss that is open inside a day does not count against the daily limit here.
- Sizes are the planned ones; real lots are rounded to the broker's step and its smallest lot.
- The reversal exit was the best-behaved of 16 variants tried on the same history: "same profit, smoother", not an edge.
- Monthly attempts overlap, so the pass rate is a few independent periods, not {steps[FINAL]['attempts']}.
"""
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
