"""
End-to-end BTC decision pipeline (shared by run_btc.py and the diagnostic).

    MT5 -> account -> symbol spec -> tick -> history -> data quality ->
    features -> ATR -> signal -> spread filter -> stop plan -> position sizing ->
    safety controller -> order validation (order_check, READ-ONLY)

Returns a single dict with every stage's result.  NEVER sends an order.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

from common.logging_setup import get_logger
from config.assets import get_asset_config
from data.quality import validate_ohlc
from execution import safety, state as state_store
from execution.order_validator import validate as validate_order
from mt5.gateway import MT5Gateway
from risk.sizing import size_position
from strategy.btc_features import compute_features, latest_feature_row
from strategy.btc_signal import generate as generate_signal
from strategy import btc_h1_signal
from strategy.stops import build_stop_plan, spread_filter

log = get_logger("btc.pipeline")


def _stage(ok: bool, name: str) -> str:
    return f"{'PASS' if ok else 'FAIL'}  {name}"


def run_pipeline(asset: str = "BTC", *, history_bars: int | None = None,
                 force_direction: str | None = None,
                 ignore_spread_filter: bool = False,
                 record_state: bool = False,
                 timeframe: str | None = None,
                 paper_sim_balance: float | None = None) -> dict:
    """
    paper_sim_balance : PAPER MODE ONLY.  Overrides the balance used for
    hypothetical position sizing / daily-loss basis so an H1 forward test can be
    run at its intended account size while the real account stays $100 and
    LIVE_TRADING stays false.  Never affects order_check or any live path.
    """
    cfg = get_asset_config(asset, timeframe=timeframe)
    if history_bars is None:
        history_bars = 4000 if cfg.timeframe in ("H1", "H4") else 3000
    out: dict = {
        "asset": asset,
        "symbol": cfg.symbol,
        "timeframe": cfg.timeframe,
        "strategy": cfg.strategy,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "live_trading_enabled": safety.live_trading_enabled(),
        "kill_switch_active": safety.kill_switch_active(),
        "stages": [],
        "ready": False,
    }
    st = out["stages"]

    gw = MT5Gateway()
    if not gw.connect():
        out["error"] = "MT5 connection failed"
        st.append(_stage(False, "mt5_connect"))
        return out
    st.append(_stage(True, "mt5_connect"))

    try:
        account = gw.account_info()
        terminal = gw.terminal_info()
        out["account"] = account
        out["terminal"] = terminal
        out["real_account_balance"] = account.get("balance", 0.0)
        # PAPER-ONLY sizing basis; the real account and every live gate are untouched.
        sizing_balance = float(paper_sim_balance) if paper_sim_balance else account.get("balance", 0.0)
        out["sizing_balance"] = sizing_balance
        if paper_sim_balance:
            out["paper_sim_balance_note"] = (
                f"hypothetical sizing on ${sizing_balance:.0f} (real account ${out['real_account_balance']:.2f}); "
                f"paper only, LIVE_TRADING={out['live_trading_enabled']}")
        st.append(_stage(bool(account), "account_info"))

        spec = gw.get_spec(cfg.symbol)
        if spec is None:
            out["error"] = f"symbol {cfg.symbol} spec unavailable"
            out["btc_symbols_found"] = gw.find_btc_symbols()
            st.append(_stage(False, "symbol_spec"))
            return out
        out["symbol_spec"] = spec.to_dict()
        # apply live spec onto cfg hints
        cfg.point = spec.point
        cfg.contract_size = spec.contract_size
        cfg.tick_size = spec.tick_size
        cfg.tick_value = spec.tick_value
        cfg.volume_min = spec.volume_min
        cfg.volume_step = spec.volume_step
        cfg.volume_max = spec.volume_max
        cfg.broker_stops_level_points = spec.stops_level_points
        st.append(_stage(spec.trade_mode == 4, "symbol_spec (trade FULL)"))

        tick = gw.get_tick(cfg.symbol)
        out["tick"] = tick
        tick_ok = bool(tick and tick["bid"] > 0 and tick["ask"] > 0)
        st.append(_stage(tick_ok, "tick"))

        # -- timeframes ---------------------------------------------
        tf_status = {tf: gw.valid_timeframe(tf) for tf in ("M1", "M5", "M15", "H1", "H4")}
        out["timeframe_support"] = tf_status
        st.append(_stage(all(tf_status.values()), "timeframe_support"))

        # -- history + quality --------------------------------------
        raw = gw.get_rates(cfg.symbol, cfg.timeframe, history_bars)
        clean, qrep = validate_ohlc(raw, symbol=cfg.symbol, timeframe=cfg.timeframe, cfg=cfg)
        out["data_quality"] = qrep.to_dict()
        out["history_bars_returned"] = int(len(raw)) if raw is not None else 0
        st.append(_stage(qrep.ok, f"data_quality ({cfg.timeframe}, {out['history_bars_returned']} bars)"))
        if not qrep.ok or clean.empty:
            out["error"] = "data quality failed: " + "; ".join(qrep.reasons)
            return out

        # signal on the last COMPLETED bar -> the most recent row of `clean` is the
        # still-forming bar; drop it for the decision so the fill is "next bar open".
        completed = clean.iloc[:-1].reset_index(drop=True) if len(clean) > 1 else clean
        out["signal_bar_utc"] = str(completed["time"].iloc[-1]) if len(completed) else None

        # -- features -----------------------------------------------
        spread_price = tick["spread"] if tick_ok else None
        feat_df = compute_features(completed, cfg, spread_price=spread_price)
        feat = latest_feature_row(feat_df)
        out["features"] = feat
        feat_ok = bool(feat) and feat.get("atr") is not None
        st.append(_stage(feat_ok, "features"))
        if not feat_ok:
            out["error"] = "feature computation produced no valid row"
            return out

        atr = float(feat["atr"])                          # ATR of THIS timeframe (H1 ATR on H1)
        out["atr"] = atr
        st.append(_stage(atr > 0, f"atr ({cfg.timeframe})"))

        # -- signal ------------------------------------------------
        min_conf = float(os.getenv("MIN_CONFIDENCE", "0.70"))
        if cfg.strategy == "momentum_rsi_mtf":
            sig = btc_h1_signal.generate(cfg.symbol, completed, min_confidence=min_conf)
        else:
            sig = generate_signal(cfg.symbol, feat, min_confidence=min_conf)
        out["signal"] = sig.to_dict()
        out["signal_strategy"] = getattr(sig, "strategy", cfg.strategy)
        direction = force_direction or (sig.decision if sig.decision in ("BUY", "SELL") else None)
        out["evaluated_direction"] = direction
        st.append(_stage(True, f"signal [{cfg.strategy}] ({sig.decision} conf {sig.confidence})"))

        # -- spread filter -------------------------------------------
        price = (tick["bid"] + tick["ask"]) / 2.0
        sp_ok, sp_ctx = spread_filter(tick["spread"], atr, price, cfg)
        sp_ctx["ignored_for_testing"] = bool(ignore_spread_filter)
        out["spread_filter"] = sp_ctx
        st.append(_stage(sp_ok, "spread_filter"))
        sp_effective = sp_ok or ignore_spread_filter

        if direction is None:
            out["decision"] = "HOLD"
            out["decision_reason"] = f"signal={sig.decision} ({sig.reason})"
            out["ready"] = qrep.ok and feat_ok and tick_ok
            log.info("[%s] no actionable direction -> HOLD", cfg.symbol)
            return out

        # -- stop plan --------------------------------------------
        plan = build_stop_plan(
            direction=direction, bid=tick["bid"], ask=tick["ask"], atr=atr, cfg=cfg,
            broker_stops_level_price=spec.stops_level_price,
            slippage_price=float(os.getenv("BTC_SLIPPAGE_PRICE", "1.0")),
        )
        out["stop_plan"] = plan.to_dict()
        st.append(_stage(plan.accepted, "stop_plan"))

        # -- sizing ----------------------------------------------
        sizing = size_position(direction=direction, entry=plan.entry, sl=plan.sl,
                               balance=sizing_balance, spec=spec, cfg=cfg, gateway=gw,
                               sim_free_margin=(sizing_balance if paper_sim_balance else None))
        out["sizing"] = sizing.to_dict()
        st.append(_stage(sizing.accepted, "position_sizing"))

        # -- safety --------------------------------------------
        state = state_store.load()
        state_store.set_start_balance(state, sizing_balance)
        safe = safety.evaluate(
            symbol=cfg.symbol, direction=direction, cfg=cfg, account=account,
            open_positions=gw.positions(),
            tick_age_seconds=tick.get("age_seconds"),
            spread_ok=sp_effective, sizing_ok=sizing.accepted, state=state,
        )
        out["safety"] = safe.to_dict()
        st.append(_stage(safe.allow_validation, "safety_controller"))

        # -- order validation (READ-ONLY) -----------------------
        if plan.accepted and sizing.accepted and sizing.volume > 0 and safe.allow_validation:
            ov = validate_order(
                gateway=gw, spec=spec, direction=direction, entry=plan.entry,
                volume=sizing.volume, sl=plan.sl, tp=plan.tp, atr=atr,
                spread_price=plan.spread_price, risk_dollars=sizing.risk_dollars,
                balance=sizing_balance,
                required_margin=sizing.required_margin, free_margin=sizing.free_margin_before,
                stop_plan_checks=plan.checks,
            )
            out["order_validation"] = ov.to_dict()
            out["order_validation_render"] = ov.render()
            st.append(_stage(ov.valid, "order_check (dry-run)"))
            approved = ov.valid
        else:
            out["order_validation"] = None
            reasons = plan.reasons + sizing.reasons + safe.reasons
            out["order_validation_render"] = "## BTC TRADE VALIDATION\nDecision : REJECTED (pre-order)\n" \
                                             "Reasons  : " + "; ".join(reasons)
            st.append(_stage(False, "order_check (skipped - pre-order rejection)"))
            approved = False

        out["decision"] = "APPROVED_DRY_RUN" if approved else "REJECTED"
        out["decision_reason"] = (
            "all stages passed; order validated read-only; NOT sent (LIVE_TRADING=%s)"
            % out["live_trading_enabled"]
            if approved else
            "; ".join(plan.reasons + sizing.reasons + safe.reasons
                      + (out.get("order_validation", {}) or {}).get("reasons", []))
        )
        if approved and record_state:
            state_store.record_intent(state, cfg.symbol)
            state_store.save(state)
        out["state_recorded"] = bool(approved and record_state)

        out["ready"] = qrep.ok and feat_ok and tick_ok and plan.accepted
        return out
    finally:
        gw.shutdown()
