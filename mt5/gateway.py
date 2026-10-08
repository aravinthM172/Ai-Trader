"""
Robust MT5 gateway -- the ONLY module that imports MetaTrader5 for the BTC path.

Everything is defensive:
  * every SymbolInfo / AccountInfo field is read with getattr(..., default)
    (the installed package exposes ``trade_exemode`` not ``trade_execution``)
  * historical requests are chunked (the terminal rejects very large counts
    with "Invalid params")
  * order_send() is NEVER called from this module.  order_check() is used for
    validation only.

Legacy mt5/client.py and mt5/multi_asset.py are left untouched for XAUUSD.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

import pandas as pd

try:
    import MetaTrader5 as _mt5
    MT5_AVAILABLE = True
except Exception:                                    # pragma: no cover
    _mt5 = None
    MT5_AVAILABLE = False

from common.logging_setup import get_logger

log = get_logger("btc.mt5")

_TF_NAMES = ("M1", "M5", "M15", "M30", "H1", "H4", "D1")
_MAX_BARS_PER_CALL = 40_000          # stay under the terminal's hard limit


def _timeframe_map() -> dict[str, int]:
    if not MT5_AVAILABLE:
        return {n: i for i, n in enumerate(_TF_NAMES)}
    return {n: getattr(_mt5, f"TIMEFRAME_{n}") for n in _TF_NAMES
            if hasattr(_mt5, f"TIMEFRAME_{n}")}


def _tf_seconds(name: str) -> int:
    return {"M1": 60, "M5": 300, "M15": 900, "M30": 1800,
            "H1": 3600, "H4": 14400, "D1": 86400}.get(name.upper(), 300)


def field(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


@dataclass
class SymbolSpec:
    symbol: str
    description: str
    digits: int
    point: float
    tick_size: float
    tick_value: float
    tick_value_profit: float
    tick_value_loss: float
    contract_size: float
    volume_min: float
    volume_max: float
    volume_step: float
    stops_level_points: float
    freeze_level_points: float
    filling_mode: int
    order_mode: int
    trade_mode: int
    trade_calc_mode: int
    trade_exemode: Optional[int]
    spread_points: float
    spread_float: bool
    currency_base: str
    currency_profit: str
    currency_margin: str

    @property
    def stops_level_price(self) -> float:
        return self.stops_level_points * self.point

    @property
    def value_per_price_unit_per_lot(self) -> float:
        """$ P/L per 1.0 price move per 1.0 lot (tick_value / tick_size)."""
        if self.tick_size > 0:
            return self.tick_value / self.tick_size
        return self.contract_size

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["stops_level_price"] = self.stops_level_price
        d["value_per_price_unit_per_lot"] = self.value_per_price_unit_per_lot
        return d


class MT5Gateway:
    def __init__(self) -> None:
        self._connected = False
        self.timeframes = _timeframe_map()
        self.server_utc_offset_seconds: int = 0     # server_time - true_utc, detected at connect

    # -- server timezone -------------------------------------------
    def _detect_server_offset(self) -> None:
        """MT5 returns timestamps in the BROKER SERVER timezone (e.g. GMT+3).
        Estimate the offset from the most recent M5 bar vs true UTC now.

        Uses the FRESHEST bar across visible symbols (plus BTCUSD.vx, which trades
        24/7): on weekends the first visible symbol is usually a closed FX pair whose
        last bar is days old, which used to leave the offset at 0."""
        if not MT5_AVAILABLE:
            return
        try:
            tf = self.timeframes.get("M5")
            best = None
            _mt5.symbol_select("BTCUSD.vx", True)
            syms = _mt5.symbols_get() or []
            for s in syms[:50]:
                if not getattr(s, "visible", False):
                    continue
                r = _mt5.copy_rates_from_pos(s.name, tf, 0, 1)
                if r is not None and len(r):
                    best = max(best or 0, int(r[-1][0]))
            if best is None:
                return
            now = datetime.now(timezone.utc).timestamp()
            diff = best - now
            # round to the nearest hour; only trust plausible broker offsets (|<=14h|)
            hrs = round(diff / 3600.0)
            if abs(hrs) <= 14:
                self.server_utc_offset_seconds = int(hrs * 3600)
                if self.server_utc_offset_seconds:
                    log.info("detected server time offset: %+d h (MT5 timestamps are broker-local)",
                             hrs)
        except Exception as e:                       # pragma: no cover
            log.warning("server offset detection failed: %s", e)

    def to_utc(self, server_epoch: float) -> datetime:
        return datetime.fromtimestamp(server_epoch - self.server_utc_offset_seconds, tz=timezone.utc)

    # -- lifecycle -----------------------------------------------------
    def connect(self) -> bool:
        if not MT5_AVAILABLE:
            log.error("MetaTrader5 package not importable")
            return False
        if self._connected:
            return True
        if not _mt5.initialize():
            log.error("mt5.initialize failed: %s", _mt5.last_error())
            return False
        self._connected = True
        self._detect_server_offset()
        acc = self.account_info()
        term = self.terminal_info()
        log.info("MT5 connected | server=%s currency=%s balance=%.2f | terminal.trade_allowed=%s",
                 acc.get("server"), acc.get("currency"), acc.get("balance", 0.0),
                 term.get("trade_allowed"))
        return True

    def shutdown(self) -> None:
        if MT5_AVAILABLE and self._connected:
            _mt5.shutdown()
        self._connected = False

    def raw(self):
        return _mt5

    # -- account / terminal -----------------------------------------
    def account_info(self) -> dict:
        if not MT5_AVAILABLE:
            return {}
        a = _mt5.account_info()
        if a is None:
            return {}
        return {
            "login": field(a, "login"),
            "server": field(a, "server"),
            "currency": field(a, "currency"),
            "balance": float(field(a, "balance", 0.0)),
            "equity": float(field(a, "equity", 0.0)),
            "margin": float(field(a, "margin", 0.0)),
            "margin_free": float(field(a, "margin_free", 0.0)),
            "margin_level": float(field(a, "margin_level", 0.0)),
            "leverage": int(field(a, "leverage", 0)),
            "trade_allowed": bool(field(a, "trade_allowed", False)),
            "profit": float(field(a, "profit", 0.0)),
            "trade_mode": int(field(a, "trade_mode", -1)),    # 0 demo, 1 contest, 2 real
        }

    def terminal_info(self) -> dict:
        if not MT5_AVAILABLE:
            return {}
        t = _mt5.terminal_info()
        if t is None:
            return {}
        return {
            "trade_allowed": bool(field(t, "trade_allowed", False)),
            "connected": bool(field(t, "connected", False)),
            "name": field(t, "name"),
            "build": field(t, "build"),
        }

    # -- symbol -----------------------------------------------------
    def find_btc_symbols(self) -> list[str]:
        if not MT5_AVAILABLE:
            return []
        syms = _mt5.symbols_get() or []
        return sorted(s.name for s in syms if "BTC" in s.name.upper())

    def ensure_symbol(self, symbol: str) -> bool:
        if not MT5_AVAILABLE:
            return False
        info = _mt5.symbol_info(symbol)
        if info is None:
            log.error("symbol %s not found", symbol)
            return False
        if not field(info, "visible", True):
            if not _mt5.symbol_select(symbol, True):
                log.error("symbol_select(%s) failed: %s", symbol, _mt5.last_error())
                return False
            time.sleep(0.2)
        return True

    def get_spec(self, symbol: str) -> Optional[SymbolSpec]:
        if not MT5_AVAILABLE or not self.ensure_symbol(symbol):
            return None
        i = _mt5.symbol_info(symbol)
        return SymbolSpec(
            symbol=field(i, "name", symbol),
            description=field(i, "description", ""),
            digits=int(field(i, "digits", 2)),
            point=float(field(i, "point", 0.01)),
            tick_size=float(field(i, "trade_tick_size", field(i, "point", 0.01))),
            tick_value=float(field(i, "trade_tick_value", 0.0)),
            tick_value_profit=float(field(i, "trade_tick_value_profit", 0.0)),
            tick_value_loss=float(field(i, "trade_tick_value_loss", 0.0)),
            contract_size=float(field(i, "trade_contract_size", 1.0)),
            volume_min=float(field(i, "volume_min", 0.01)),
            volume_max=float(field(i, "volume_max", 0.0)),
            volume_step=float(field(i, "volume_step", 0.01)),
            stops_level_points=float(field(i, "trade_stops_level", 0)),
            freeze_level_points=float(field(i, "trade_freeze_level", 0)),
            filling_mode=int(field(i, "filling_mode", 0)),
            order_mode=int(field(i, "order_mode", 0)),
            trade_mode=int(field(i, "trade_mode", 0)),
            trade_calc_mode=int(field(i, "trade_calc_mode", 0)),
            trade_exemode=field(i, "trade_exemode", None),
            spread_points=float(field(i, "spread", 0)),
            spread_float=bool(field(i, "spread_float", True)),
            currency_base=field(i, "currency_base", ""),
            currency_profit=field(i, "currency_profit", ""),
            currency_margin=field(i, "currency_margin", ""),
        )

    def calc_value_per_unit(self, symbol: str, direction: str, price: float) -> Optional[float]:
        """Account-currency loss of 1.0 lot over a 1.0 adverse price move, from the broker's own
        profit calculator (order_calc_profit).  None when MT5 cannot compute it.  Independent of
        trade_tick_value, which some brokers report wrongly (Valetax DAX40.vx: EUR contract, tick
        value says $1/point/lot, real P/L is ~$11)."""
        if not MT5_AVAILABLE or price <= 0:
            return None
        buy = direction == "BUY"
        order_type = _mt5.ORDER_TYPE_BUY if buy else _mt5.ORDER_TYPE_SELL
        exit_px = price - 1.0 if buy else price + 1.0
        v = _mt5.order_calc_profit(order_type, symbol, 1.0, price, exit_px)
        return abs(float(v)) if v else None

    def get_tick(self, symbol: str) -> Optional[dict]:
        if not MT5_AVAILABLE or not self.ensure_symbol(symbol):
            return None
        t = _mt5.symbol_info_tick(symbol)
        if t is None:
            return None
        bid = float(field(t, "bid", 0.0))
        ask = float(field(t, "ask", 0.0))
        ts = int(field(t, "time", 0))
        t_utc = self.to_utc(ts) if ts else None
        age = (datetime.now(timezone.utc) - t_utc).total_seconds() if t_utc else None
        return {
            "symbol": symbol,
            "bid": bid,
            "ask": ask,
            "last": float(field(t, "last", 0.0)),
            "spread": max(0.0, ask - bid),
            "time": ts,
            "time_utc": t_utc.isoformat() if t_utc else None,
            "age_seconds": age,
            "server_utc_offset_hours": self.server_utc_offset_seconds // 3600,
        }

    # -- history --------------------------------------------------
    def valid_timeframe(self, name: str) -> bool:
        return name.upper() in self.timeframes

    def get_rates(self, symbol: str, timeframe: str, count: int = 5000) -> Optional[pd.DataFrame]:
        """copy_rates_from_pos, chunked to stay within terminal limits."""
        if not MT5_AVAILABLE or not self.ensure_symbol(symbol):
            return None
        if not self.valid_timeframe(timeframe):
            log.error("unsupported timeframe %s", timeframe)
            return None
        tf = self.timeframes[timeframe.upper()]
        frames: list[pd.DataFrame] = []
        got = 0
        pos = 0
        remaining = int(count)
        while remaining > 0:
            n = min(remaining, _MAX_BARS_PER_CALL)
            rates = _mt5.copy_rates_from_pos(symbol, tf, pos, n)
            if rates is None or len(rates) == 0:
                if got == 0:
                    log.warning("copy_rates_from_pos(%s,%s) returned nothing: %s",
                                symbol, timeframe, _mt5.last_error())
                break
            frames.append(pd.DataFrame(rates))
            got += len(rates)
            if len(rates) < n:
                break
            pos += len(rates)
            remaining -= len(rates)
        if not frames:
            return None
        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
        df["time"] = pd.to_datetime(df["time"] - self.server_utc_offset_seconds, unit="s", utc=True)
        return df

    def get_rates_range(self, symbol: str, timeframe: str,
                        start: datetime, end: datetime,
                        chunk_days: int = 20) -> Optional[pd.DataFrame]:
        """Deep history via copy_rates_range in day-chunks (for backtests)."""
        if not MT5_AVAILABLE or not self.ensure_symbol(symbol):
            return None
        if not self.valid_timeframe(timeframe):
            return None
        tf = self.timeframes[timeframe.upper()]
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        frames = []
        cur = start
        while cur < end:
            nxt = min(cur + timedelta(days=chunk_days), end)
            rates = _mt5.copy_rates_range(symbol, tf, cur, nxt)
            if rates is not None and len(rates):
                frames.append(pd.DataFrame(rates))
            cur = nxt
            time.sleep(0.05)
        if not frames:
            return None
        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
        df["time"] = pd.to_datetime(df["time"] - self.server_utc_offset_seconds, unit="s", utc=True)
        return df

    # -- broker maths (read-only) --------------------------------
    def calc_margin(self, symbol: str, order_type: str, volume: float, price: float) -> Optional[float]:
        if not MT5_AVAILABLE:
            return None
        ot = _mt5.ORDER_TYPE_BUY if order_type.upper() == "BUY" else _mt5.ORDER_TYPE_SELL
        m = _mt5.order_calc_margin(ot, symbol, volume, price)
        return None if m is None else float(m)

    def calc_profit(self, symbol: str, order_type: str, volume: float,
                    price_open: float, price_close: float) -> Optional[float]:
        if not MT5_AVAILABLE:
            return None
        ot = _mt5.ORDER_TYPE_BUY if order_type.upper() == "BUY" else _mt5.ORDER_TYPE_SELL
        p = _mt5.order_calc_profit(ot, symbol, volume, price_open, price_close)
        return None if p is None else float(p)

    def order_check(self, request: dict) -> Optional[dict]:
        """READ-ONLY broker validation.  Never sends."""
        if not MT5_AVAILABLE:
            return None
        r = _mt5.order_check(request)
        if r is None:
            return None
        out = {
            "retcode": int(field(r, "retcode", -1)),
            "comment": field(r, "comment", ""),
            "balance": float(field(r, "balance", 0.0)),
            "equity": float(field(r, "equity", 0.0)),
            "margin": float(field(r, "margin", 0.0)),
            "margin_free": float(field(r, "margin_free", 0.0)),
            "margin_level": float(field(r, "margin_level", 0.0)),
        }
        req = field(r, "request", None)
        if req is not None:
            out["request"] = {
                "price": float(field(req, "price", 0.0)),
                "sl": float(field(req, "sl", 0.0)),
                "tp": float(field(req, "tp", 0.0)),
                "volume": float(field(req, "volume", 0.0)),
            }
        return out

    def positions(self, symbol: Optional[str] = None) -> list[dict]:
        if not MT5_AVAILABLE:
            return []
        pos = _mt5.positions_get(symbol=symbol) if symbol else _mt5.positions_get()
        if pos is None:
            return []
        return [{
            "ticket": field(p, "ticket"),
            "symbol": field(p, "symbol"),
            "type": "BUY" if field(p, "type", 0) == 0 else "SELL",
            "volume": float(field(p, "volume", 0.0)),
            "price_open": float(field(p, "price_open", 0.0)),
            "sl": float(field(p, "sl", 0.0)),
            "tp": float(field(p, "tp", 0.0)),
            "profit": float(field(p, "profit", 0.0)),
            "swap": float(field(p, "swap", 0.0)),
            "time": int(field(p, "time", 0)),
            "magic": int(field(p, "magic", 0)),
        } for p in pos]

    def history_deals_today(self) -> list[dict]:
        if not MT5_AVAILABLE:
            return []
        now = datetime.now(timezone.utc)
        start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
        deals = _mt5.history_deals_get(start, now + timedelta(minutes=1))
        if deals is None:
            return []
        return [{
            "symbol": field(d, "symbol"),
            "profit": float(field(d, "profit", 0.0)),
            "time": int(field(d, "time", 0)),
            "entry": int(field(d, "entry", 0)),
        } for d in deals]

    # -- filling mode helper ------------------------------------
    def preferred_filling(self, spec: SymbolSpec) -> int:
        """Return an ORDER_FILLING_* constant the symbol accepts (bitmask)."""
        if not MT5_AVAILABLE:
            return 0
        fm = spec.filling_mode
        if fm & 1:                       # SYMBOL_FILLING_FOK
            return _mt5.ORDER_FILLING_FOK
        if fm & 2:                       # SYMBOL_FILLING_IOC
            return _mt5.ORDER_FILLING_IOC
        return _mt5.ORDER_FILLING_RETURN
