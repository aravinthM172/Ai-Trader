
import MetaTrader5 as mt5

SYMBOL = "XAUUSD"

if not mt5.initialize():
    raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

s = mt5.symbol_info(SYMBOL)
t = mt5.symbol_info_tick(SYMBOL)
a = mt5.account_info()

if s is None or t is None or a is None:
    mt5.shutdown()
    raise RuntimeError("Could not read MT5 symbol/tick/account data")

print("=" * 70)
print("XAUUSD BROKER SPEC CHECK")
print("=" * 70)

print(f"Symbol              : {s.name}")
print(f"Digits              : {s.digits}")
print(f"Point               : {s.point}")
print(f"Tick size           : {s.trade_tick_size}")
print(f"Tick value          : {s.trade_tick_value}")
print(f"Contract size       : {s.trade_contract_size}")
print(f"Min lot             : {s.volume_min}")
print(f"Lot step            : {s.volume_step}")
print(f"Max lot             : {s.volume_max}")
print(f"Stops level         : {s.trade_stops_level}")
print(f"Freeze level        : {s.trade_freeze_level}")
print(f"Leverage            : 1:{a.leverage}")
print(f"Bid                 : {t.bid}")
print(f"Ask                 : {t.ask}")
print(f"Spread              : {t.ask - t.bid:.2f}")
print(f"Spread points       : {(t.ask-t.bid)/s.point:.0f}")

# MT5 P/L identity:
# profit = price_move / tick_size * tick_value * volume
for move in [0.01, 0.10, 1.00]:
    pnl = (move / s.trade_tick_size) * s.trade_tick_value * 1.0
    print(f"P/L for ${move:.2f} move @ 1.00 lot : ${pnl:.2f}")

print("=" * 70)
print("EXPECTED ENGINE CONSTANTS")
print("=" * 70)
print("TICK_SIZE  = 0.01")
print("TICK_VALUE = 0.10")
print("VALUE/$1   = $10.00 per 1.00 lot")
print("=" * 70)

mt5.shutdown()
