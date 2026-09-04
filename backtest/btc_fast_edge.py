import pandas as pd
import numpy as np

df = pd.read_csv(r".\data\btc_m5_history.csv")

c = df["close"].to_numpy(float)
h = df["high"].to_numpy(float)
l = df["low"].to_numpy(float)
tv = df["tick_volume"].to_numpy(float)
rv = df["real_volume"].to_numpy(float)

prev = np.r_[np.nan, c[:-1]]

tr = np.maximum.reduce([
    h - l,
    np.abs(h - prev),
    np.abs(l - prev)
])

atr = pd.Series(tr).rolling(20).mean().to_numpy()


def spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    mask = np.isfinite(x) & np.isfinite(y)

    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan

    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()

    sx = rx.std()
    sy = ry.std()

    if sx == 0 or sy == 0:
        return np.nan

    return np.corrcoef(rx, ry)[0, 1]


print("=" * 80)
print("BTC FAST EDGE TEST")
print("=" * 80)
print("BARS:", len(df))

features = [
    ("tick_volume", tv),
    ("real_volume", rv),
    ("ATR", atr),
]

for name, x in features:

    print("\nFEATURE:", name)
    print("-" * 80)

    for H in [1, 3, 6, 12, 24]:

        fwd = np.r_[
            c[H:] / c[:-H] - 1,
            np.full(H, np.nan)
        ]

        mask = (
            (x > 0)
            & np.isfinite(x)
            & np.isfinite(fwd)
        )

        xx = x[mask]
        yy = fwd[mask]

        rho_mag = spearman(xx, np.abs(yy))
        rho_dir = spearman(xx, yy)

        print(
            "H={:2d}: magnitude rho={:+.4f} | "
            "direction rho={:+.4f} | n={}".format(
                H,
                rho_mag,
                rho_dir,
                len(xx)
            )
        )


print("\n")
print("=" * 80)
print("TRAIN / VALIDATION / FINAL - H=6")
print("=" * 80)

H = 6

fwd = np.r_[
    c[H:] / c[:-H] - 1,
    np.full(H, np.nan)
]

idx = np.arange(len(df))

splits = [
    ("TRAIN", 0, int(len(df) * 0.70)),
    ("VALIDATION", int(len(df) * 0.70), int(len(df) * 0.85)),
    ("FINAL", int(len(df) * 0.85), len(df)),
]

for name, x in features:

    print("\nFEATURE:", name)

    for label, start, end in splits:

        mask = (
            (idx >= start)
            & (idx < end)
            & (x > 0)
            & np.isfinite(x)
            & np.isfinite(fwd)
        )

        rho_mag = spearman(
            x[mask],
            np.abs(fwd[mask])
        )

        rho_dir = spearman(
            x[mask],
            fwd[mask]
        )

        print(
            "{:10s}: magnitude rho={:+.4f} | "
            "direction rho={:+.4f} | n={}".format(
                label,
                rho_mag,
                rho_dir,
                mask.sum()
            )
        )

print("\nDONE")
