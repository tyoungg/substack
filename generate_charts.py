import os
import yaml
import numpy as np
import pandas as pd
import yfinance as yf
import mplfinance as mpf

# ----------------------------
# Load symbols
# ----------------------------
with open("symbols.yaml", "r") as f:
    symbols = yaml.safe_load(f)["symbols"]

os.makedirs("charts", exist_ok=True)

# ----------------------------
# Trendline helper
# ----------------------------
def find_trendline_points(closes):
    i1 = int(np.argmin(closes))

    i2 = None
    for i in range(i1 + 5, len(closes)):
        if closes[i] > closes[i1]:
            i2 = i
            break

    if i2 is None:
        i2 = min(i1 + 1, len(closes) - 1)

    return i1, i2

# ----------------------------
# Main loop
# ----------------------------
for symbol in symbols:
    df = yf.download(
        symbol,
        period="1y",
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    df = df.dropna()

    # 🔑 CRITICAL FIX: flatten MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Build mplfinance-safe DataFrame (guaranteed 1-D floats)
    clean_df = pd.DataFrame(
        {
            "Open":  df["Open"].to_numpy().astype("float64").ravel(),
            "High":  df["High"].to_numpy().astype("float64").ravel(),
            "Low":   df["Low"].to_numpy().astype("float64").ravel(),
            "Close": df["Close"].to_numpy().astype("float64").ravel(),
        },
        index=pd.to_datetime(df.index)
    )

    # ----------------------------
    # Trendline (pure NumPy)
    # ----------------------------
    # closes = clean_df["Close"].to_numpy()
    # i1, i2 = find_trendline_points(closes)

    # x = np.arange(len(closes), dtype="float64")
    # y1 = closes[i1]
    # y2 = closes[i2]

    # slope = (y2 - y1) / (i2 - i1)
    # trendline = y1 + slope * (x - i1)

    # ap = mpf.make_addplot(trendline, color="black", width=2)

    # ----------------------------
    # Plot
    # ----------------------------
    mpf.plot(
        clean_df,
        type="candle",
        style="yahoo",
 #       addplot=ap,
        title=f"{symbol} — 1 Year Daily Chart",
        figsize=(16, 9),
        savefig=f"charts/{symbol}_1y.png",
        tight_layout=True,
    )
