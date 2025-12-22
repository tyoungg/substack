import yfinance as yf
import yaml
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

# Load symbols
with open("symbols.yaml", "r") as f:
    symbols = yaml.safe_load(f)["symbols"]

os.makedirs("charts", exist_ok=True)

def find_trendline_points(df):
    closes = df["Close"].values
    idx_low = np.argmin(closes)

    for i in range(idx_low + 5, len(closes)):
        if closes[i] > closes[idx_low]:
            return idx_low, i
    return idx_low, idx_low + 1

for symbol in symbols:
#    df = yf.download(symbol, period="1y", interval="1d")
    df = yf.download(
    symbol,
    period="1y",
    interval="1d",
    auto_adjust=False,
    progress=False
)

    df.dropna(inplace=True)
    # Ensure mplfinance-compatible dtypes
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = df[col].astype(float)


    i1, i2 = find_trendline_points(df)

#    x = np.arange(len(df))
#    y1 = df["Close"].iloc[i1]
#    y2 = df["Close"].iloc[i2]
#    slope = (y2 - y1) / (i2 - i1)
#    trendline = y1 + slope * (x - i1)

    # --- Trendline calculation (robust numpy-only math) ---
    x = np.arange(len(df), dtype=float)

#    y1 = float(df["Close"].iloc[i1])
#    y2 = float(df["Close"].iloc[i2])
    y1 = df["Close"].iloc[i1].item()
    y2 = df["Close"].iloc[i2].item()

    slope = (y2 - y1) / float(i2 - i1)
    trendline = y1 + slope * (x - float(i1))


    add_plot = mpf.make_addplot(trendline, color="black", width=2)

    mpf.plot(
        df,
        type="candle",
        style="yahoo",
        addplot=add_plot,
        title=f"{symbol} — 1 Year Daily Chart",
        figsize=(16, 9),
        savefig=f"charts/{symbol}_1y.png",
        tight_layout=True
    )
