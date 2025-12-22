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
    df = yf.download(symbol, period="1y", interval="1d")
    df.dropna(inplace=True)

    i1, i2 = find_trendline_points(df)

    x = np.arange(len(df))
    y1 = df["Close"].iloc[i1]
    y2 = df["Close"].iloc[i2]
    slope = (y2 - y1) / (i2 - i1)
    trendline = y1 + slope * (x - i1)

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
