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
df = yf.download(
    symbol,
    period="1y",
    interval="1d",
    auto_adjust=False,
    progress=False
)

df = df.dropna()

# --- Rebuild OHLC dataframe to guarantee float dtypes ---
ohlc = df[["Open", "High", "Low", "Close"]].to_numpy(dtype=float)

clean_df = mpf.make_marketcolors(up='g', down='r')
clean_df = df.copy()

for col in ["Open", "High", "Low", "Close"]:
    clean_df[col] = clean_df[col].to_numpy(dtype=float)

# --- Trendline logic ---
closes = clean_df["Close"].to_numpy()
i1 = closes.argmin()

i2 = None
for i in range(i1 + 5, len(closes)):
    if closes[i] > closes[i1]:
        i2 = i
        break

if i2 is None:
    i2 = i1 + 1

x = np.arange(len(closes), dtype=float)

y1 = closes[i1]
y2 = closes[i2]

slope = (y2 - y1) / (i2 - i1)
trendline = y1 + slope * (x - i1)

ap = mpf.make_addplot(trendline, color="black", width=2)

mpf.plot(
    clean_df,
    type="candle",
    style="yahoo",
    addplot=ap,
    title=f"{symbol} — 1 Year Daily Chart",
    figsize=(16, 9),
    savefig=f"charts/{symbol}_1y.png",
    tight_layout=True
)
