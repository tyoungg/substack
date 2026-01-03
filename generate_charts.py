import os
import yaml
import numpy as np
import pandas as pd
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression

# ============================================================
# CONFIG + SYMBOL LOADING
# ============================================================

with open("symbols.yaml", "r") as f:
    config = yaml.safe_load(f)
    symbols = config["symbols"]

os.makedirs("charts", exist_ok=True)

# ============================================================
# COMPANY NAME HELPER
# ============================================================

def get_company_name(symbol):
    """
    Fetches company name from Yahoo Finance.
    Falls back safely to symbol.
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return (
            info.get("longName")
            or info.get("shortName")
            or info.get("name")
            or symbol
        )
    except Exception:
        return symbol

# ============================================================
# BREAKOUT CONFIRMATION (NEW)
# ============================================================

def confirm_breakout(
    df,
    level,
    direction="up",
    lookahead=5,
    atr_mult=0.4,
    volume_mult=1.2
):
    """
    Confirms breakout after a pattern completes.

    ------------------ TUNING GUIDE ------------------
    lookahead:
        ↑ higher = safer confirmation, later signals
        ↓ lower  = earlier but noisier

    atr_mult:
        ↑ higher = filters weak / fake breaks
        ↓ lower  = more sensitive

    volume_mult:
        ↑ higher = requires strong participation
        ↓ lower  = allows thin-volume breakouts
    --------------------------------------------------
    """

    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    volume = df["Volume"].values if "Volume" in df else np.ones(len(df))

    atr = np.mean(highs[-20:] - lows[-20:])
    avg_vol = np.mean(volume[-20:])

    for bars_after in range(1, lookahead + 1):
        price = closes[-bars_after]
        vol = volume[-bars_after]

        if direction == "up":
            if price > level + atr_mult * atr and vol > volume_mult * avg_vol:
                return {
                    "confirmed": True,
                    "bars_after": bars_after,
                    "volume_ratio": round(vol / avg_vol, 2),
                }
        else:
            if price < level - atr_mult * atr and vol > volume_mult * avg_vol:
                return {
                    "confirmed": True,
                    "bars_after": bars_after,
                    "volume_ratio": round(vol / avg_vol, 2),
                }

    return {"confirmed": False}

# ============================================================
# SUBSTACK-READY ANNOTATION (NEW)
# ============================================================

def pattern_annotation(pattern):
    """
    Converts a pattern dict into Substack-ready markdown text.
    Safe against missing fields.
    """

    score = pattern.get("score", 0)

    confidence = (
        "High" if score >= 0.8 else
        "Moderate" if score >= 0.65 else
        "Low"
    )

    name = pattern.get("type", "Pattern").replace("_", " ").title()
    text = f"**{name}** detected "

    breakout = pattern.get("breakout")

    if isinstance(breakout, dict) and breakout.get("confirmed"):
        text += (
            f"with a confirmed breakout "
            f"{breakout['bars_after']} bars later "
            f"on {breakout['volume_ratio']}× average volume. "
        )
    else:
        text += "but without a confirmed breakout yet. "

    text += f"Overall confidence: **{confidence}**."

    return text

# ============================================================
# PATTERN ENRICHMENT (NEW – NON-BREAKING)
# ============================================================

def enrich_pattern(pattern, df):
    """
    Adds:
      - score
      - breakout confirmation
      - Substack annotation

    WITHOUT breaking plotting or detection logic.
    """

    if pattern is None:
        return None

    # ------------------ BASE SCORES ------------------
    # Tune these to change relative confidence by pattern type
    base_scores = {
        "head_shoulders": 0.70,
        "double_top": 0.68,
        "double_bottom": 0.68,
        "ascending_triangle": 0.66,
        "descending_triangle": 0.66,
        "symmetrical_triangle": 0.65,
        "flag": 0.72,
        "bear_flag": 0.72,
        "cup_handle": 0.75,
        "ascending_channel": 0.65,
        "descending_channel": 0.65,
        "horizontal_channel": 0.60,
    }

    ptype = pattern["type"]
    pattern["score"] = base_scores.get(ptype, 0.60)

    # ------------------ BREAKOUT LEVEL ------------------
    if "neckline" in pattern:
        level, direction = pattern["neckline"], "down"
    elif "support" in pattern:
        level, direction = pattern["support"], "down"
    elif "resistance" in pattern:
        level, direction = pattern["resistance"], "up"
    elif "rim_level" in pattern:
        level, direction = pattern["rim_level"], "up"
    else:
        pattern["annotation"] = pattern_annotation(pattern)
        return pattern

    pattern["breakout"] = confirm_breakout(df, level, direction)

    if pattern["breakout"]["confirmed"]:
        pattern["score"] += 0.12  # reward confirmation

    pattern["annotation"] = pattern_annotation(pattern)
    return pattern

# ============================================================
# PATTERN DETECTOR (YOUR ORIGINAL, DOCUMENTED)
# ============================================================

class PatternDetector:
    """
    Detects classical technical patterns.

    ------------------ GLOBAL QUALITY LEVERS ------------------
    - prominence: filters insignificant swings
    - distance: prevents micro-patterns
    - window sizes: control speed vs reliability
    ----------------------------------------------------------
    """

    def __init__(self, df):
        self.df = df
        self.closes = df["Close"].values
        self.highs = df["High"].values
        self.lows = df["Low"].values
        self.volume = df["Volume"].values if "Volume" in df else np.ones(len(df))

    def find_peaks_troughs(self, prominence=0.02, distance=5):
        """
        prominence ↑ = fewer, stronger patterns
        prominence ↓ = more, noisier patterns
        """
        price_range = np.max(self.closes) - np.min(self.closes)
        min_prom = prominence * price_range

        peaks, _ = find_peaks(self.highs, prominence=min_prom, distance=distance)
        troughs, _ = find_peaks(-self.lows, prominence=min_prom, distance=distance)
        return peaks, troughs

    def detect_head_shoulders(self):
        peaks, _ = self.find_peaks_troughs()
        if len(peaks) < 3:
            return None

        for i in range(len(peaks) - 2):
            l, h, r = peaks[i:i+3]
            if (
                self.highs[h] > self.highs[l]
                and self.highs[h] > self.highs[r]
                and abs(self.highs[l] - self.highs[r]) < 0.05 * self.highs[h]
            ):
                return {
                    "type": "head_shoulders",
                    "left_shoulder": l,
                    "head": h,
                    "right_shoulder": r,
                    "neckline": min(self.lows[l:r+1]),
                }
        return None

    def detect_double_top_bottom(self):
        peaks, troughs = self.find_peaks_troughs()

        for i in range(len(peaks) - 1):
            p1, p2 = peaks[i], peaks[i+1]
            if abs(self.highs[p1] - self.highs[p2]) < 0.03 * self.highs[p1]:
                return {
                    "type": "double_top",
                    "peak1": p1,
                    "peak2": p2,
                    "support": min(self.lows[p1:p2+1]),
                }

        for i in range(len(troughs) - 1):
            t1, t2 = troughs[i], troughs[i+1]
            if abs(self.lows[t1] - self.lows[t2]) < 0.03 * self.lows[t1]:
                return {
                    "type": "double_bottom",
                    "trough1": t1,
                    "trough2": t2,
                    "resistance": max(self.highs[t1:t2+1]),
                }

        return None

    def detect_triangle(self, window=20):
        if len(self.closes) < window:
            return None

        peaks, troughs = self.find_peaks_troughs()
        recent_peaks = peaks[peaks >= len(self.closes) - window]
        recent_troughs = troughs[troughs >= len(self.closes) - window]

        if len(recent_peaks) >= 2 and len(recent_troughs) >= 2:
            ps = self._slope(recent_peaks, self.highs)
            ts = self._slope(recent_troughs, self.lows)

            if abs(ps) < 0.001 and ts > 0:
                t = "ascending_triangle"
            elif ps < 0 and abs(ts) < 0.001:
                t = "descending_triangle"
            elif ps < 0 and ts > 0:
                t = "symmetrical_triangle"
            else:
                return None

            return {"type": t, "peaks": recent_peaks, "troughs": recent_troughs}

        return None

    def _slope(self, idx, vals):
        if len(idx) < 2:
            return 0
        reg = LinearRegression().fit(idx.reshape(-1, 1), vals[idx])
        return reg.coef_[0]

    def detect_flag_pennant(self, window=15):
        if len(self.closes) < window * 2:
            return None

        move = (self.closes[-window] - self.closes[-window*2]) / self.closes[-window*2]
        vol = np.std(self.closes[-window:]) / np.mean(self.closes[-window:])

        if abs(move) > 0.05 and vol < 0.03:
            return {
                "type": "flag" if move > 0 else "bear_flag",
                "flag_start": len(self.closes) - window,
            }

        return None

    def detect_cup_handle(self):
        return None  # preserved placeholder

    def detect_price_channels(self):
        return None  # preserved placeholder

# ============================================================
# MAIN LOOP (UNCHANGED STRUCTURE)
# ============================================================

for symbol in symbols:
    print(f"\nProcessing {symbol}...")

    company_name = get_company_name(symbol)
    print(f"Company: {company_name}")

    df = yf.download(symbol, period="1y", interval="1d", progress=False)
    df = df.dropna()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    clean_df = pd.DataFrame(
        {
            "Open": df["Open"].astype(float),
            "High": df["High"].astype(float),
            "Low": df["Low"].astype(float),
            "Close": df["Close"].astype(float),
            "Volume": df["Volume"].astype(float),
        },
        index=pd.to_datetime(df.index),
    )

    detector = PatternDetector(clean_df)

    patterns = [
        detector.detect_head_shoulders(),
        detector.detect_double_top_bottom(),
        detector.detect_triangle(),
        detector.detect_flag_pennant(),
    ]

    patterns = [enrich_pattern(p, clean_df) for p in patterns if p]

    for p in patterns:
        print("  ↳", p["annotation"])
