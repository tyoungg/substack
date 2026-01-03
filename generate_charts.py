import os
from pathlib import Path
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

# ----------------------------
# Create charts directory if missing
# ----------------------------
Path("charts").mkdir(parents=True, exist_ok=True)

# ----------------------------
# Load symbols from YAML
# ----------------------------
with open("symbols.yaml", "r") as f:
    config = yaml.safe_load(f)
    symbols = config.get("symbols", [])

# ----------------------------
# Helper: Get company name from yfinance
# ----------------------------
def get_company_name(symbol):
    """Get company name from yfinance; fallback to symbol if unavailable"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return info.get("longName") or info.get("shortName") or info.get("name") or symbol
    except:
        return symbol

# ----------------------------
# Pattern Detector Class
# ----------------------------
class PatternDetector:
    """
    Enhanced pattern detection class.
    Includes:
    - Head & Shoulders
    - Double Top / Bottom
    - Triangles
    - Flags / Pennants
    - Cup & Handle
    - Price Channels
    """

    def __init__(self, df):
        self.df = df
        self.closes = df["Close"].values
        self.highs = df["High"].values
        self.lows = df["Low"].values
        self.volume = df["Volume"].values if "Volume" in df.columns else np.ones(len(df))

    # ----------------------------
    # Peak/trough detection
    # ----------------------------
    def find_peaks_troughs(self, prominence=0.02, distance=5):
        """Find significant peaks and troughs"""
        price_range = np.max(self.closes) - np.min(self.closes)
        min_prominence = prominence * price_range
        peaks, _ = find_peaks(self.highs, prominence=min_prominence, distance=distance)
        troughs, _ = find_peaks(-self.lows, prominence=min_prominence, distance=distance)
        return peaks, troughs

    # ----------------------------
    # Head & Shoulders detection
    # ----------------------------
    def detect_head_shoulders(self, shoulder_tolerance=0.05):
        """Detect Head and Shoulders pattern"""
        peaks, _ = self.find_peaks_troughs()
        if len(peaks) < 3:
            return None

        for i in range(len(peaks) - 2):
            left, head, right = peaks[i:i+3]
            # Head must be higher than shoulders
            if (
                self.highs[head] > self.highs[left]
                and self.highs[head] > self.highs[right]
                and abs(self.highs[left] - self.highs[right]) < shoulder_tolerance * self.highs[head]
            ):
                # Tunable: shoulder_tolerance controls symmetry requirement
                return {
                    "type": "head_shoulders",
                    "left_shoulder": left,
                    "head": head,
                    "right_shoulder": right,
                    "neckline": min(self.lows[left:right+1]),
                    "score": 0.75  # confidence placeholder
                }
        return None

    # ----------------------------
    # Double Top / Bottom detection
    # ----------------------------
    def detect_double_top_bottom(self, similarity_threshold=0.03):
        """Detect Double Top or Bottom"""
        peaks, troughs = self.find_peaks_troughs()

        # Double Top
        if len(peaks) >= 2:
            for i in range(len(peaks) - 1):
                p1, p2 = peaks[i], peaks[i + 1]
                if abs(self.highs[p1] - self.highs[p2]) < similarity_threshold * self.highs[p1]:
                    return {
                        "type": "double_top",
                        "peak1": p1,
                        "peak2": p2,
                        "support": min(self.lows[p1:p2+1]),
                        "score": 0.7
                    }

        # Double Bottom
        if len(troughs) >= 2:
            for i in range(len(troughs) - 1):
                t1, t2 = troughs[i], troughs[i + 1]
                if abs(self.lows[t1] - self.lows[t2]) < similarity_threshold * self.lows[t1]:
                    return {
                        "type": "double_bottom",
                        "trough1": t1,
                        "trough2": t2,
                        "resistance": max(self.highs[t1:t2+1]),
                        "score": 0.7
                    }
        return None

    # ----------------------------
    # Triangle detection
    # ----------------------------
    def detect_triangle(self, window=20, slope_tolerance=0.001):
        """Detect Triangle patterns"""
        if len(self.closes) < window:
            return None

        peaks, troughs = self.find_peaks_troughs()
        recent_peaks = peaks[peaks >= len(self.closes) - window]
        recent_troughs = troughs[troughs >= len(self.closes) - window]

        if len(recent_peaks) >= 2 and len(recent_troughs) >= 2:
            peak_slope = self._calculate_trendline_slope(recent_peaks, self.highs)
            trough_slope = self._calculate_trendline_slope(recent_troughs, self.lows)
            # slope_tolerance controls how flat trendlines must be for ascending/descending
            if abs(peak_slope) < slope_tolerance and trough_slope > 0:
                ttype = "ascending_triangle"
            elif peak_slope < 0 and abs(trough_slope) < slope_tolerance:
                ttype = "descending_triangle"
            elif peak_slope < 0 and trough_slope > 0:
                ttype = "symmetrical_triangle"
            else:
                return None
            return {"type": ttype, "peaks": recent_peaks, "troughs": recent_troughs, "score": 0.65}
        return None

    # ----------------------------
    # Trendline slope helper
    # ----------------------------
    def _calculate_trendline_slope(self, indices, values):
        if len(indices) < 2:
            return 0
        x = np.array(indices).reshape(-1, 1)
        y = values[indices]
        reg = LinearRegression().fit(x, y)
        return reg.coef_[0]

    # ----------------------------
    # Flag / Pennant detection
    # ----------------------------
    def detect_flag_pennant(self, window=15, pole_threshold=0.05, consolidation_vol=0.03):
        """Detect flag/pennant pattern"""
        if len(self.closes) < window * 2:
            return None
        recent = self.closes[-window:]
        prev_period = self.closes[-(window * 2):-window]
        pole_move = (recent[0] - prev_period[0]) / prev_period[0]

        if abs(pole_move) > pole_threshold:
            volatility = np.std(recent) / np.mean(recent)
            if volatility < consolidation_vol:
                return {
                    "type": "flag" if pole_move > 0 else "bear_flag",
                    "pole_start": len(self.closes) - window * 2,
                    "flag_start": len(self.closes) - window,
                    "pole_move": pole_move,
                    "score": 0.6
                }
        return None

    # ----------------------------
    # Cup & Handle detection
    # ----------------------------
    def detect_cup_handle(self, min_cup_length=30, handle_ratio=0.3):
        """Detect cup and handle pattern"""
        if len(self.closes) < min_cup_length + 10:
            return None
        for start_idx in range(len(self.closes) - min_cup_length):
            end_idx = start_idx + min_cup_length
            cup_data = self.closes[start_idx:end_idx]
            bottom_idx = start_idx + np.argmin(cup_data)
            left_side = cup_data[: bottom_idx - start_idx]
            right_side = cup_data[bottom_idx - start_idx :]
            if len(left_side) < 5 or len(right_side) < 5:
                continue
            cup_start_price = self.closes[start_idx]
            cup_end_price = self.closes[end_idx - 1]
            if abs(cup_start_price - cup_end_price) > 0.05 * cup_start_price:
                continue
            handle_start = end_idx
            max_handle_length = int(min_cup_length * handle_ratio)
            if handle_start + max_handle_length > len(self.closes):
                continue
            cup_bottom = self.closes[bottom_idx]
            cup_top = max(cup_start_price, cup_end_price)
            upper_third = cup_bottom + 0.67 * (cup_top - cup_bottom)
            for handle_end in range(handle_start + 5, min(handle_start + max_handle_length, len(self.closes))):
                handle_data = self.closes[handle_start:handle_end]
                if (
                    np.min(handle_data) > upper_third
                    and handle_data[-1] < handle_data[0]
                    and abs(handle_data[-1] - handle_data[0]) < 0.03 * handle_data[0]
                ):
                    return {
                        "type": "cup_handle",
                        "cup_start": start_idx,
                        "cup_bottom": bottom_idx,
                        "cup_end": end_idx - 1,
                        "handle_start": handle_start,
                        "handle_end": handle_end,
                        "rim_level": cup_top,
                        "score": 0.7,
                    }
        return None

    # ----------------------------
    # Price Channels detection
    # ----------------------------
    def detect_price_channels(self, min_touches=3, parallel_tolerance=0.02, lookback_period=60):
        """Detect horizontal, ascending, descending price channels"""
        start_idx = max(0, len(self.closes) - lookback_period)
        recent_closes = self.closes[start_idx:]
        recent_highs = self.highs[start_idx:]
        recent_lows = self.lows[start_idx:]
        temp_detector = PatternDetector.__new__(PatternDetector)
        temp_detector.closes = recent_closes
        temp_detector.highs = recent_highs
        temp_detector.lows = recent_lows
        peaks, troughs = temp_detector.find_peaks_troughs(prominence=0.015)
        if len(peaks) < min_touches or len(troughs) < min_touches:
            return None
        peaks = peaks + start_idx
        troughs = troughs + start_idx

        for i in range(len(peaks) - min_touches + 1):
            upper_points = peaks[i : i + min_touches]
            upper_slope = self._calculate_trendline_slope(upper_points, self.highs)
            upper_intercept = self.highs[upper_points[0]] - upper_slope * upper_points[0]
            for j in range(len(troughs) - min_touches + 1):
                lower_points = troughs[j : j + min_touches]
                lower_slope = self._calculate_trendline_slope(lower_points, self.lows)
                lower_intercept = self.lows[lower_points[0]] - lower_slope * lower_points[0]
                slope_diff = abs(upper_slope - lower_slope)
                avg_slope = abs(upper_slope + lower_slope) / 2
                if avg_slope == 0 or slope_diff / avg_slope < parallel_tolerance:
                    mid_point = (upper_points[-1] + lower_points[-1]) / 2
                    upper_level = upper_slope * mid_point + upper_intercept
                    lower_level = lower_slope * mid_point + lower_intercept
                    channel_type = (
                        "horizontal_channel"
                        if abs(upper_slope) < 0.01
                        else "ascending_channel"
                        if upper_slope > 0
                        else "descending_channel"
                    )
                    return {
                        "type": channel_type,
                        "upper_points": upper_points,
                        "lower_points": lower_points,
                        "upper_slope": upper_slope,
                        "upper_intercept": upper_intercept,
                        "lower_slope": lower_slope,
                        "lower_intercept": lower_intercept,
                        "channel_width": upper_level - lower_level,
                        "start_idx": max(min(upper_points[0], lower_points[0]), start_idx),
                        "end_idx": max(upper_points[-1], lower_points[-1]),
                        "lookback_start": start_idx,
                        "score": 0.7,
                    }

# ----------------------------
# Pattern annotation for Substack
# ----------------------------
def pattern_annotation(pattern):
    """
    Converts pattern dict into Substack-ready text.
    """
    if pattern is None:
        return None

    score = pattern.get("score", 0)
    confidence = (
        "High" if score >= 0.8 else "Moderate" if score >= 0.65 else "Low"
    )

    text = f"**{pattern['type'].replace('_', ' ').title()}** detected "

    if "breakout" in pattern and pattern["breakout"].get("confirmed"):
        text += (
            f"with a confirmed breakout "
            f"{pattern['breakout']['bars_after']} bars later "
            f"on {pattern['breakout']['volume_ratio']}× average volume. "
        )
    else:
        text += "but without a confirmed breakout yet. "

    text += f"Overall confidence: **{confidence}**."

    return text

# ----------------------------
# Chart plotting functions
# ----------------------------
def plot_simple_chart(df, symbol, company_name):
    """Plot clean chart without patterns"""
    fig, ax = mpf.plot(
        df,
        type="candle",
        style="yahoo",
        title=f"{company_name} ({symbol}) — 1 Year Daily Chart",
        figsize=(16, 9),
        returnfig=True,
        tight_layout=True,
    )
    output_path = f"charts/{symbol}_1y.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"{symbol}: Simple chart generated -> {output_path}")


# ----------------------------
# Main loop
# ----------------------------
for symbol in symbols:
    print(f"Processing {symbol}...")
    company_name = get_company_name(symbol)
    print(f"Company: {company_name}")

    # ----------------------------
    # Download 1 year daily data
    # ----------------------------
    df = yf.download(symbol, period="1y", interval="1d", progress=False)
    df = df.dropna()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Build mplfinance-safe DataFrame
    clean_df = pd.DataFrame(
        {
            "Open": df["Open"].to_numpy().astype("float64").ravel(),
            "High": df["High"].to_numpy().astype("float64").ravel(),
            "Low": df["Low"].to_numpy().astype("float64").ravel(),
            "Close": df["Close"].to_numpy().astype("float64").ravel(),
            "Volume": df["Volume"].to_numpy().astype("float64").ravel(),
        },
        index=pd.to_datetime(df.index),
    )

    # ----------------------------
    # Generate simple chart (no patterns)
    # ----------------------------
    plot_simple_chart(clean_df, symbol, company_name)

    # ----------------------------
    # Detect patterns
    # ----------------------------
    detector = PatternDetector(clean_df)
    patterns = [
        detector.detect_head_shoulders(),
        detector.detect_double_top_bottom(),
        detector.detect_triangle(),
        detector.detect_flag_pennant(),
        detector.detect_cup_handle(),
        detector.detect_price_channels(),
    ]

    # ----------------------------
    # Generate Substack-ready annotations
    # ----------------------------
    annotations = []
    for p in patterns:
        note = pattern_annotation(p)
        if note:
            annotations.append(note)
    if annotations:
        print(f"{symbol} Patterns Annotations:\n" + "\n".join(annotations))
    else:
        print(f"{symbol}: No patterns detected")

