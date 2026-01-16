import os
import yaml
import numpy as np
import pandas as pd
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression

# ----------------------------
# Load symbol and config
# ----------------------------
with open("single_symbol.yaml", "r") as f:
    config = yaml.safe_load(f)
    symbol = config["symbol"]

os.makedirs("charts", exist_ok=True)

# ----------------------------
# Helper function to get company name
# ----------------------------
def get_company_name(symbol):
    """Get company name from yfinance, fallback to symbol if unavailable"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        company_name = info.get('longName') or info.get('shortName') or symbol
        return company_name
    except:
        return symbol

# ----------------------------
# Enhanced Pattern Detection
# ----------------------------
class PatternDetector:
    def __init__(self, df):
        self.df = df
        self.closes = df['Close'].values
        self.highs = df['High'].values
        self.lows = df['Low'].values
        if 'Volume' in df.columns:
            self.volume = df['Volume'].values
        else:
            self.volume = np.ones(len(df))

    def find_peaks_troughs(self, prominence=0.02):
        """Find significant peaks and troughs"""
        price_range = np.max(self.closes) - np.min(self.closes)
        min_prominence = prominence * price_range

        peaks, _ = find_peaks(self.highs, prominence=min_prominence, distance=5)
        troughs, _ = find_peaks(-self.lows, prominence=min_prominence, distance=5)
        return peaks, troughs

    def detect_head_shoulders(self):
        """Detect Head and Shoulders pattern"""
        peaks, _ = self.find_peaks_troughs()
        if len(peaks) < 3:
            return None
        for i in range(len(peaks) - 2):
            left, head, right = peaks[i:i+3]
            if (self.highs[head] > self.highs[left] and
                self.highs[head] > self.highs[right] and
                abs(self.highs[left] - self.highs[right]) < 0.05 * self.highs[head]):
                return {
                    'type': 'head_shoulders',
                    'left_shoulder': left,
                    'head': head,
                    'right_shoulder': right,
                    'neckline': min(self.lows[left:right+1])
                }
        return None

    def detect_double_top_bottom(self):
        """Detect Double Top/Bottom patterns"""
        peaks, troughs = self.find_peaks_troughs()
        if len(peaks) >= 2:
            for i in range(len(peaks) - 1):
                p1, p2 = peaks[i], peaks[i+1]
                if abs(self.highs[p1] - self.highs[p2]) < 0.03 * self.highs[p1]:
                    return {'type': 'double_top', 'peak1': p1, 'peak2': p2}
        if len(troughs) >= 2:
            for i in range(len(troughs) - 1):
                t1, t2 = troughs[i], troughs[i+1]
                if abs(self.lows[t1] - self.lows[t2]) < 0.03 * self.lows[t1]:
                    return {'type': 'double_bottom', 'trough1': t1, 'trough2': t2}
        return None

    def detect_triangle(self, window=20):
        """Detect Triangle patterns"""
        if len(self.closes) < window:
            return None
        recent_data = self.closes[-window:]
        x = np.arange(len(recent_data))
        peaks, troughs = self.find_peaks_troughs()
        recent_peaks = peaks[peaks >= len(self.closes) - window]
        recent_troughs = troughs[troughs >= len(self.closes) - window]
        if len(recent_peaks) >= 2 and len(recent_troughs) >= 2:
            peak_slope = self._calculate_trendline_slope(recent_peaks, self.highs)
            trough_slope = self._calculate_trendline_slope(recent_troughs, self.lows)
            if abs(peak_slope) < 0.001 and trough_slope > 0:
                return {'type': 'ascending_triangle', 'peaks': recent_peaks, 'troughs': recent_troughs}
            elif peak_slope < 0 and abs(trough_slope) < 0.001:
                return {'type': 'descending_triangle', 'peaks': recent_peaks, 'troughs': recent_troughs}
            elif peak_slope < 0 and trough_slope > 0:
                return {'type': 'symmetrical_triangle', 'peaks': recent_peaks, 'troughs': recent_troughs}
        return None

    def _calculate_trendline_slope(self, indices, values):
        """Calculate slope of trendline through given points"""
        if len(indices) < 2:
            return 0
        x = np.array(indices).reshape(-1, 1)
        y = values[indices]
        reg = LinearRegression().fit(x, y)
        return reg.coef_[0]

# ----------------------------
# Plotting with patterns and legend
# ----------------------------
def plot_with_patterns_and_legend(clean_df, symbol, company_name, patterns):
    """Plot chart with colored pattern lines and custom x-axis"""
    addplots = []
    legend_handles = []

    for pattern in patterns:
        if pattern is None:
            continue
        if pattern['type'] == 'head_shoulders':
            left_shoulder, right_shoulder = pattern['left_shoulder'], pattern['right_shoulder']
            neckline = np.full(len(clean_df), np.nan)
            for i in range(max(0, left_shoulder - 5), min(right_shoulder + 15, len(clean_df))):
                neckline[i] = pattern['neckline']
            addplots.append(mpf.make_addplot(neckline, color='red', linestyle='--', width=2))
            legend_handles.append(plt.Line2D([], [], color='red', linestyle='--', label='Head & Shoulders'))
        elif pattern['type'] == 'double_top':
            peak1, peak2 = pattern['peak1'], pattern['peak2']
            resistance_line = np.full(len(clean_df), np.nan)
            for i in range(max(0, peak1 - 5), min(peak2 + 15, len(clean_df))):
                resistance_line[i] = max(clean_df['High'].iloc[peak1], clean_df['High'].iloc[peak2])
            addplots.append(mpf.make_addplot(resistance_line, color='blue', linestyle='--', width=2))
            legend_handles.append(plt.Line2D([], [], color='blue', linestyle='--', label='Double Top'))
        elif pattern['type'] == 'double_bottom':
            trough1, trough2 = pattern['trough1'], pattern['trough2']
            support_line = np.full(len(clean_df), np.nan)
            for i in range(max(0, trough1 - 5), min(trough2 + 15, len(clean_df))):
                support_line[i] = min(clean_df['Low'].iloc[trough1], clean_df['Low'].iloc[trough2])
            addplots.append(mpf.make_addplot(support_line, color='blue', linestyle='--', label='Double Bottom'))
        elif 'triangle' in pattern['type']:
            peaks, troughs = pattern['peaks'], pattern['troughs']
            triangle_name = pattern['type'].replace('_', ' ').title()
            pattern_start = min(peaks[0] if len(peaks) > 0 else len(clean_df), troughs[0] if len(troughs) > 0 else len(clean_df))
            pattern_end = max(peaks[-1] if len(peaks) > 0 else 0, troughs[-1] if len(troughs) > 0 else 0)
            if len(peaks) >= 2:
                upper_line = np.full(len(clean_df), np.nan)
                slope = (clean_df['High'].iloc[peaks[-1]] - clean_df['High'].iloc[peaks[0]]) / (peaks[-1] - peaks[0])
                intercept = clean_df['High'].iloc[peaks[0]] - slope * peaks[0]
                for i in range(max(0, pattern_start - 5), min(pattern_end + 15, len(clean_df))):
                    upper_line[i] = slope * i + intercept
                addplots.append(mpf.make_addplot(upper_line, color='green', width=2))
            if len(troughs) >= 2:
                lower_line = np.full(len(clean_df), np.nan)
                slope = (clean_df['Low'].iloc[troughs[-1]] - clean_df['Low'].iloc[troughs[0]]) / (troughs[-1] - troughs[0])
                intercept = clean_df['Low'].iloc[troughs[0]] - slope * troughs[0]
                for i in range(max(0, pattern_start - 5), min(pattern_end + 15, len(clean_df))):
                    lower_line[i] = slope * i + intercept
                addplots.append(mpf.make_addplot(lower_line, color='green', width=2))
            legend_handles.append(plt.Line2D([], [], color='green', linestyle='-', label=triangle_name))

    fig, axes = mpf.plot(
        clean_df, type="candle", style="yahoo",
        addplot=addplots if addplots else None,
        title=f"{company_name} ({symbol}) — 1 Year Daily Chart",
        figsize=(16, 9), returnfig=True, tight_layout=True,
    )

    if legend_handles:
        axes[0].legend(
            handles=legend_handles, loc='upper left', bbox_to_anchor=(0.02, 0.98),
            frameon=True, fancybox=True, shadow=False,
            fontsize=9, framealpha=0.7,
            edgecolor='gray', facecolor='white'
        )
        print(f"{symbol}: {', '.join([h.get_label() for h in legend_handles])}")
    else:
        print(f"{symbol}: No patterns detected")

    fig.savefig(f"charts/{symbol}_1y_patterns.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    return [h.get_label() for h in legend_handles]

# ----------------------------
# Main execution
# ----------------------------
def main():
    print(f"Processing {symbol}...")
    company_name = get_company_name(symbol)
    print(f"Company: {company_name}")

    df = yf.download(
        symbol, period="1y", interval="1d",
        auto_adjust=False, progress=False,
    )
    df = df.dropna()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    clean_df = pd.DataFrame({
        "Open": df["Open"].to_numpy().astype("float64").ravel(),
        "High": df["High"].to_numpy().astype("float64").ravel(),
        "Low": df["Low"].to_numpy().astype("float64").ravel(),
        "Close": df["Close"].to_numpy().astype("float64").ravel(),
    }, index=pd.to_datetime(df.index))

    detector = PatternDetector(clean_df)
    patterns = [
        detector.detect_head_shoulders(),
        detector.detect_double_top_bottom(),
        detector.detect_triangle(),
    ]

    plot_with_patterns_and_legend(clean_df, symbol, company_name, patterns)

if __name__ == "__main__":
    main()
