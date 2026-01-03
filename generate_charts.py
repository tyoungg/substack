# generate_charts.py
# ----------------------------
# Full auto-chart pipeline: plain & pattern-enhanced charts
# Includes top-scoring pattern detection, breakout confirmation, and Substack-ready annotation
# ----------------------------

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

# ----------------------------
# CONFIGURATION
# ----------------------------
SYMBOLS_FILE = "symbols.yaml"
CHART_DIR = "charts"
TOP_N_PATTERNS = 1            # Only plot top-scoring pattern(s)
ENABLE_VOLUME_BREAKOUT = True  # Use volume to confirm breakout

os.makedirs(CHART_DIR, exist_ok=True)

# ----------------------------
# Load symbols
# ----------------------------
with open(SYMBOLS_FILE, "r") as f:
    config = yaml.safe_load(f)
    symbols = config.get("symbols", [])

# ----------------------------
# Helper: company name from yfinance
# ----------------------------
def get_company_name(symbol):
    """Get company name with fallbacks"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return info.get('longName') or info.get('shortName') or info.get('name') or symbol
    except:
        return symbol

# ----------------------------
# Pattern Detection Class
# ----------------------------
class PatternDetector:
    """Detect classic price patterns and rank them"""
    def __init__(self, df):
        self.df = df.copy()
        self.closes = df['Close'].values
        self.highs = df['High'].values
        self.lows = df['Low'].values
        self.volumes = df['Volume'].values if 'Volume' in df.columns else np.ones(len(df))
    
    # ----------------------------
    # Generic peaks/troughs
    # ----------------------------
    def find_peaks_troughs(self, prominence=0.02, distance=5):
        """Find significant peaks and troughs"""
        price_range = np.max(self.closes) - np.min(self.closes)
        min_prominence = prominence * price_range
        peaks, _ = find_peaks(self.highs, prominence=min_prominence, distance=distance)
        troughs, _ = find_peaks(-self.lows, prominence=min_prominence, distance=distance)
        return peaks, troughs

    # ----------------------------
    # Trendline calculation
    # ----------------------------
    def _calculate_trendline_slope(self, indices, values):
        if len(indices) < 2:
            return 0
        x = np.array(indices).reshape(-1, 1)
        y = values[indices]
        reg = LinearRegression().fit(x, y)
        return reg.coef_[0]

    # ----------------------------
    # Head & Shoulders
    # ----------------------------
    def detect_head_shoulders(self):
        """Detect classic Head & Shoulders"""
        peaks, _ = self.find_peaks_troughs()
        if len(peaks) < 3:
            return None
        # Evaluate recent 3 peaks
        for i in range(len(peaks) - 2):
            left, head, right = peaks[i:i+3]
            # Head higher than shoulders and shoulders roughly equal
            if (self.highs[head] > self.highs[left] and self.highs[head] > self.highs[right] and
                abs(self.highs[left]-self.highs[right]) < 0.05*self.highs[head]):
                # Score: symmetry + head prominence
                shoulder_diff = abs(self.highs[left]-self.highs[right])/self.highs[head]
                head_prominence = (self.highs[head]-min(self.lows[left:right+1]))/self.highs[head]
                score = (1-shoulder_diff)*0.4 + head_prominence*0.6
                return {"type":"head_shoulders", "left_shoulder":left, "head":head, "right_shoulder":right,
                        "neckline": min(self.lows[left:right+1]), "score":score}
        return None

    # ----------------------------
    # Double Top / Bottom
    # ----------------------------
    def detect_double_top_bottom(self):
        peaks, troughs = self.find_peaks_troughs()
        # Double Top
        if len(peaks) >= 2:
            for i in range(len(peaks)-1):
                p1, p2 = peaks[i], peaks[i+1]
                if abs(self.highs[p1]-self.highs[p2]) < 0.03*self.highs[p1]:
                    score = 1 - abs(self.highs[p1]-self.highs[p2])/self.highs[p1]
                    return {"type":"double_top", "peak1":p1, "peak2":p2, "support":min(self.lows[p1:p2+1]), "score":score}
        # Double Bottom
        if len(troughs) >= 2:
            for i in range(len(troughs)-1):
                t1, t2 = troughs[i], troughs[i+1]
                if abs(self.lows[t1]-self.lows[t2]) < 0.03*self.lows[t1]:
                    score = 1 - abs(self.lows[t1]-self.lows[t2])/self.lows[t1]
                    return {"type":"double_bottom", "trough1":t1, "trough2":t2, "resistance":max(self.highs[t1:t2+1]), "score":score}
        return None

    # ----------------------------
    # Simple breakout check
    # ----------------------------
    def confirm_breakout(self, pattern, bars_after=5, min_volume_ratio=1.5):
        """Check if pattern had breakout above/below pattern range"""
        end_idx = min(len(self.closes)-1, pattern.get("head", pattern.get("peak2", 0)) + bars_after)
        breakout = {"confirmed": False, "bars_after": bars_after, "volume_ratio": None}
        if end_idx < len(self.closes):
            # Bullish patterns
            if pattern['type'] in ['head_shoulders','double_bottom','ascending_triangle','bull_flag','cup_handle']:
                breakout_price = max(self.highs[pattern.get('head', 0):end_idx+1])
                if breakout_price > max(self.highs[pattern.get('head',0):pattern.get('right_shoulder',end_idx)+1]):
                    avg_vol = np.mean(self.volumes[pattern.get('head',0):end_idx])
                    current_vol = self.volumes[end_idx]
                    ratio = current_vol / avg_vol if avg_vol>0 else 0
                    if ratio >= min_volume_ratio or not ENABLE_VOLUME_BREAKOUT:
                        breakout['confirmed'] = True
                        breakout['volume_ratio'] = round(ratio,2)
            # Bearish patterns
            if pattern['type'] in ['double_top','bear_flag']:
                breakout_price = min(self.lows[pattern.get('peak2',0):end_idx+1])
                if breakout_price < min(self.lows[pattern.get('peak1',0):pattern.get('peak2',0)+1]):
                    avg_vol = np.mean(self.volumes[pattern.get('peak1',0):end_idx])
                    current_vol = self.volumes[end_idx]
                    ratio = current_vol / avg_vol if avg_vol>0 else 0
                    if ratio >= min_volume_ratio or not ENABLE_VOLUME_BREAKOUT:
                        breakout['confirmed'] = True
                        breakout['volume_ratio'] = round(ratio,2)
        pattern['breakout'] = breakout
        return pattern

# ----------------------------
# Substack annotation
# ----------------------------
def pattern_annotation(pattern):
    """Convert pattern dict to Substack-ready text"""
    score = pattern.get("score",0)
    confidence = "High" if score>=0.8 else "Moderate" if score>=0.65 else "Low"
    text = f"**{pattern['type'].replace('_',' ').title()}** detected "
    if "breakout" in pattern and pattern["breakout"]["confirmed"]:
        text += f"with confirmed breakout {pattern['breakout']['bars_after']} bars later on {pattern['breakout']['volume_ratio']}× avg volume. "
    else:
        text += "but no confirmed breakout yet. "
    text += f"Overall confidence: **{confidence}**."
    return text

# ----------------------------
# Plot plain chart
# ----------------------------
def plot_simple_chart(clean_df,symbol,company_name):
    fig, axes = mpf.plot(clean_df,
        type="candle",
        style="yahoo",
        title=f"{company_name} ({symbol}) — 1 Year Daily Chart",
        figsize=(16,9),
        returnfig=True,
        tight_layout=True
    )
    fig.savefig(f"{CHART_DIR}/{symbol}_1y.png",dpi=300,bbox_inches='tight')
    plt.close(fig)

# ----------------------------
# Plot chart with top pattern(s)
# ----------------------------
def plot_with_patterns(clean_df,symbol,company_name,patterns):
    addplots = []
    legend_items=[]
    legend_colors=[]
    for pattern in patterns:
        if pattern is None:
            continue
        # Draw a simple line to indicate pattern (neckline, resistance/support, trendline)
        line = np.full(len(clean_df), np.nan)
        start = 0
        end = len(clean_df)-1
        if pattern['type'] == 'head_shoulders':
            start = pattern['left_shoulder']
            end = pattern['right_shoulder']
            line[start:end+1] = pattern['neckline']
            addplots.append(mpf.make_addplot(line,color='red',linestyle='--',width=2))
            legend_items.append("Head & Shoulders")
            legend_colors.append('red')
        elif pattern['type']=='double_top':
            start=pattern['peak1']
            end=pattern['peak2']
            line[start:end+1]=max(clean_df['High'].iloc[start:end+1])
            addplots.append(mpf.make_addplot(line,color='blue',linestyle='--',width=2))
            legend_items.append("Double Top")
            legend_colors.append('blue')
        elif pattern['type']=='double_bottom':
            start=pattern['trough1']
            end=pattern['trough2']
            line[start:end+1]=min(clean_df['Low'].iloc[start:end+1])
            addplots.append(mpf.make_addplot(line,color='blue',linestyle='--',width=2))
            legend_items.append("Double Bottom")
            legend_colors.append('blue')
        # More patterns can be added similarly with their own visualization

    fig, axes = mpf.plot(clean_df,
        type="candle",
        style="yahoo",
        addplot=addplots if addplots else None,
        title=f"{company_name} ({symbol}) — 1 Year Daily Chart",
        figsize=(16,9),
        returnfig=True,
        tight_layout=True
    )
    if legend_items:
        legend_patches = [mpatches.Patch(color=c,label=l) for l,c in zip(legend_items,legend_colors)]
        axes[0].legend(handles=legend_patches, loc='upper left', framealpha=0.7, fontsize=9)
    fig.savefig(f"{CHART_DIR}/{symbol}_1y_patterns.png",dpi=300,bbox_inches='tight')
    plt.close(fig)

# ----------------------------
# MAIN LOOP
# ----------------------------
for symbol in symbols:
    print(f"Processing {symbol}...")
    company_name = get_company_name(symbol)
    print(f"Company: {company_name}")
    df = yf.download(symbol,period="1y",interval="1d",auto_adjust=False,progress=False)
    df = df.dropna()
    if isinstance(df.columns,pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ensure numeric
    clean_df = pd.DataFrame({
        "Open": df["Open"].astype(float).values,
        "High": df["High"].astype(float).values,
        "Low": df["Low"].astype(float).values,
        "Close": df["Close"].astype(float).values,
        "Volume": df["Volume"].astype(float).values
    }, index=pd.to_datetime(df.index))

    # ----------------------------
    # Generate plain chart (_1y.png)
    # ----------------------------
    plot_simple_chart(clean_df,symbol,company_name)
    print(f"{symbol}: Simple chart generated (_1y.png)")

    # ----------------------------
    # Detect top patterns
    # ----------------------------
    detector = PatternDetector(clean_df)
    raw_patterns = [
        detector.detect_head_shoulders(),
        detector.detect_double_top_bottom()
        # Can add triangle, flag, cup_handle, etc.
    ]
    # Filter None
    raw_patterns = [p for p in raw_patterns if p is not None]
    # Confirm breakout
    for p in raw_patterns:
        detector.confirm_breakout(p)
    # Select top N patterns by score
    top_patterns = sorted(raw_patterns,key=lambda x:x['score'],reverse=True)[:TOP_N_PATTERNS]
    for p in top_patterns:
        annotation = pattern_annotation(p)
        print(f"{symbol} pattern annotation: {annotation}")

    # ----------------------------
    # Generate pattern chart (_1y_patterns.png)
    # ----------------------------
    plot_with_patterns(clean_df,symbol,company_name,top_patterns)
    print(f"{symbol}: Pattern chart generated (_1y_patterns.png)")

print("All symbols processed.")
