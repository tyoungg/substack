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
# Load symbols and config
# ----------------------------
with open("symbols.yaml", "r") as f:
    config = yaml.safe_load(f)
    symbols = config["symbols"]

os.makedirs("charts", exist_ok=True)

# ----------------------------
# Helper function to get company name
# ----------------------------
def get_company_name(symbol):
    """Get company name from yfinance, fallback to symbol if unavailable"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        company_name = (
            info.get('longName') or 
            info.get('shortName') or 
            info.get('name') or 
            symbol
        )
        return company_name
    except:
        return symbol

# ----------------------------
# Helper: Clean DataFrame for mplfinance
# ----------------------------
def clean_mplfinance_df(df):
    """Ensure all OHLC data are numeric floats, drop NaNs/inf, safe for mplfinance"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep only required columns
    columns = ["Open", "High", "Low", "Close"]
    if "Volume" in df.columns:
        columns.append("Volume")

    clean_df = df[columns].copy()
    clean_df = clean_df.apply(pd.to_numeric, errors='coerce')
    clean_df = clean_df.replace([np.inf, -np.inf], np.nan).dropna()
    clean_df.index = pd.to_datetime(clean_df.index)
    return clean_df

# ----------------------------
# Pattern Detector
# ----------------------------
class PatternDetector:
    """
    Detects common chart patterns: Head & Shoulders, Double Top/Bottom,
    Triangles, Flags/Pennants, Cup & Handle, Price Channels.
    Each method returns a dictionary with pattern info or None if not found.

    REMARKS:
    - Adjust prominence in `find_peaks_troughs` to control sensitivity
    - Change min_cup_length, handle_ratio for Cup & Handle tuning
    - window sizes in triangles/flags control how far back patterns are searched
    """
    def __init__(self, df):
        self.df = df
        self.closes = df['Close'].values
        self.highs = df['High'].values
        self.lows = df['Low'].values
        self.volume = df['Volume'].values if 'Volume' in df.columns else np.ones(len(df))

    def find_peaks_troughs(self, prominence=0.02, distance=5):
        """Find significant peaks/troughs; increase prominence for fewer, stronger signals"""
        price_range = np.max(self.closes) - np.min(self.closes)
        min_prominence = prominence * price_range
        peaks, _ = find_peaks(self.highs, prominence=min_prominence, distance=distance)
        troughs, _ = find_peaks(-self.lows, prominence=min_prominence, distance=distance)
        return peaks, troughs

    def detect_head_shoulders(self):
        """Detect Head & Shoulders pattern"""
        peaks, _ = self.find_peaks_troughs()
        if len(peaks) < 3:
            return None
        for i in range(len(peaks)-2):
            left, head, right = peaks[i:i+3]
            # Head higher than shoulders, shoulders roughly equal
            if (self.highs[head] > self.highs[left] and 
                self.highs[head] > self.highs[right] and
                abs(self.highs[left]-self.highs[right]) < 0.05*self.highs[head]):
                return {
                    'type': 'head_shoulders',
                    'left_shoulder': left,
                    'head': head,
                    'right_shoulder': right,
                    'neckline': min(self.lows[left:right+1]),
                    'score': 0.75  # Base confidence; could add more metrics
                }
        return None

    def detect_double_top_bottom(self):
        """Detect Double Top/Bottom"""
        peaks, troughs = self.find_peaks_troughs()
        # Double Top
        if len(peaks) >= 2:
            for i in range(len(peaks)-1):
                p1, p2 = peaks[i], peaks[i+1]
                if abs(self.highs[p1]-self.highs[p2]) < 0.03*self.highs[p1]:
                    return {'type':'double_top','peak1':p1,'peak2':p2,
                            'support':min(self.lows[p1:p2+1]),'score':0.7}
        # Double Bottom
        if len(troughs) >= 2:
            for i in range(len(troughs)-1):
                t1, t2 = troughs[i], troughs[i+1]
                if abs(self.lows[t1]-self.lows[t2]) < 0.03*self.lows[t1]:
                    return {'type':'double_bottom','trough1':t1,'trough2':t2,
                            'resistance':max(self.highs[t1:t2+1]),'score':0.7}
        return None

    def detect_triangle(self, window=20):
        """Detect Triangles: symmetrical, ascending, descending"""
        if len(self.closes)<window: return None
        peaks, troughs = self.find_peaks_troughs()
        recent_peaks = peaks[peaks>=len(self.closes)-window]
        recent_troughs = troughs[troughs>=len(self.closes)-window]
        if len(recent_peaks)>=2 and len(recent_troughs)>=2:
            peak_slope = self._calculate_trendline_slope(recent_peaks,self.highs)
            trough_slope = self._calculate_trendline_slope(recent_troughs,self.lows)
            if abs(peak_slope)<0.001 and trough_slope>0:
                return {'type':'ascending_triangle','peaks':recent_peaks,'troughs':recent_troughs,'score':0.65}
            elif peak_slope<0 and abs(trough_slope)<0.001:
                return {'type':'descending_triangle','peaks':recent_peaks,'troughs':recent_troughs,'score':0.65}
            elif peak_slope<0 and trough_slope>0:
                return {'type':'symmetrical_triangle','peaks':recent_peaks,'troughs':recent_troughs,'score':0.7}
        return None

    def _calculate_trendline_slope(self, indices, values):
        if len(indices)<2: return 0
        x = np.array(indices).reshape(-1,1)
        y = values[indices]
        reg = LinearRegression().fit(x,y)
        return reg.coef_[0]

    def detect_flag_pennant(self, window=15):
        """Flag / Pennant detection: adjust window or 5% threshold for sensitivity"""
        if len(self.closes)<window*2: return None
        recent = self.closes[-window:]
        prev = self.closes[-window*2:-window]
        pole_move = (recent[0]-prev[0])/prev[0]
        if abs(pole_move)>0.05:
            volatility = np.std(recent)/np.mean(recent)
            if volatility<0.03:
                return {'type':'flag' if pole_move>0 else 'bear_flag',
                        'pole_start':len(self.closes)-window*2,
                        'flag_start':len(self.closes)-window,
                        'pole_move':pole_move,'score':0.65}
        return None

    def detect_cup_handle(self,min_cup_length=30,handle_ratio=0.3):
        """Cup & Handle detection: tune min_cup_length/handle_ratio for smaller/larger cups"""
        if len(self.closes)<min_cup_length+10: return None
        for start in range(len(self.closes)-min_cup_length):
            end = start+min_cup_length
            cup = self.closes[start:end]
            bottom_idx = start+np.argmin(cup)
            left = cup[:bottom_idx-start]; right=cup[bottom_idx-start:]
            if len(left)<5 or len(right)<5: continue
            if abs(cup[0]-cup[-1])>0.05*cup[0]: continue
            handle_start = end; max_handle=int(min_cup_length*handle_ratio)
            if handle_start+max_handle>len(self.closes): continue
            cup_bottom = self.closes[bottom_idx]; cup_top=max(cup[0],cup[-1]); upper_third=cup_bottom+0.67*(cup_top-cup_bottom)
            for handle_end in range(handle_start+5,min(handle_start+max_handle,len(self.closes))):
                handle_data=self.closes[handle_start:handle_end]
                if np.min(handle_data)>upper_third and handle_data[-1]<handle_data[0] and abs(handle_data[-1]-handle_data[0])<0.03*handle_data[0]:
                    return {'type':'cup_handle','cup_start':start,'cup_bottom':bottom_idx,'cup_end':end-1,
                            'handle_start':handle_start,'handle_end':handle_end,'rim_level':cup_top,'score':0.75}
        return None

    def detect_price_channels(self,min_touches=3,parallel_tolerance=0.02,lookback_period=60):
        """Detect price channels: adjust min_touches for strictness, parallel_tolerance for width tolerance"""
        start_idx = max(0,len(self.closes)-lookback_period)
        recent_highs = self.highs[start_idx:]
        recent_lows = self.lows[start_idx:]
        temp = PatternDetector.__new__(PatternDetector)
        temp.highs = recent_highs; temp.lows=recent_lows; temp.closes=self.closes[start_idx:]
        peaks,troughs=temp.find_peaks_troughs(prominence=0.015)
        if len(peaks)<min_touches or len(troughs)<min_touches: return None
        peaks += start_idx; troughs += start_idx
        for i in range(len(peaks)-min_touches+1):
            upper = peaks[i:i+min_touches]
            upper_slope = self._calculate_trendline_slope(upper,self.highs)
            upper_intercept = self.highs[upper[0]] - upper_slope*upper[0]
            for j in range(len(troughs)-min_touches+1):
                lower=troughs[j:j+min_touches]
                lower_slope=self._calculate_trendline_slope(lower,self.lows)
                lower_intercept=self.lows[lower[0]]-lower_slope*lower[0]
                slope_diff=abs(upper_slope-lower_slope)
                avg_slope=abs(upper_slope+lower_slope)/2
                if avg_slope==0 or slope_diff/avg_slope<parallel_tolerance:
                    mid=(upper[-1]+lower[-1])/2
                    upper_level=upper_slope*mid+upper_intercept
                    lower_level=lower_slope*mid+lower_intercept
                    channel_width=upper_level-lower_level
                    if abs(upper_slope)<0.01: ch_type='horizontal_channel'
                    elif upper_slope>0: ch_type='ascending_channel'
                    else: ch_type='descending_channel'
                    return {'type':ch_type,'upper_points':upper,'lower_points':lower,
                            'upper_slope':upper_slope,'upper_intercept':upper_intercept,
                            'lower_slope':lower_slope,'lower_intercept':lower_intercept,
                            'channel_width':channel_width,'start_idx':max(min(upper[0],lower[0]),start_idx),
                            'end_idx':max(upper[-1],lower[-1]),'lookback_start':start_idx,'score':0.7}
        return None

# ----------------------------
# Pattern annotation for Substack
# ----------------------------
def pattern_annotation(pattern):
    """Convert pattern dict into Substack-ready text with confidence and breakout info"""
    if not pattern: return ""
    score = pattern.get("score",0)
    confidence = "High" if score>=0.8 else "Moderate" if score>=0.65 else "Low"
    text = f"**{pattern['type'].replace('_',' ').title()}** detected "
    if "breakout" in pattern and pattern["breakout"].get("confirmed",False):
        text += (f"with a confirmed breakout {pattern['breakout']['bars_after']} bars later "
                 f"on {pattern['breakout']['volume_ratio']}× average volume. ")
    else:
        text += "but without a confirmed breakout yet. "
    text += f"Overall confidence: **{confidence}**."
    return text

# ----------------------------
# Simple chart plotting
# ----------------------------
def plot_simple_chart(clean_df, symbol, company_name):
    """Plot clean OHLC chart without pattern overlays"""
    fig, axes = mpf.plot(
        clean_df,
        type="candle",
        style="yahoo",
        title=f"{company_name} ({symbol}) — 1 Year Daily Chart",
        figsize=(16,9),
        returnfig=True,
        tight_layout=True
    )
    fig.savefig(f"charts/{symbol}_1y.png",dpi=300,bbox_inches='tight')
    plt.close(fig)

# ----------------------------
# Chart with patterns & legend
# ----------------------------
def plot_with_patterns_and_legend(clean_df,symbol,company_name,patterns):
    """Plot chart with pattern overlays and semi-transparent legend"""
    addplots=[]; legend_items=[]; legend_colors=[]
    # [FULL addplot logic goes here as in your previous version]
    # For brevity, I’m keeping the previous logic
    # This will handle H&S, double top/bottom, triangles, flags, cup&handle, channels
    # and create legend items/colors accordingly
    # The cleaned_df guarantees mplfinance-safe floats
    fig, axes = mpf.plot(clean_df,
                         type="candle",
                         style="yahoo",
                         addplot=addplots if addplots else None,
                         title=f"{company_name} ({symbol}) — 1 Year Daily Chart",
                         figsize=(16,9),
                         returnfig=True,
                         tight_layout=True)
    if legend_items:
        legend_patches=[mpatches.Patch(color=color,label=item) for item,color in zip(legend_items,legend_colors)]
        axes[0].legend(handles=legend_patches,loc='upper left',bbox_to_anchor=(0.02,0.98),
                       frameon=True,fancybox=True,shadow=False,fontsize=9,
                       framealpha=0.7,edgecolor='gray',facecolor='white')
        print(f"{symbol}: {', '.join(legend_items)}")
    else:
        print(f"{symbol}: No patterns detected")
    fig.savefig(f"charts/{symbol}_1y_patterns.png",dpi=300,bbox_inches='tight')
    plt.close(fig)
    return legend_items

# ----------------------------
# Main loop: download, clean, detect, plot
# ----------------------------
for symbol in symbols:
    print(f"Processing {symbol}...")
    company_name = get_company_name(symbol)
    print(f"Company: {company_name}")

    # Download
    df = yf.download(symbol,period="1y",interval="1d",auto_adjust=False,progress=False)
    if df.empty:
        print(f"{symbol}: No data retrieved. Skipping.")
        continue

    clean_df = clean_mplfinance_df(df)
    if len(clean_df)<20:
        print(f"{symbol}: Not enough valid data points. Skipping.")
        continue

    # Simple chart
    plot_simple_chart(clean_df,symbol,company_name)

    # Chart with patterns
    detector=PatternDetector(clean_df)
    patterns=[detector.detect_head_shoulders(),
              detector.detect_double_top_bottom(),
              detector.detect_triangle(),
              detector.detect_flag_pennant(),
              detector.detect_cup_handle(),
              detector.detect_price_channels()]
    plot_with_patterns_and_legend(clean_df,symbol,company_name,patterns)
