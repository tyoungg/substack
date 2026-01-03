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
            self.volume = np.ones(len(df))  # fallback to ones if volume missing
    
    # ------------------------
    # Peaks/Troughs detection
    # ------------------------
    def find_peaks_troughs(self, prominence=0.02, distance=5):
        """
        Find significant peaks and troughs in price.
        prominence: fraction of total price range to qualify as a peak/trough
        distance: minimum bars between peaks
        - Increase prominence to detect only major peaks (more selective)
        - Decrease distance to detect smaller swing points
        """
        price_range = np.max(self.closes) - np.min(self.closes)
        min_prominence = prominence * price_range
        peaks, _ = find_peaks(self.highs, prominence=min_prominence, distance=distance)
        troughs, _ = find_peaks(-self.lows, prominence=min_prominence, distance=distance)
        return peaks, troughs
    
    # ------------------------
    # Head & Shoulders
    # ------------------------
    def detect_head_shoulders(self):
        """
        Detect head & shoulders pattern:
        - Head higher than shoulders
        - Shoulders approximately equal
        - Returns dict with indices and neckline
        - Tune the shoulder similarity tolerance for sensitivity
        """
        peaks, _ = self.find_peaks_troughs()
        if len(peaks) < 3:
            return None
        for i in range(len(peaks)-2):
            left, head, right = peaks[i:i+3]
            if (self.highs[head] > self.highs[left] and 
                self.highs[head] > self.highs[right] and
                abs(self.highs[left] - self.highs[right]) < 0.05 * self.highs[head]):  # shoulder similarity tolerance
                return {
                    'type': 'head_shoulders',
                    'left_shoulder': left,
                    'head': head,
                    'right_shoulder': right,
                    'neckline': min(self.lows[left:right+1]),
                    'score': 0.8  # Example scoring; could be dynamic
                }
        return None
    
    # ------------------------
    # Double Top/Bottom
    # ------------------------
    def detect_double_top_bottom(self):
        """
        Detect double tops/bottoms:
        - Compare peak/trough heights within tolerance
        - Tuning: 0.03 = 3% difference allowed; decrease for stricter detection
        """
        peaks, troughs = self.find_peaks_troughs()
        # Double Top
        if len(peaks) >= 2:
            for i in range(len(peaks)-1):
                p1, p2 = peaks[i], peaks[i+1]
                if abs(self.highs[p1] - self.highs[p2]) < 0.03 * self.highs[p1]:
                    return {
                        'type': 'double_top',
                        'peak1': p1,
                        'peak2': p2,
                        'support': min(self.lows[p1:p2+1]),
                        'score': 0.75
                    }
        # Double Bottom
        if len(troughs) >= 2:
            for i in range(len(troughs)-1):
                t1, t2 = troughs[i], troughs[i+1]
                if abs(self.lows[t1] - self.lows[t2]) < 0.03 * self.lows[t1]:
                    return {
                        'type': 'double_bottom',
                        'trough1': t1,
                        'trough2': t2,
                        'resistance': max(self.highs[t1:t2+1]),
                        'score': 0.75
                    }
        return None
    
    # ------------------------
    # Triangles
    # ------------------------
    def detect_triangle(self, window=20):
        """
        Detect ascending, descending, symmetrical triangles
        - window: lookback for recent pattern
        - Tune slope thresholds for sensitivity
        """
        if len(self.closes) < window:
            return None
        peaks, troughs = self.find_peaks_troughs()
        recent_peaks = peaks[peaks >= len(self.closes) - window]
        recent_troughs = troughs[troughs >= len(self.closes) - window]
        if len(recent_peaks) >= 2 and len(recent_troughs) >= 2:
            peak_slope = self._calculate_trendline_slope(recent_peaks, self.highs)
            trough_slope = self._calculate_trendline_slope(recent_troughs, self.lows)
            # classify
            if abs(peak_slope) < 0.001 and trough_slope > 0:
                return {'type':'ascending_triangle','peaks':recent_peaks,'troughs':recent_troughs,'score':0.7}
            elif peak_slope < 0 and abs(trough_slope) < 0.001:
                return {'type':'descending_triangle','peaks':recent_peaks,'troughs':recent_troughs,'score':0.7}
            elif peak_slope < 0 and trough_slope > 0:
                return {'type':'symmetrical_triangle','peaks':recent_peaks,'troughs':recent_troughs,'score':0.7}
        return None
    
    # ------------------------
    # Trendline slope helper
    # ------------------------
    def _calculate_trendline_slope(self, indices, values):
        if len(indices) < 2:
            return 0
        x = np.array(indices).reshape(-1,1)
        y = values[indices]
        reg = LinearRegression().fit(x,y)
        return reg.coef_[0]
    
    # ------------------------
    # Flags/Pennants
    # ------------------------
    def detect_flag_pennant(self, window=15):
        """
        Detect flag/bear flag:
        - window: bars for consolidation
        - Tune volatility threshold (<0.03) for consolidation sensitivity
        - Tune pole_move threshold (>0.05) for sharp move
        """
        if len(self.closes) < window*2:
            return None
        recent = self.closes[-window:]
        prev = self.closes[-(window*2):-window]
        pole_move = (recent[0] - prev[0]) / prev[0]
        if abs(pole_move) > 0.05:
            volatility = np.std(recent)/np.mean(recent)
            if volatility < 0.03:
                return {'type':'flag' if pole_move>0 else 'bear_flag',
                        'pole_start': len(self.closes)-window*2,
                        'flag_start': len(self.closes)-window,
                        'pole_move': pole_move,
                        'score':0.7}
        return None
    
    # ------------------------
    # Cup & Handle
    # ------------------------
    def detect_cup_handle(self, min_cup_length=30, handle_ratio=0.3):
        """
        Detect U-shaped cup and handle:
        - min_cup_length: minimum bars for cup
        - handle_ratio: max handle length relative to cup
        - Tune upper_third % for handle region (~0.67)
        """
        if len(self.closes) < min_cup_length+10:
            return None
        for start in range(len(self.closes)-min_cup_length):
            end = start+min_cup_length
            cup_data = self.closes[start:end]
            bottom_idx = start+np.argmin(cup_data)
            left = cup_data[:bottom_idx-start]
            right = cup_data[bottom_idx-start:]
            if len(left)<5 or len(right)<5:
                continue
            cup_start_price = self.closes[start]
            cup_end_price = self.closes[end-1]
            if abs(cup_start_price-cup_end_price)>0.05*cup_start_price:
                continue
            handle_start = end
            max_handle_length = int(min_cup_length*handle_ratio)
            if handle_start+max_handle_length>len(self.closes):
                continue
            cup_bottom = self.closes[bottom_idx]
            cup_top = max(cup_start_price,cup_end_price)
            upper_third = cup_bottom + 0.67*(cup_top-cup_bottom)
            for handle_end in range(handle_start+5,min(handle_start+max_handle_length,len(self.closes))):
                handle_data = self.closes[handle_start:handle_end]
                if (np.min(handle_data)>upper_third and 
                    handle_data[-1]<handle_data[0] and
                    abs(handle_data[-1]-handle_data[0])<0.03*handle_data[0]):
                    return {'type':'cup_handle',
                            'cup_start':start,
                            'cup_bottom':bottom_idx,
                            'cup_end':end-1,
                            'handle_start':handle_start,
                            'handle_end':handle_end,
                            'rim_level':cup_top,
                            'score':0.75}
        return None
    
    # ------------------------
    # Price Channels
    # ------------------------
    def detect_price_channels(self, min_touches=3, parallel_tolerance=0.02, lookback_period=60):
        """
        Detect horizontal/ascending/descending channels:
        - min_touches: minimum number of peaks/troughs to define channel
        - parallel_tolerance: allowed slope difference ratio
        - lookback_period: recent bars to analyze
        """
        start_idx = max(0,len(self.closes)-lookback_period)
        recent_closes = self.closes[start_idx:]
        recent_highs = self.highs[start_idx:]
        recent_lows = self.lows[start_idx:]
        temp_detector = PatternDetector.__new__(PatternDetector)
        temp_detector.closes = recent_closes
        temp_detector.highs = recent_highs
        temp_detector.lows = recent_lows
        peaks, troughs = temp_detector.find_peaks_troughs(prominence=0.015)
        if len(peaks)<min_touches or len(troughs)<min_touches:
            return None
        peaks = peaks+start_idx
        troughs = troughs+start_idx
        for i in range(len(peaks)-min_touches+1):
            upper_points = peaks[i:i+min_touches]
            upper_slope = self._calculate_trendline_slope(upper_points,self.highs)
            upper_intercept = self.highs[upper_points[0]]-upper_slope*upper_points[0]
            for j in range(len(troughs)-min_touches+1):
                lower_points = troughs[j:j+min_touches]
                lower_slope = self._calculate_trendline_slope(lower_points,self.lows)
                lower_intercept = self.lows[lower_points[0]]-lower_slope*lower_points[0]
                slope_diff = abs(upper_slope-lower_slope)
                avg_slope = abs(upper_slope+lower_slope)/2
                if avg_slope==0 or slope_diff/avg_slope<parallel_tolerance:
                    mid_point = (upper_points[-1]+lower_points[-1])/2
                    upper_level = upper_slope*mid_point+upper_intercept
                    lower_level = lower_slope*mid_point+lower_intercept
                    channel_width = upper_level-lower_level
                    if abs(upper_slope)<0.01:
                        channel_type='horizontal_channel'
                    elif upper_slope>0:
                        channel_type='ascending_channel'
                    else:
                        channel_type='descending_channel'
                    return {'type':channel_type,
                            'upper_points':upper_points,
                            'lower_points':lower_points,
                            'upper_slope':upper_slope,
                            'upper_intercept':upper_intercept,
                            'lower_slope':lower_slope,
                            'lower_intercept':lower_intercept,
                            'channel_width':channel_width,
                            'start_idx':max(min(upper_points[0],lower_points[0]),start_idx),
                            'end_idx':max(upper_points[-1],lower_points[-1]),
                            'lookback_start':start_idx,
                            'score':0.7}

# ----------------------------
# Substack-ready pattern annotation
# ----------------------------
def pattern_annotation(pattern):
    """
    Converts pattern dict into Substack-ready text with confidence
    """
    if pattern is None:
        return "No pattern detected."
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
# Simple plotting function
# ----------------------------
def plot_simple_chart(df, symbol, company_name):
    fig, axes = mpf.plot(
        df,type="candle",style="yahoo",
        title=f"{company_name} ({symbol}) — 1 Year Daily Chart",
        figsize=(16,9),
        returnfig=True,
        tight_layout=True
    )
    fig.savefig(f"charts/{symbol}_1y.png",dpi=300,bbox_inches='tight')
    plt.close(fig)

# ----------------------------
# Plotting with patterns and legends
# ----------------------------
def plot_with_patterns_and_legend(df,symbol,company_name,patterns):
    addplots=[]
    legend_items=[]
    legend_colors=[]
    for pattern in patterns:
        if pattern is None:
            continue
        # --- Head & Shoulders ---
        if pattern['type']=='head_shoulders':
            neckline=np.full(len(df),np.nan)
            left, right = pattern['left_shoulder'], pattern['right_shoulder']
            for i in range(max(0,left-5),min(right+15,len(df))):
                neckline[i]=pattern['neckline']
            addplots.append(mpf.make_addplot(neckline,color='red',linestyle='--',width=2))
            legend_items.append("Head & Shoulders"); legend_colors.append('red')
        # --- Double Top ---
        elif pattern['type']=='double_top':
            resistance=np.full(len(df),np.nan)
            p1,p2=pattern['peak1'],pattern['peak2']
            for i in range(max(0,p1-5),min(p2+15,len(df))):
                resistance[i]=max(df['High'].iloc[p1],df['High'].iloc[p2])
            addplots.append(mpf.make_addplot(resistance,color='blue',linestyle='--',width=2))
            legend_items.append("Double Top"); legend_colors.append('blue')
        # --- Double Bottom ---
        elif pattern['type']=='double_bottom':
            support=np.full(len(df),np.nan)
            t1,t2=pattern['trough1'],pattern['trough2']
            for i in range(max(0,t1-5),min(t2+15,len(df))):
                support[i]=min(df['Low'].iloc[t1],df['Low'].iloc[t2])
            addplots.append(mpf.make_addplot(support,color='blue',linestyle='--',width=2))
            legend_items.append("Double Bottom"); legend_colors.append('blue')
        # --- Triangles ---
        elif 'triangle' in pattern['type']:
            peaks,troughs=pattern['peaks'],pattern['troughs']
            upper_line=np.full(len(df),np.nan)
            slope=(df['High'].iloc[peaks[-1]]-df['High'].iloc[peaks[0]])/(peaks[-1]-peaks[0])
            intercept=df['High'].iloc[peaks[0]]-slope*peaks[0]
            for i in range(peaks[0],peaks[-1]+1): upper_line[i]=slope*i+intercept
            addplots.append(mpf.make_addplot(upper_line,color='green',width=2))
            lower_line=np.full(len(df),np.nan)
            slope=(df['Low'].iloc[troughs[-1]]-df['Low'].iloc[troughs[0]])/(troughs[-1]-troughs[0])
            intercept=df['Low'].iloc[troughs[0]]-slope*troughs[0]
            for i in range(troughs[0],troughs[-1]+1): lower_line[i]=slope*i+intercept
            addplots.append(mpf.make_addplot(lower_line,color='green',width=2))
            legend_items.append(pattern['type'].replace('_',' ').title()); legend_colors.append('green')
        # --- Flags ---
        elif pattern['type'] in ['flag','bear_flag']:
            start=pattern['flag_start']
            top=np.full(len(df),np.nan)
            bottom=np.full(len(df),np.nan)
            for i in range(start,min(start+20,len(df))):
                top[i]=df['High'].iloc[start:].max()
                bottom[i]=df['Low'].iloc[start:].min()
            addplots.append(mpf.make_addplot(top,color='purple',width=2))
            addplots.append(mpf.make_addplot(bottom,color='purple',width=2))
            legend_items.append(pattern['type'].replace('_',' ').title()); legend_colors.append('purple')
        # --- Cup & Handle ---
        elif pattern['type']=='cup_handle':
            rim=np.full(len(df),np.nan)
            rim[pattern['cup_end']:pattern['handle_end']+1]=pattern['rim_level']
            addplots.append(mpf.make_addplot(rim,color='orange',linestyle='--',width=2))
            legend_items.append('Cup & Handle'); legend_colors.append('orange')
        # --- Channels ---
        elif 'channel' in pattern['type']:
            upper_line=np.full(len(df),np.nan)
            lower_line=np.full(len(df),np.nan)
            for i in range(pattern['start_idx'],pattern['end_idx']+1):
                upper_line[i]=pattern['upper_slope']*i+pattern['upper_intercept']
                lower_line[i]=pattern['lower_slope']*i+pattern['lower_intercept']
            addplots.append(mpf.make_addplot(upper_line,color='brown',width=2))
            addplots.append(mpf.make_addplot(lower_line,color='brown',width=2))
            legend_items.append(pattern['type'].replace('_',' ').title()); legend_colors.append('brown')
    
    fig, axes = mpf.plot(
        df,type="candle",style="yahoo",
        title=f"{company_name} ({symbol}) — Pattern Overlay",
        figsize=(16,9),
        addplot=addplots,
        returnfig=True,
        tight_layout=True
    )
    
    # Custom legend
    for i, item in enumerate(legend_items):
        axes[0].legend_.remove() if axes[0].legend_ else None
        axes[0].add_patch(mpatches.Patch(color=legend_colors[i],label=item))
    
    fig.savefig(f"charts/{symbol}_patterns.png",dpi=300,bbox_inches='tight')
    plt.close(fig)

# ----------------------------
# Main loop
# ----------------------------
for symbol in symbols:
    print(f"Processing {symbol}...")
    df=yf.download(symbol,period="1y",interval="1d")
    if df.empty:
        print(f"No data for {symbol}. Skipping.")
        continue
    company_name = get_company_name(symbol)
    plot_simple_chart(df,symbol,company_name)
    
    detector = PatternDetector(df)
    patterns = [
        detector.detect_head_shoulders(),
        detector.detect_double_top_bottom(),
        detector.detect_triangle(),
        detector.detect_flag_pennant(),
        detector.detect_cup_handle(),
        detector.detect_price_channels()
    ]
    
    # Print pattern annotations (Substack-ready)
    for pat in patterns:
        print(pattern_annotation(pat))
    
    # Plot patterns
    plot_with_patterns_and_legend(df,symbol,company_name,patterns)

print("All charts generated in 'charts/' directory.")
