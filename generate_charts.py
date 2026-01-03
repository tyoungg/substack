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
# Configuration
# ----------------------------
os.makedirs("charts", exist_ok=True)

with open("symbols.yaml", "r") as f:
    config = yaml.safe_load(f)
    symbols = config["symbols"]

# ----------------------------
# Helper Functions
# ----------------------------
def get_company_name(symbol):
    """Get company name from yfinance, fallback to symbol if unavailable"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return info.get('longName') or info.get('shortName') or info.get('name') or symbol
    except:
        return symbol

def pattern_annotation(pattern):
    """
    Converts pattern dict into Substack-ready text.
    Includes breakout confirmation and confidence ranking.
    """
    if not pattern:
        return "No significant pattern detected."

    score = pattern.get("score", 0)
    confidence = (
        "High" if score >= 0.8 else
        "Moderate" if score >= 0.65 else
        "Low"
    )

    text = f"**{pattern['type'].replace('_', ' ').title()}** detected "

    breakout = pattern.get("breakout", {})
    if breakout.get("confirmed"):
        text += (
            f"with a confirmed breakout "
            f"{breakout.get('bars_after', '?')} bars later "
            f"on {breakout.get('volume_ratio', '?')}× average volume. "
        )
    else:
        text += "but without a confirmed breakout yet. "

    text += f"Overall confidence: **{confidence}**."
    return text

# ----------------------------
# Pattern Detection Class
# ----------------------------
class PatternDetector:
    """
    Detects common technical patterns:
      - Head & Shoulders
      - Double Top/Bottom
      - Triangles
      - Flags/Pennants
      - Cup & Handle
      - Price Channels

    Each detection returns a dictionary with:
      - 'type': pattern type
      - indices for plotting
      - 'score': confidence 0–1
      - optional breakout info
    """

    def __init__(self, df):
        self.df = df
        self.closes = df['Close'].values
        self.highs = df['High'].values
        self.lows = df['Low'].values
        self.volumes = df['Volume'].values if 'Volume' in df.columns else np.ones(len(df))

    # ------------------------
    # Helper for peak/trough detection
    # ------------------------
    def find_peaks_troughs(self, prominence=0.02):
        price_range = np.max(self.closes) - np.min(self.closes)
        min_prominence = prominence * price_range
        peaks, _ = find_peaks(self.highs, prominence=min_prominence, distance=5)
        troughs, _ = find_peaks(-self.lows, prominence=min_prominence, distance=5)
        return peaks, troughs

    def _calculate_trendline_slope(self, indices, values):
        if len(indices) < 2:
            return 0
        x = np.array(indices).reshape(-1,1)
        y = values[indices]
        return LinearRegression().fit(x,y).coef_[0]

    # ------------------------
    # Pattern Detectors
    # ------------------------
    def detect_head_shoulders(self):
        peaks, _ = self.find_peaks_troughs()
        if len(peaks) < 3:
            return None

        for i in range(len(peaks)-2):
            left, head, right = peaks[i:i+3]
            if (self.highs[head] > self.highs[left] and 
                self.highs[head] > self.highs[right] and
                abs(self.highs[left]-self.highs[right]) < 0.05*self.highs[head]):

                # Score based on head height vs shoulders
                shoulder_avg = (self.highs[left]+self.highs[right])/2
                score = min(1.0, (self.highs[head]-shoulder_avg)/self.highs[head]*2)
                return {
                    'type':'head_shoulders',
                    'left_shoulder':left,
                    'head':head,
                    'right_shoulder':right,
                    'neckline': min(self.lows[left:right+1]),
                    'score': score,
                    'breakout': self._check_breakout(min(self.lows[left:right+1]), right)
                }
        return None

    def detect_double_top_bottom(self):
        peaks, troughs = self.find_peaks_troughs()
        # Double Top
        if len(peaks)>=2:
            for i in range(len(peaks)-1):
                p1,p2 = peaks[i],peaks[i+1]
                if abs(self.highs[p1]-self.highs[p2]) < 0.03*self.highs[p1]:
                    score = 0.7 + 0.3*(len(peaks)/10)  # Example scoring
                    return {'type':'double_top','peak1':p1,'peak2':p2,
                            'support':min(self.lows[p1:p2+1]),'score':score,
                            'breakout': self._check_breakout(min(self.lows[p1:p2+1]),p2)}
        # Double Bottom
        if len(troughs)>=2:
            for i in range(len(troughs)-1):
                t1,t2 = troughs[i],troughs[i+1]
                if abs(self.lows[t1]-self.lows[t2])<0.03*self.lows[t1]:
                    score = 0.7 + 0.3*(len(troughs)/10)
                    return {'type':'double_bottom','trough1':t1,'trough2':t2,
                            'resistance':max(self.highs[t1:t2+1]),'score':score,
                            'breakout': self._check_breakout(max(self.highs[t1:t2+1]),t2)}
        return None

    def detect_triangle(self, window=20):
        if len(self.closes)<window:
            return None
        peaks, troughs = self.find_peaks_troughs()
        recent_peaks = peaks[peaks>=len(self.closes)-window]
        recent_troughs = troughs[troughs>=len(self.closes)-window]
        if len(recent_peaks)>=2 and len(recent_troughs)>=2:
            peak_slope = self._calculate_trendline_slope(recent_peaks,self.highs)
            trough_slope = self._calculate_trendline_slope(recent_troughs,self.lows)
            triangle_type='symmetrical_triangle'
            if abs(peak_slope)<0.001 and trough_slope>0:
                triangle_type='ascending_triangle'
            elif peak_slope<0 and abs(trough_slope)<0.001:
                triangle_type='descending_triangle'
            score = min(1.0, len(recent_peaks)/window + len(recent_troughs)/window)
            return {'type':triangle_type,'peaks':recent_peaks,'troughs':recent_troughs,'score':score}
        return None

    def detect_flag_pennant(self, window=15):
        if len(self.closes)<window*2:
            return None
        recent=self.closes[-window:]
        prev=self.closes[-window*2:-window]
        pole_move=(recent[0]-prev[0])/prev[0]
        volatility=np.std(recent)/np.mean(recent)
        if abs(pole_move)>0.05 and volatility<0.03:
            score=min(1.0, abs(pole_move)*5)
            return {'type':'flag' if pole_move>0 else 'bear_flag',
                    'flag_start':len(self.closes)-window,'pole_start':len(self.closes)-window*2,
                    'score':score,'breakout': self._check_breakout(recent[-1], len(self.closes)-1)}
        return None

    def detect_cup_handle(self,min_cup_length=30,handle_ratio=0.3):
        if len(self.closes)<min_cup_length+10:
            return None
        for start_idx in range(len(self.closes)-min_cup_length):
            end_idx=start_idx+min_cup_length
            cup_data=self.closes[start_idx:end_idx]
            bottom_idx=start_idx+np.argmin(cup_data)
            left_side=cup_data[:bottom_idx-start_idx]
            right_side=cup_data[bottom_idx-start_idx:]
            if len(left_side)<5 or len(right_side)<5:
                continue
            if abs(self.closes[start_idx]-self.closes[end_idx-1])>0.05*self.closes[start_idx]:
                continue
            handle_start=end_idx
            max_handle_length=int(min_cup_length*handle_ratio)
            if handle_start+max_handle_length>len(self.closes):
                continue
            cup_bottom=self.closes[bottom_idx]
            cup_top=max(self.closes[start_idx],self.closes[end_idx-1])
            upper_third=cup_bottom+0.67*(cup_top-cup_bottom)
            for handle_end in range(handle_start+5,min(handle_start+max_handle_length,len(self.closes))):
                handle_data=self.closes[handle_start:handle_end]
                if np.min(handle_data)>upper_third and handle_data[-1]<handle_data[0] and abs(handle_data[-1]-handle_data[0])<0.03*handle_data[0]:
                    score=0.7 + 0.3*((end_idx-start_idx)/min_cup_length)
                    return {'type':'cup_handle','cup_start':start_idx,'cup_bottom':bottom_idx,'cup_end':end_idx-1,
                            'handle_start':handle_start,'handle_end':handle_end,'rim_level':cup_top,'score':score,
                            'breakout': self._check_breakout(cup_top, handle_end)}
        return None

    def _check_breakout(self, level, idx):
        """
        Simple breakout check: did price close above/below the level in last 3 bars?
        Returns dict with 'confirmed', 'bars_after', 'volume_ratio'
        """
        if idx+1 >= len(self.closes):
            return {'confirmed':False}
        lookahead = self.closes[idx+1:idx+4]
        if len(lookahead)==0:
            return {'confirmed':False}
        confirmed = any((lookahead>level) if self.closes[idx]>level else (lookahead<level))
        bars_after = np.argmax((lookahead>level) if self.closes[idx]>level else (lookahead<level))+1 if confirmed else None
        volume_ratio = self.volumes[idx+1:idx+4].mean()/np.mean(self.volumes) if confirmed else None
        return {'confirmed':confirmed,'bars_after':bars_after,'volume_ratio':volume_ratio}

# ----------------------------
# Plotting Functions
# ----------------------------
def plot_simple_chart(df,symbol,company_name):
    """Plain candlestick chart without patterns"""
    fig, axes = mpf.plot(
        df,
        type='candle',
        style='yahoo',
        title=f"{company_name} ({symbol}) — 1 Year Daily Chart",
        figsize=(16,9),
        returnfig=True,
        tight_layout=True
    )
    fig.savefig(f"charts/{symbol}_1y.png",dpi=300,bbox_inches='tight')
    plt.close(fig)

def plot_with_pattern(df,symbol,company_name,pattern):
    """Candlestick chart with top pattern plotted"""
    addplots=[]
    legend_items=[]
    if not pattern:
        plot_simple_chart(df,symbol,company_name)
        return

    # Example: for H&S or double top, draw lines
    if pattern['type']=='head_shoulders':
        neckline=np.full(len(df),np.nan)
        ls,rs=pattern['left_shoulder'],pattern['right_shoulder']
        for i in range(max(0,ls-5),min(rs+15,len(df))):
            neckline[i]=pattern['neckline']
        addplots.append(mpf.make_addplot(neckline,color='red',linestyle='--',width=2))
        legend_items.append('Head & Shoulders')

    if pattern['type']=='double_top':
        peak1,peak2=pattern['peak1'],pattern['peak2']
        resistance=np.full(len(df),np.nan)
        for i in range(max(0,peak1-5),min(peak2+15,len(df))):
            resistance[i]=max(df['High'].iloc[peak1],df['High'].iloc[peak2])
        addplots.append(mpf.make_addplot(resistance,color='blue',linestyle='--',width=2))
        legend_items.append('Double Top')

    # More plotting for other patterns can be added similarly...

    fig, axes = mpf.plot(
        df,
        type='candle',
        style='yahoo',
        addplot=addplots if addplots else None,
        title=f"{company_name} ({symbol}) — Top Pattern: {pattern['type'].replace('_',' ').title()}",
        figsize=(16,9),
        returnfig=True,
        tight_layout=True
    )

    if legend_items:
        legend_patches=[mpatches.Patch(color='red',label=legend_items[0])] # Extend for multiple
        axes[0].legend(handles=legend_patches,loc='upper left',framealpha=0.7)
    fig.savefig(f"charts/{symbol}_1y_patterns.png",dpi=300,bbox_inches='tight')
    plt.close(fig)

# ----------------------------
# Main Loop
# ----------------------------
for symbol in symbols:
    print(f"Processing {symbol}...")
    company_name=get_company_name(symbol)
    print(f"Company: {company_name}")

    df=yf.download(symbol,period='1y',interval='1d',auto_adjust=False,progress=False)
    df=df.dropna()
    if isinstance(df.columns,pd.MultiIndex):
        df.columns=df.columns.get_level_values(0)

    clean_df=pd.DataFrame({
        'Open':df['Open'].to_numpy().astype('float64'),
        'High':df['High'].to_numpy().astype('float64'),
        'Low':df['Low'].to_numpy().astype('float64'),
        'Close':df['Close'].to_numpy().astype('float64'),
        'Volume':df['Volume'].to_numpy().astype('float64')
    }, index=pd.to_datetime(df.index))

    # Plain chart
    plot_simple_chart(clean_df,symbol,company_name)
    print(f"{symbol}: Plain chart saved.")

    # Detect patterns
    detector=PatternDetector(clean_df)
    detected_patterns=[
        detector.detect_head_shoulders(),
        detector.detect_double_top_bottom(),
        detector.detect_triangle(),
        detector.detect_flag_pennant(),
        detector.detect_cup_handle()
    ]
    # Keep only non-None
    detected_patterns=[p for p in detected_patterns if p]

    # Rank patterns by score
    top_pattern=max(detected_patterns,key=lambda x:x.get('score',0)) if detected_patterns else None

    # Plot top pattern chart
    plot_with_pattern(clean_df,symbol,company_name,top_pattern)

    # Generate Substack annotation
    annotation=pattern_annotation(top_pattern)
    print(f"{symbol} annotation: {annotation}")
