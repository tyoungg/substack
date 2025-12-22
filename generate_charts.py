import os
import yaml
import numpy as np
import pandas as pd
import yfinance as yf
import mplfinance as mpf
from scipy.signal import find_peaks, find_peaks_cwt
from sklearn.linear_model import LinearRegression

# ----------------------------
# Enhanced Pattern Detection
# ----------------------------
class PatternDetector:
    def __init__(self, df):
        self.df = df
        self.closes = df['Close'].values
        self.highs = df['High'].values
        self.lows = df['Low'].values
        # Fix: Use numpy array directly
        self.volume = df.get('Volume', pd.Series(np.ones(len(df)), index=df.index)).values  

        
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
            
        # Look for recent 3-peak pattern
        for i in range(len(peaks) - 2):
            left, head, right = peaks[i:i+3]
            
            # Head should be higher than shoulders
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
        
        # Double Top
        if len(peaks) >= 2:
            for i in range(len(peaks) - 1):
                p1, p2 = peaks[i], peaks[i+1]
                if abs(self.highs[p1] - self.highs[p2]) < 0.03 * self.highs[p1]:
                    return {
                        'type': 'double_top',
                        'peak1': p1,
                        'peak2': p2,
                        'support': min(self.lows[p1:p2+1])
                    }
        
        # Double Bottom
        if len(troughs) >= 2:
            for i in range(len(troughs) - 1):
                t1, t2 = troughs[i], troughs[i+1]
                if abs(self.lows[t1] - self.lows[t2]) < 0.03 * self.lows[t1]:
                    return {
                        'type': 'double_bottom',
                        'trough1': t1,
                        'trough2': t2,
                        'resistance': max(self.highs[t1:t2+1])
                    }
        return None
    
    def detect_triangle(self, window=20):
        """Detect Triangle patterns"""
        if len(self.closes) < window:
            return None
            
        recent_data = self.closes[-window:]
        x = np.arange(len(recent_data))
        
        # Find upper and lower trendlines
        peaks, troughs = self.find_peaks_troughs()
        recent_peaks = peaks[peaks >= len(self.closes) - window]
        recent_troughs = troughs[troughs >= len(self.closes) - window]
        
        if len(recent_peaks) >= 2 and len(recent_troughs) >= 2:
            # Calculate slopes
            peak_slope = self._calculate_trendline_slope(recent_peaks, self.highs)
            trough_slope = self._calculate_trendline_slope(recent_troughs, self.lows)
            
            # Classify triangle type
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
    
    def detect_flag_pennant(self, window=15):
        """Detect Flag and Pennant patterns"""
        if len(self.closes) < window * 2:
            return None
            
        # Look for sharp move (pole) followed by consolidation
        recent = self.closes[-window:]
        prev_period = self.closes[-(window*2):-window]
        
        # Check for significant move (pole)
        pole_move = (recent[0] - prev_period[0]) / prev_period[0]
        if abs(pole_move) > 0.05:  # 5% move
            # Check if recent period is consolidating
            volatility = np.std(recent) / np.mean(recent)
            if volatility < 0.03:  # Low volatility = consolidation
                return {
                    'type': 'flag' if pole_move > 0 else 'bear_flag',
                    'pole_start': len(self.closes) - window * 2,
                    'flag_start': len(self.closes) - window,
                    'pole_move': pole_move
                }
        return None

# ----------------------------
# Enhanced plotting function
# ----------------------------
def plot_with_patterns(clean_df, symbol, patterns):
    """Plot chart with detected patterns"""
    addplots = []
    
    for pattern in patterns:
        if pattern is None:
            continue
            
        if pattern['type'] == 'head_shoulders':
            # Draw neckline
            neckline = np.full(len(clean_df), pattern['neckline'])
            addplots.append(mpf.make_addplot(neckline, color='red', linestyle='--', width=2))
            
        elif pattern['type'] in ['double_top', 'double_bottom']:
            # Draw support/resistance line
            level = pattern.get('support', pattern.get('resistance'))
            if level:
                line = np.full(len(clean_df), level)
                addplots.append(mpf.make_addplot(line, color='blue', linestyle='--', width=2))
                
        elif 'triangle' in pattern['type']:
            # Draw triangle trendlines
            peaks, troughs = pattern['peaks'], pattern['troughs']
            if len(peaks) >= 2 and len(troughs) >= 2:
                # Upper trendline
                x_peaks = peaks - (len(clean_df) - len(clean_df))
                y_peaks = clean_df['High'].iloc[peaks].values
                upper_line = np.interp(range(len(clean_df)), x_peaks, y_peaks)
                addplots.append(mpf.make_addplot(upper_line, color='green', width=2))
    
    # Plot with patterns
    mpf.plot(
        clean_df,
        type="candle",
        style="yahoo",
        addplot=addplots if addplots else None,
        title=f"{symbol} — 1 Year Daily Chart with Patterns",
        figsize=(16, 9),
        savefig=f"charts/{symbol}_1y_patterns.png",
        tight_layout=True,
    )

# ----------------------------
# Updated main loop
# ----------------------------
with open("symbols.yaml", "r") as f:
    symbols = yaml.safe_load(f)["symbols"]
os.makedirs("charts", exist_ok=True)

for symbol in symbols:
    df = yf.download(symbol, period="1y", interval="1d", auto_adjust=False, progress=False)
    df = df.dropna()
    
    # Fix MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    clean_df = pd.DataFrame({
        "Open": df["Open"].to_numpy().astype("float64").ravel(),
        "High": df["High"].to_numpy().astype("float64").ravel(),
        "Low": df["Low"].to_numpy().astype("float64").ravel(),
        "Close": df["Close"].to_numpy().astype("float64").ravel(),
    }, index=pd.to_datetime(df.index))
    
    # Detect patterns
    detector = PatternDetector(clean_df)
    patterns = [
        detector.detect_head_shoulders(),
        detector.detect_double_top_bottom(),
        detector.detect_triangle(),
        detector.detect_flag_pennant()
    ]
    
    # Print detected patterns
    detected = [p for p in patterns if p is not None]
    if detected:
        print(f"{symbol}: {[p['type'] for p in detected]}")
    
    # Plot with patterns
    plot_with_patterns(clean_df, symbol, patterns)
