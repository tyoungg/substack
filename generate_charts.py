import os
import yaml
import numpy as np
import pandas as pd
import yfinance as yf
import mplfinance as mpf
from scipy.signal import find_peaks, find_peaks_cwt
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ----------------------------
# Load symbols
# ----------------------------
with open("symbols.yaml", "r") as f:
    symbols = yaml.safe_load(f)["symbols"]
os.makedirs("charts", exist_ok=True)

# ----------------------------
# Enhanced Pattern Detection
# ----------------------------
class PatternDetector:
    def __init__(self, df):
        self.df = df
        self.closes = df['Close'].values
        self.highs = df['High'].values
        self.lows = df['Low'].values
        # Fix: Handle volume properly
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

    def detect_cup_handle(self, min_cup_length=30, handle_ratio=0.3):
        """Detect Cup and Handle pattern"""
        if len(self.closes) < min_cup_length + 10:
            return None
        
        # Look for cup formation (U-shaped bottom)
        for start_idx in range(len(self.closes) - min_cup_length):
            end_idx = start_idx + min_cup_length
            cup_data = self.closes[start_idx:end_idx]
            
            # Find the deepest point (bottom of cup)
            bottom_idx = start_idx + np.argmin(cup_data)
            
            # Check if it's U-shaped (not V-shaped)
            left_side = cup_data[:bottom_idx - start_idx]
            right_side = cup_data[bottom_idx - start_idx:]
            
            if len(left_side) < 5 or len(right_side) < 5:
                continue
                
            # Cup should have similar start and end levels
            cup_start_price = self.closes[start_idx]
            cup_end_price = self.closes[end_idx - 1]
            
            if abs(cup_start_price - cup_end_price) > 0.05 * cup_start_price:
                continue
                
            # Look for handle after the cup
            handle_start = end_idx
            max_handle_length = int(min_cup_length * handle_ratio)
            
            if handle_start + max_handle_length > len(self.closes):
                continue
                
            # Handle should be in upper third of cup range
            cup_bottom = self.closes[bottom_idx]
            cup_top = max(cup_start_price, cup_end_price)
            upper_third = cup_bottom + 0.67 * (cup_top - cup_bottom)
            
            # Find handle formation
            for handle_end in range(handle_start + 5, min(handle_start + max_handle_length, len(self.closes))):
                handle_data = self.closes[handle_start:handle_end]
                
                # Handle should drift slightly downward but stay in upper third
                if (np.min(handle_data) > upper_third and 
                    handle_data[-1] < handle_data[0] and  # Slight downward drift
                    abs(handle_data[-1] - handle_data[0]) < 0.03 * handle_data[0]):  # Small drift
                    
                    return {
                        'type': 'cup_handle',
                        'cup_start': start_idx,
                        'cup_bottom': bottom_idx,
                        'cup_end': end_idx - 1,
                        'handle_start': handle_start,
                        'handle_end': handle_end,
                        'rim_level': cup_top
                    }
        
        return None

    def detect_price_channels(self, min_touches=3, parallel_tolerance=0.02, lookback_period=60):
        """Detect Price Channels focusing on recent data"""
        # Only look at recent data
        start_idx = max(0, len(self.closes) - lookback_period)
        recent_closes = self.closes[start_idx:]
        recent_highs = self.highs[start_idx:]
        recent_lows = self.lows[start_idx:]
        
        # Create temporary detector for recent data
        recent_df = self.df.iloc[start_idx:].copy()
        temp_detector = PatternDetector.__new__(PatternDetector)
        temp_detector.closes = recent_closes
        temp_detector.highs = recent_highs
        temp_detector.lows = recent_lows
        
        # Find peaks and troughs in recent data only
        peaks, troughs = temp_detector.find_peaks_troughs(prominence=0.015)
        
        if len(peaks) < min_touches or len(troughs) < min_touches:
            return None
        
        # Adjust indices back to original dataframe
        peaks = peaks + start_idx
        troughs = troughs + start_idx
        
        # Try different combinations of peaks for upper trendline
        for i in range(len(peaks) - min_touches + 1):
            upper_points = peaks[i:i + min_touches]
            
            # Calculate upper trendline
            upper_slope = self._calculate_trendline_slope(upper_points, self.highs)
            upper_intercept = self.highs[upper_points[0]] - upper_slope * upper_points[0]
            
            # Try to find parallel lower trendline
            for j in range(len(troughs) - min_touches + 1):
                lower_points = troughs[j:j + min_touches]
                
                # Calculate lower trendline
                lower_slope = self._calculate_trendline_slope(lower_points, self.lows)
                lower_intercept = self.lows[lower_points[0]] - lower_slope * lower_points[0]
                
                # Check if lines are roughly parallel
                slope_diff = abs(upper_slope - lower_slope)
                avg_slope = abs(upper_slope + lower_slope) / 2
                
                if avg_slope == 0 or slope_diff / avg_slope < parallel_tolerance:
                    # Calculate channel width
                    mid_point = (upper_points[-1] + lower_points[-1]) / 2
                    upper_level = upper_slope * mid_point + upper_intercept
                    lower_level = lower_slope * mid_point + lower_intercept
                    channel_width = upper_level - lower_level
                    
                    # Classify channel direction
                    if abs(upper_slope) < 0.01:
                        channel_type = 'horizontal_channel'
                    elif upper_slope > 0:
                        channel_type = 'ascending_channel'
                    else:
                        channel_type = 'descending_channel'
                    
                    return {
                        'type': channel_type,
                        'upper_points': upper_points,
                        'lower_points': lower_points,
                        'upper_slope': upper_slope,
                        'upper_intercept': upper_intercept,
                        'lower_slope': lower_slope,
                        'lower_intercept': lower_intercept,
                        'channel_width': channel_width,
                        'start_idx': max(min(upper_points[0], lower_points[0]), start_idx),
                        'end_idx': max(upper_points[-1], lower_points[-1]),
                        'lookback_start': start_idx
                    }
        
        return None

# ----------------------------
# Plotting with dynamic legend
# ----------------------------
    def plot_with_patterns_and_legend(clean_df, symbol, patterns):
        """Plot chart with colored pattern lines and on-chart legend for detected patterns"""
        addplots = []
        legend_items = []
        legend_colors = []
        
        for pattern in patterns:
            if pattern is None:
                continue
                
            if pattern['type'] == 'head_shoulders':
                # Draw neckline (red dashed)
                neckline = np.full(len(clean_df), pattern['neckline'])
                addplots.append(mpf.make_addplot(neckline, color='red', linestyle='--', width=2))
                legend_items.append("Head & Shoulders")
                legend_colors.append('red')
                
            elif pattern['type'] in ['double_top', 'double_bottom']:
                # Draw support/resistance line (blue dashed)
                level = pattern.get('support', pattern.get('resistance'))
                if level:
                    line = np.full(len(clean_df), level)
                    addplots.append(mpf.make_addplot(line, color='blue', linestyle='--', width=2))
                    pattern_name = "Double Top" if pattern['type'] == 'double_top' else "Double Bottom"
                    legend_items.append(pattern_name)
                    legend_colors.append('blue')
                    
            elif 'triangle' in pattern['type']:
                # Draw triangle trendlines (green)
                peaks, troughs = pattern['peaks'], pattern['troughs']
                triangle_name = pattern['type'].replace('_', ' ').title()
                
                if len(peaks) >= 2:
                    # Upper trendline
                    upper_line = np.full(len(clean_df), np.nan)
                    start_idx = peaks[0]
                    end_idx = min(peaks[-1] + 20, len(clean_df) - 1)
                    
                    slope = (clean_df['High'].iloc[peaks[-1]] - clean_df['High'].iloc[peaks[0]]) / (peaks[-1] - peaks[0])
                    intercept = clean_df['High'].iloc[peaks[0]] - slope * peaks[0]
                    
                    for i in range(start_idx, end_idx):
                        upper_line[i] = slope * i + intercept
                    addplots.append(mpf.make_addplot(upper_line, color='green', width=2))
                    
                if len(troughs) >= 2:
                    # Lower trendline
                    lower_line = np.full(len(clean_df), np.nan)
                    start_idx = troughs[0]
                    end_idx = min(troughs[-1] + 20, len(clean_df) - 1)
                    
                    slope = (clean_df['Low'].iloc[troughs[-1]] - clean_df['Low'].iloc[troughs[0]]) / (troughs[-1] - troughs[0])
                    intercept = clean_df['Low'].iloc[troughs[0]] - slope * troughs[0]
                    
                    for i in range(start_idx, end_idx):
                        lower_line[i] = slope * i + intercept
                    addplots.append(mpf.make_addplot(lower_line, color='green', width=2))
                
                legend_items.append(triangle_name)
                legend_colors.append('green')
            
            elif pattern['type'] in ['flag', 'bear_flag']:
                # Draw flag boundaries (orange)
                flag_start = pattern['flag_start']
                flag_high = clean_df['High'].iloc[flag_start:].max()
                flag_low = clean_df['Low'].iloc[flag_start:].min()
                
                flag_top = np.full(len(clean_df), np.nan)
                flag_bottom = np.full(len(clean_df), np.nan)
                
                for i in range(flag_start, len(clean_df)):
                    flag_top[i] = flag_high
                    flag_bottom[i] = flag_low
                
                addplots.append(mpf.make_addplot(flag_top, color='orange', width=2, linestyle='--'))
                addplots.append(mpf.make_addplot(flag_bottom, color='orange', width=2, linestyle='--'))
                
                flag_type = "Bull Flag" if pattern['type'] == 'flag' else "Bear Flag"
                legend_items.append(flag_type)
                legend_colors.append('orange')
            
            elif pattern['type'] == 'cup_handle':
                # Draw rim level (purple dashed)
                rim_line = np.full(len(clean_df), pattern['rim_level'])
                addplots.append(mpf.make_addplot(rim_line, color='purple', linestyle='--', width=2))
                
                # Highlight cup region (orange)
        return legend_items

# ----------------------------
# Main loop
# ----------------------------
for symbol in symbols:
    df = yf.download(
        symbol,
        period="1y",
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    df = df.dropna()
    
    # Fix MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Build mplfinance-safe DataFrame
    clean_df = pd.DataFrame(
        {
            "Open":  df["Open"].to_numpy().astype("float64").ravel(),
            "High":  df["High"].to_numpy().astype("float64").ravel(),
            "Low":   df["Low"].to_numpy().astype("float64").ravel(),
            "Close": df["Close"].to_numpy().astype("float64").ravel(),
        },
        index=pd.to_datetime(df.index)
    )
    
    # Detect patterns
    detector = PatternDetector(clean_df)
    patterns = [
        detector.detect_head_shoulders(),
        detector.detect_double_top_bottom(),
        detector.detect_triangle(),
        detector.detect_flag_pennant(),
        detector.detect_cup_handle(),
        detector.detect_price_channels()
    ]
    
    # Plot with patterns and legend (only for detected patterns)
    legend_items = plot_with_patterns_and_legend(clean_df, symbol, patterns)
