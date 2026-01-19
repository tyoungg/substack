import numpy as np
import pandas as pd
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import find_peaks, argrelextrema
from sklearn.linear_model import LinearRegression

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
    except Exception as e:
        print(f"Error fetching company name for {symbol}: {e}")
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

    def detect_head_shoulders(self, order=5, min_head_prominence=0.1, shoulder_prominence_ratio=0.3):
        """Detect Head and Shoulders pattern with neckline validation"""
        highs = self.df['High']
        lows = self.df['Low']

        peaks_indices = argrelextrema(highs.values, np.greater, order=order)[0]
        troughs_indices = argrelextrema(lows.values, np.less, order=order)[0]

        patterns = []

        if len(peaks_indices) < 3 or len(troughs_indices) < 2:
            return None

        for i in range(1, len(peaks_indices) - 1):
            head_idx = peaks_indices[i]

            left_shoulders = peaks_indices[peaks_indices < head_idx]
            if len(left_shoulders) == 0: continue
            left_shoulder_idx = left_shoulders[-1]

            left_troughs = troughs_indices[(troughs_indices > left_shoulder_idx) & (troughs_indices < head_idx)]
            if len(left_troughs) == 0: continue
            left_trough_idx = left_troughs[0]

            right_shoulders = peaks_indices[peaks_indices > head_idx]
            if len(right_shoulders) == 0: continue
            right_shoulder_idx = right_shoulders[0]

            right_troughs = troughs_indices[(troughs_indices > head_idx) & (troughs_indices < right_shoulder_idx)]
            if len(right_troughs) == 0: continue
            right_trough_idx = right_troughs[-1]

            head_price = highs.iloc[head_idx]
            left_shoulder_price = highs.iloc[left_shoulder_idx]
            right_shoulder_price = highs.iloc[right_shoulder_idx]
            left_trough_price = lows.iloc[left_trough_idx]
            right_trough_price = lows.iloc[right_trough_idx]

            if not (head_price > left_shoulder_price and head_price > right_shoulder_price):
                continue

            if abs(left_shoulder_price - right_shoulder_price) / head_price > 0.2:
                continue

            # Using numerical indices for interpolation
            trough_dates_num = np.array([left_trough_idx, right_trough_idx])
            trough_prices = np.array([left_trough_price, right_trough_price])

            # Avoid singular matrix error if troughs are at the same index
            if len(np.unique(trough_dates_num)) < 2:
                continue

            neckline_coeffs = np.polyfit(trough_dates_num, trough_prices, 1)
            neckline_poly = np.poly1d(neckline_coeffs)

            neckline_at_head = neckline_poly(head_idx)
            head_prominence = (head_price - neckline_at_head) / neckline_at_head if neckline_at_head > 0 else 0

            if head_prominence < min_head_prominence:
                continue

            neckline_at_ls = neckline_poly(left_shoulder_idx)
            neckline_at_rs = neckline_poly(right_shoulder_idx)

            ls_prominence = (left_shoulder_price - neckline_at_ls) / neckline_at_ls if neckline_at_ls > 0 else 0
            rs_prominence = (right_shoulder_price - neckline_at_rs) / neckline_at_rs if neckline_at_rs > 0 else 0

            if ls_prominence > head_prominence * shoulder_prominence_ratio or \
               rs_prominence > head_prominence * shoulder_prominence_ratio:
                continue

            # Return the first valid pattern found
            return {
                'type': 'head_shoulders',
                'left_shoulder': left_shoulder_idx,
                'head': head_idx,
                'right_shoulder': right_shoulder_idx,
                'left_trough': left_trough_idx,
                'right_trough': right_trough_idx
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

    def detect_flag_pennant(self, window=15):
        """Detect Flag and Pennant patterns"""
        if len(self.closes) < window * 2:
            return None
        recent = self.closes[-window:]
        prev_period = self.closes[-(window*2):-window]
        pole_move = (recent[0] - prev_period[0]) / prev_period[0]
        if abs(pole_move) > 0.05:
            volatility = np.std(recent) / np.mean(recent)
            if volatility < 0.03:
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
        for start_idx in range(len(self.closes) - min_cup_length):
            end_idx = start_idx + min_cup_length
            cup_data = self.closes[start_idx:end_idx]
            bottom_idx = start_idx + np.argmin(cup_data)
            left_side = cup_data[:bottom_idx - start_idx]
            right_side = cup_data[bottom_idx - start_idx:]
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
                if (np.min(handle_data) > upper_third and
                    handle_data[-1] < handle_data[0] and
                    abs(handle_data[-1] - handle_data[0]) < 0.03 * handle_data[0]):
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
        start_idx = max(0, len(self.closes) - lookback_period)
        recent_closes = self.closes[start_idx:]
        recent_highs = self.highs[start_idx:]
        recent_lows = self.lows[start_idx:]

        recent_df = self.df.iloc[start_idx:].copy()
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
            upper_points = peaks[i:i + min_touches]
            upper_slope = self._calculate_trendline_slope(upper_points, self.highs)
            upper_intercept = self.highs[upper_points[0]] - upper_slope * upper_points[0]

            for j in range(len(troughs) - min_touches + 1):
                lower_points = troughs[j:j + min_touches]
                lower_slope = self._calculate_trendline_slope(lower_points, self.lows)

                slope_diff = abs(upper_slope - lower_slope)
                avg_slope = abs(upper_slope + lower_slope) / 2

                if avg_slope == 0 or slope_diff / avg_slope < parallel_tolerance:
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
                        'lower_intercept': self.lows[lower_points[0]] - lower_slope * lower_points[0],
                        'start_idx': max(min(upper_points[0], lower_points[0]), start_idx),
                        'end_idx': max(upper_points[-1], lower_points[-1]),
                    }
        return None

    def detect_regime_start(self, window=21, std_dev_threshold=2.0):
        """Detects a regime start, defined as a significant price change."""
        price_changes = self.df['Close'].pct_change().abs()
        rolling_std = price_changes.rolling(window=window).std()
        regime_starts = price_changes[price_changes > std_dev_threshold * rolling_std]
        if not regime_starts.empty:
            start_index = self.df.index.get_loc(regime_starts.index[0])
            return {'type': 'regime_start', 'index': start_index}
        return None

    def detect_threat_line(self, lookback=60, prominence_pct=0.08):
        """
        Detects a "threat line" which is a short-term trendline connecting recent significant peaks or troughs.
        A resistance line is formed by connecting two recent, descending peaks.
        A support line is formed by connecting two recent, ascending troughs.
        The most recent line is prioritized.
        """
        if len(self.highs) < lookback:
            return None

        # Work with the recent part of the data
        recent_highs = self.highs[-lookback:]
        recent_lows = self.lows[-lookback:]
        offset = len(self.highs) - lookback

        # Find significant peaks and troughs in the recent data
        price_range = np.max(recent_highs) - np.min(recent_lows)
        if price_range == 0: return None
        min_prominence = price_range * prominence_pct

        peaks, _ = find_peaks(recent_highs, prominence=min_prominence, distance=5)
        troughs, _ = find_peaks(-recent_lows, prominence=min_prominence, distance=5)

        # Convert local indices to global DataFrame indices
        peaks = [p + offset for p in peaks]
        troughs = [t + offset for t in troughs]

        resistance_line = None
        if len(peaks) >= 2:
            # Consider the last two peaks to form a potential resistance line
            p1, p2 = peaks[-2], peaks[-1]
            # A valid resistance line should be downward sloping
            if self.highs[p2] < self.highs[p1]:
                if p2 - p1 > 0: # a little safety check
                    line_slope = (self.highs[p2] - self.highs[p1]) / (p2 - p1)
                    intercept = self.highs[p1] - line_slope * p1
                    resistance_line = {
                        'type': 'threat_line_resistance',
                        'p1': p1, 'p2': p2,
                        'slope': line_slope,
                        'intercept': intercept
                    }

        support_line = None
        if len(troughs) >= 2:
            # Consider the last two troughs to form a potential support line
            t1, t2 = troughs[-2], troughs[-1]
            # A valid support line should be upward sloping
            if self.lows[t2] > self.lows[t1]:
                 if t2 - t1 > 0: # a little safety check
                    line_slope = (self.lows[t2] - self.lows[t1]) / (t2 - t1)
                    intercept = self.lows[t1] - line_slope * t1
                    support_line = {
                        'type': 'threat_line_support',
                        'p1': t1, 'p2': t2,
                        'slope': line_slope,
                        'intercept': intercept
                    }

        # Prioritize the most recent line
        if resistance_line and support_line:
            if resistance_line['p2'] > support_line['p2']:
                return resistance_line
            else:
                return support_line
        elif resistance_line:
            return resistance_line
        elif support_line:
            return support_line

        return None

# ----------------------------
# Plotting with patterns and legend
# ----------------------------
def plot_with_patterns_and_legend(clean_df, symbol, company_name, patterns):
    """
    Plots a financial chart with detected technical analysis patterns.

    This function takes a DataFrame of stock data and a list of detected patterns,
    then generates a candlestick chart with the patterns overlaid. It also creates
    a legend for the plotted patterns.

    Each pattern type has its own drawing logic and customizable appearance.
    To customize a pattern's look, modify the arguments in the `mpf.make_addplot()`
    call for that specific pattern. For example, to change the color of the
    "Head & Shoulders" pattern, modify the `color` parameter.

    Args:
        clean_df (pd.DataFrame): DataFrame containing the stock data with columns
                                 like 'High', 'Low', 'Close'.
        symbol (str): The stock symbol.
        company_name (str): The name of the company.
        patterns (list): A list of dictionaries, where each dictionary represents
                         a detected pattern and its key points.
    """
    addplots = []
    legend_handles = []
    deferred_drawings = []

    for pattern in patterns:
        if pattern is None:
            continue

        # ----------------------------------------------------------------------
        # Head & Shoulders Pattern
        # ----------------------------------------------------------------------
        if pattern['type'] == 'head_shoulders':
            # --- Key points ---
            left_shoulder_idx = pattern['left_shoulder']
            head_idx = pattern['head']
            right_shoulder_idx = pattern['right_shoulder']
            trough1_idx = pattern['left_trough']
            trough2_idx = pattern['right_trough']

            # --- Visualization ---
            # This section draws the Head & Shoulders pattern by connecting the peaks.
            # It first defines the key points (shoulders, head) and then interpolates
            # lines between them to form the characteristic shape.
            #   - color: 'red' (customize the line color)
            #   - marker: 'o' (marks the peak of each part of the pattern)
            #   - linestyle: '-' (solid line connecting the peaks)
            hs_line = np.full(len(clean_df), np.nan)
            points = {
                left_shoulder_idx: clean_df['High'].iloc[left_shoulder_idx],
                head_idx: clean_df['High'].iloc[head_idx],
                right_shoulder_idx: clean_df['High'].iloc[right_shoulder_idx]
            }
            sorted_indices = sorted(points.keys())

            for i in range(len(sorted_indices) - 1):
                start_idx, end_idx = sorted_indices[i], sorted_indices[i+1]
                start_val, end_val = points[start_idx], points[end_idx]
                if end_idx > start_idx:
                    slope = (end_val - start_val) / (end_idx - start_idx)
                    for j in range(start_idx, end_idx + 1):
                        hs_line[j] = start_val + slope * (j - start_idx)

            addplots.append(mpf.make_addplot(hs_line, color='red', marker='o', linestyle='-'))

            # This part calculates and draws the neckline, connecting the troughs between the head and shoulders.
            #   - color: 'red' (customize the line color)
            #   - linestyle: '--' (customize the line style)
            #   - width: 1.5 (customize the line thickness)
            if trough2_idx != trough1_idx:
                slope = (clean_df['Low'].iloc[trough2_idx] - clean_df['Low'].iloc[trough1_idx]) / (trough2_idx - trough1_idx)
                intercept = clean_df['Low'].iloc[trough1_idx] - slope * trough1_idx
                neckline = [slope * i + intercept for i in range(len(clean_df))]
                addplots.append(mpf.make_addplot(neckline, color='red', linestyle='--', width=1.5))

            legend_handles.append(plt.Line2D([], [], color='red', linestyle='-', marker='o', label='Head & Shoulders'))

        # ----------------------------------------------------------------------
        # Double Top Pattern
        # ----------------------------------------------------------------------
        elif pattern['type'] == 'double_top':
            # --- Key points ---
            peak1_idx, peak2_idx = pattern['peak1'], pattern['peak2']

            # --- Visualization ---
            # This line connects the two peaks of the double top.
            #   - color: 'blue' (customize the line color)
            #   - marker: 'o' (customize the marker for the peaks)
            #   - linestyle: '-' (customize the line style)
            #   - width: 2 (customize the line thickness)
            resistance_line = np.full(len(clean_df), np.nan)
            resistance_line[peak1_idx] = clean_df['High'].iloc[peak1_idx]
            resistance_line[peak2_idx] = clean_df['High'].iloc[peak2_idx]
            addplots.append(mpf.make_addplot(resistance_line, color='blue', marker='o', linestyle='-', width=2))

            legend_handles.append(plt.Line2D([], [], color='blue', linestyle='-', marker='o', label='Double Top'))

        # ----------------------------------------------------------------------
        # Double Bottom Pattern
        # ----------------------------------------------------------------------
        elif pattern['type'] == 'double_bottom':
            # --- Key points ---
            trough1_idx, trough2_idx = pattern['trough1'], pattern['trough2']

            # --- Visualization ---
            # This line connects the two troughs of the double bottom.
            #   - color: 'blue' (customize the line color)
            #   - marker: 'o' (customize the marker for the troughs)
            #   - linestyle: '-' (customize the line style)
            #   - width: 2 (customize the line thickness)
            support_line = np.full(len(clean_df), np.nan)
            support_line[trough1_idx] = clean_df['Low'].iloc[trough1_idx]
            support_line[trough2_idx] = clean_df['Low'].iloc[trough2_idx]
            addplots.append(mpf.make_addplot(support_line, color='blue', marker='o', linestyle='-', width=2, label='Double Bottom'))

        # ----------------------------------------------------------------------
        # Triangle Patterns (Ascending, Descending, Symmetrical)
        # ----------------------------------------------------------------------
        elif 'triangle' in pattern['type']:
            peaks, troughs = pattern['peaks'], pattern['troughs']
            triangle_name = pattern['type'].replace('_', ' ').title()

            # --- Visualization ---
            # This logic draws the upper and lower trendlines for the triangle.
            #   - color: 'green' (customize the line color)
            #   - width: 2 (customize the line thickness)
            if len(peaks) >= 2:
                slope = (clean_df['High'].iloc[peaks[-1]] - clean_df['High'].iloc[peaks[0]]) / (peaks[-1] - peaks[0])
                intercept = clean_df['High'].iloc[peaks[0]] - slope * peaks[0]
                upper_line = [slope * i + intercept for i in range(len(clean_df))]
                addplots.append(mpf.make_addplot(upper_line, color='green', width=2))
            if len(troughs) >= 2:
                slope = (clean_df['Low'].iloc[troughs[-1]] - clean_df['Low'].iloc[troughs[0]]) / (troughs[-1] - troughs[0])
                intercept = clean_df['Low'].iloc[troughs[0]] - slope * troughs[0]
                lower_line = [slope * i + intercept for i in range(len(clean_df))]
                addplots.append(mpf.make_addplot(lower_line, color='green', width=2))

            legend_handles.append(plt.Line2D([], [], color='green', linestyle='-', label=triangle_name))

        # ----------------------------------------------------------------------
        # Flag Patterns (Bull Flag, Bear Flag)
        # ----------------------------------------------------------------------
        elif pattern['type'] in ['flag', 'bear_flag']:
            flag_start_idx = pattern['flag_start']

            # --- Visualization ---
            # This logic draws the parallel lines of the flag.
            #   - color: 'orange' (customize the line color)
            #   - linestyle: '--' (customize the line style)
            #   - width: 2 (customize the line thickness)
            flag_high = clean_df['High'].iloc[flag_start_idx:].max()
            flag_low = clean_df['Low'].iloc[flag_start_idx:].min()
            flag_top = np.full(len(clean_df), np.nan)
            flag_bottom = np.full(len(clean_df), np.nan)
            for i in range(max(0, flag_start_idx - 2), min(len(clean_df), flag_start_idx + 20)):
                flag_top[i] = flag_high
                flag_bottom[i] = flag_low
            addplots.append(mpf.make_addplot(flag_top, color='orange', width=2, linestyle='--'))
            addplots.append(mpf.make_addplot(flag_bottom, color='orange', width=2, linestyle='--'))

            flag_type = "Bull Flag" if pattern['type'] == 'flag' else "Bear Flag"
            legend_handles.append(plt.Line2D([], [], color='orange', linestyle='--', label=flag_type))

        # ----------------------------------------------------------------------
        # Cup & Handle Pattern
        # ----------------------------------------------------------------------
        elif pattern['type'] == 'cup_handle':
            cup_start_idx, cup_bottom_idx, cup_end_idx = pattern['cup_start'], pattern['cup_bottom'], pattern['cup_end']
            handle_start_idx, handle_end_idx = pattern['handle_start'], pattern['handle_end']

            # --- Visualization: Cup ---
            # This part draws the 'U' shape for the cup using a simple parabola.
            # The calculation scales the parabola between the cup's rim and its bottom.
            #   - color: 'purple' (customize the line color)
            #   - width: 2 (customize the line thickness)
            cup_line = np.full(len(clean_df), np.nan)
            cup_indices = np.arange(cup_start_idx, cup_end_idx + 1)
            x = np.linspace(-1, 1, len(cup_indices))
            y = x**2 # Parabola equation
            min_y, max_y = np.min(y), np.max(y)
            scaled_y = clean_df['Low'].iloc[cup_bottom_idx] + (clean_df['High'].iloc[cup_start_idx] - clean_df['Low'].iloc[cup_bottom_idx]) * (y - min_y) / (max_y - min_y)
            cup_line[cup_indices] = scaled_y
            addplots.append(mpf.make_addplot(cup_line, color='purple', width=2))

            # --- Visualization: Handle ---
            # This part draws the handle as a downward-sloping channel.
            #   - color: 'purple' (customize the line color)
            #   - linestyle: '--' (customize the line style)
            #   - width: 1.5 (customize the line thickness)
            handle_data = clean_df.iloc[handle_start_idx:handle_end_idx + 1]
            handle_indices = np.arange(handle_start_idx, handle_end_idx + 1)

            # Fit a line to the middle of the handle data
            y_handle = (handle_data['High'] + handle_data['Low']) / 2
            slope, intercept = np.polyfit(handle_indices, y_handle, 1)

            # Calculate the width of the channel based on max deviation
            handle_midline = slope * handle_indices + intercept
            max_deviation = max(
                (handle_data['High'] - handle_midline).max(),
                (handle_midline - handle_data['Low']).max()
            )

            # Extend the channel slightly for better visualization
            extended_start = handle_start_idx - 2
            extended_end = handle_end_idx + 2

            # Upper line
            upper_handle_line = np.full(len(clean_df), np.nan)
            for i in range(extended_start, extended_end):
                 if 0 <= i < len(upper_handle_line):
                    upper_handle_line[i] = slope * i + intercept + max_deviation
            addplots.append(mpf.make_addplot(upper_handle_line, color='purple', linestyle='--', width=1.5))

            # Lower line
            lower_handle_line = np.full(len(clean_df), np.nan)
            for i in range(extended_start, extended_end):
                if 0 <= i < len(lower_handle_line):
                    lower_handle_line[i] = slope * i + intercept - max_deviation
            addplots.append(mpf.make_addplot(lower_handle_line, color='purple', linestyle='--', width=1.5))

            legend_handles.append(plt.Line2D([], [], color='purple', linestyle='-', label='Cup & Handle'))

        # ----------------------------------------------------------------------
        # Channel Patterns (Ascending, Descending, Horizontal)
        # ----------------------------------------------------------------------
        elif 'channel' in pattern['type']:
            start_idx, end_idx = pattern['start_idx'], min(pattern['end_idx'] + 20, len(clean_df) - 1)

            # --- Visualization ---
            # This logic draws the upper and lower parallel lines of the channel.
            #   - color: 'cyan' (customize the line color)
            #   - width: 2 (customize the line thickness)
            upper_line = np.full(len(clean_df), np.nan)
            for i in range(start_idx, end_idx):
                upper_line[i] = pattern['upper_slope'] * i + pattern['upper_intercept']
            addplots.append(mpf.make_addplot(upper_line, color='cyan', width=2))

            lower_line = np.full(len(clean_df), np.nan)
            for i in range(start_idx, end_idx):
                lower_line[i] = pattern['lower_slope'] * i + pattern['lower_intercept']
            addplots.append(mpf.make_addplot(lower_line, color='cyan', width=2))

            channel_name = pattern['type'].replace('_', ' ').title()
            legend_handles.append(plt.Line2D([], [], color='cyan', linestyle='-', label=channel_name))

        # ----------------------------------------------------------------------
        # Threat Line
        # ----------------------------------------------------------------------
        elif 'threat_line' in pattern['type']:
            start_idx, slope, intercept = pattern['p1'], pattern['slope'], pattern['intercept']

            # --- Visualization ---
            # This logic draws a threatening trendline based on recent peaks or troughs.
            #   - color: 'black' (customize the line color)
            #   - linestyle: ':' (customize the line style, e.g., '--', '-.')
            #   - width: 2 (customize the line thickness)
            line_values = np.full(len(clean_df), np.nan)
            for i in range(start_idx, len(clean_df)):
                line_values[i] = slope * i + intercept
            addplots.append(mpf.make_addplot(line_values, color='black', linestyle=':', width=2))

            legend_handles.append(plt.Line2D([], [], color='black', linestyle=':', label='Threat Line'))

        # ----------------------------------------------------------------------
        # Regime Start
        # ----------------------------------------------------------------------
        elif pattern['type'] == 'regime_start':
            start_index = pattern['index']

            # --- Visualization (Deferred Drawing) ---
            # This draws a vertical line to mark a change in market regime.
            # It uses a deferred drawing because mplfinance handles vertical lines differently.
            #   - color: 'magenta' (customize the line color)
            #   - linestyle: '--' (customize the line style)
            #   - linewidth: 2 (customize the line thickness)
            deferred_drawings.append(
                lambda ax: ax.axvline(x=start_index, color='magenta', linestyle='--', linewidth=2)
            )
            legend_handles.append(plt.Line2D([], [], color='magenta', linestyle='--', label='Regime Start'))

    fig, axes = mpf.plot(
        clean_df, type="candle", style="yahoo",
        addplot=addplots if addplots else None,
        title=f"{company_name} ({symbol}) — 1 Year Daily Chart",
        figsize=(16, 9), returnfig=True, tight_layout=True,
    )

    for draw_func in deferred_drawings:
        draw_func(axes[0])

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

def plot_simple_chart(clean_df, symbol, company_name):
    """Plot clean chart without patterns with custom x-axis"""
    fig, axes = mpf.plot(
        clean_df,
        type="candle",
        style="yahoo",
        title=f"{company_name} ({symbol}) — 1 Year Daily Chart",
        figsize=(16, 9),
        returnfig=True,
        tight_layout=True,
    )
    fig.savefig(f"charts/{symbol}_1y.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
