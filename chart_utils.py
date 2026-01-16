import numpy as np
import pandas as pd
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
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

    def detect_threat_line(self, lookback=30):
        """Detects a threat line connecting the last two significant peaks or troughs."""
        recent_highs = self.highs[-lookback:]
        recent_lows = self.lows[-lookback:]
        peaks, _ = find_peaks(recent_highs, prominence=(np.max(recent_highs) - np.min(recent_highs)) * 0.05, distance=3)
        troughs, _ = find_peaks(-recent_lows, prominence=(np.max(recent_lows) - np.min(recent_lows)) * 0.05, distance=3)
        offset = len(self.highs) - lookback
        peaks = [p + offset for p in peaks]
        troughs = [t + offset for t in troughs]
        x = np.arange(lookback)
        y = self.closes[-lookback:]
        slope = np.polyfit(x, y, 1)[0]
        if slope > 0 and len(troughs) >= 2: # Uptrend, connect troughs
            p1, p2 = troughs[-2], troughs[-1]
            line_slope = (self.lows[p2] - self.lows[p1]) / (p2 - p1)
            intercept = self.lows[p1] - line_slope * p1
            return {'type': 'threat_line_support', 'p1': p1, 'p2': p2, 'slope': line_slope, 'intercept': intercept}
        elif slope < 0 and len(peaks) >= 2: # Downtrend, connect peaks
            p1, p2 = peaks[-2], peaks[-1]
            line_slope = (self.highs[p2] - self.highs[p1]) / (p2 - p1)
            intercept = self.highs[p1] - line_slope * p1
            return {'type': 'threat_line_resistance', 'p1': p1, 'p2': p2, 'slope': line_slope, 'intercept': intercept}
        return None

# ----------------------------
# Plotting with patterns and legend
# ----------------------------
def plot_with_patterns_and_legend(clean_df, symbol, company_name, patterns):
    """Plot chart with colored pattern lines and custom x-axis"""
    addplots = []
    legend_handles = []
    deferred_drawings = []

    for pattern in patterns:
        if pattern is None:
            continue
        if pattern['type'] == 'head_shoulders':
            left_shoulder, right_shoulder = pattern['left_shoulder'], pattern['right_shoulder']
            neckline_val = min(clean_df['Low'].iloc[left_shoulder:right_shoulder+1])
            neckline = np.full(len(clean_df), np.nan)
            for i in range(max(0, left_shoulder - 5), min(right_shoulder + 15, len(clean_df))):
                neckline[i] = neckline_val
            addplots.append(mpf.make_addplot(neckline, color='red', linestyle='--', width=2))
            legend_handles.append(plt.Line2D([], [], color='red', linestyle='--', label='Head & Shoulders'))
        elif pattern['type'] == 'double_top':
            peak1, peak2 = pattern['peak1'], pattern['peak2']
            resistance_val = max(clean_df['High'].iloc[peak1], clean_df['High'].iloc[peak2])
            resistance_line = np.full(len(clean_df), np.nan)
            for i in range(max(0, peak1 - 5), min(peak2 + 15, len(clean_df))):
                resistance_line[i] = resistance_val
            addplots.append(mpf.make_addplot(resistance_line, color='blue', linestyle='--', width=2))
            legend_handles.append(plt.Line2D([], [], color='blue', linestyle='--', label='Double Top'))
        elif pattern['type'] == 'double_bottom':
            trough1, trough2 = pattern['trough1'], pattern['trough2']
            support_val = min(clean_df['Low'].iloc[trough1], clean_df['Low'].iloc[trough2])
            support_line = np.full(len(clean_df), np.nan)
            for i in range(max(0, trough1 - 5), min(trough2 + 15, len(clean_df))):
                support_line[i] = support_val
            addplots.append(mpf.make_addplot(support_line, color='blue', linestyle='--', label='Double Bottom'))
        elif 'triangle' in pattern['type']:
            peaks, troughs = pattern['peaks'], pattern['troughs']
            triangle_name = pattern['type'].replace('_', ' ').title()
            pattern_start = min(peaks[0] if len(peaks) > 0 else len(clean_df), troughs[0] if len(troughs) > 0 else len(clean_df))
            pattern_end = max(peaks[-1] if len(peaks) > 0 else 0, troughs[-1] if len(troughs) > 0 else 0)
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
        elif pattern['type'] in ['flag', 'bear_flag']:
            flag_start = pattern['flag_start']
            flag_high = clean_df['High'].iloc[flag_start:].max()
            flag_low = clean_df['Low'].iloc[flag_start:].min()
            flag_top = np.full(len(clean_df), np.nan)
            flag_bottom = np.full(len(clean_df), np.nan)
            for i in range(max(0, flag_start - 2), min(len(clean_df), flag_start + 20)):
                flag_top[i] = flag_high
                flag_bottom[i] = flag_low
            addplots.append(mpf.make_addplot(flag_top, color='orange', width=2, linestyle='--'))
            addplots.append(mpf.make_addplot(flag_bottom, color='orange', width=2, linestyle='--'))
            flag_type = "Bull Flag" if pattern['type'] == 'flag' else "Bear Flag"
            legend_handles.append(plt.Line2D([], [], color='orange', linestyle='--', label=flag_type))
        elif pattern['type'] == 'cup_handle':
            cup_start, handle_end = pattern['cup_start'], pattern['handle_end']
            rim_line = np.full(len(clean_df), np.nan)
            for i in range(max(0, cup_start - 5), min(handle_end + 10, len(clean_df))):
                rim_line[i] = pattern['rim_level']
            addplots.append(mpf.make_addplot(rim_line, color='purple', linestyle='--', width=2))
            legend_handles.append(plt.Line2D([], [], color='purple', linestyle='--', label='Cup & Handle'))
        elif 'channel' in pattern['type']:
            start_idx, end_idx = pattern['start_idx'], min(pattern['end_idx'] + 20, len(clean_df) - 1)
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
        elif 'threat_line' in pattern['type']:
            p1, slope, intercept = pattern['p1'], pattern['slope'], pattern['intercept']
            line_values = np.full(len(clean_df), np.nan)
            for i in range(p1, len(clean_df)):
                line_values[i] = slope * i + intercept
            addplots.append(mpf.make_addplot(line_values, color='black', linestyle=':', width=2))
            legend_handles.append(plt.Line2D([], [], color='black', linestyle=':', label='Threat Line'))
        elif pattern['type'] == 'regime_start':
            start_index = pattern['index']
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
