import os
import yaml
import numpy as np
import pandas as pd
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from datetime import datetime
from scipy.signal import find_peaks, find_peaks_cwt
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
        
        # Try different name fields in order of preference
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
def pattern_annotation(pattern):
    """
    Converts pattern dict into Substack-ready text.
    Safe against missing keys and partial patterns.
    """

    if not pattern or not isinstance(pattern, dict):
        return None

    # ---- Pattern name ----
    pattern_type = pattern.get("type", "pattern")
    pattern_name = pattern_type.replace("_", " ").title()

    # ---- Score & confidence ----
    score = pattern.get("score")
    if score is None:
        confidence = "Unknown"
    elif score >= 0.8:
        confidence = "High"
    elif score >= 0.65:
        confidence = "Moderate"
    else:
        confidence = "Low"

    text = f"**{pattern_name}** detected "

    # ---- Breakout handling (defensive) ----
    breakout = pattern.get("breakout")

    if isinstance(breakout, dict) and breakout.get("confirmed"):
        bars = breakout.get("bars_after", "?")
        vol = breakout.get("volume_ratio", "?")

        text += (
            f"with a confirmed breakout "
            f"{bars} bars later "
            f"on {vol}× average volume. "
        )
    else:
        text += "but without a confirmed breakout yet. "

    text += f"Overall confidence: **{confidence}**."

    return text


# ----------------------------
# Date axis customization 
# ----------------------------
# def customize_date_axis(ax, clean_df):
#    """Customize x-axis using actual data dates"""
    # Get unique months from the actual data
    
    # Create tick positions at the start of each month that has data
#    tick_positions = []
#    tick_labels = []
    
#    for month_period in data_months:
        # Convert back to timestamp for positioning
#        month_start = month_period.start_time
#        tick_positions.append(month_start)
        
        # Label format: year for January, month abbreviation for others
#        if month_start.month == 1:
#            tick_labels.append(str(month_start.year))
#        else:
#            tick_labels.append(month_start.strftime('%b'))
    
#    ax.set_xticks(tick_positions)
#    ax.set_xticklabels(tick_labels)

# ----------------------------
# Simple plotting function for charts without patterns
# ----------------------------
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
    
    # Customize x-axis to show years for January, months for others
#    customize_date_axis(axes[0], clean_df)
#    customize_date_axis(axes[0])

    # Save the figure
    fig.savefig(f"charts/{symbol}_1y.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

# ----------------------------
# Plotting with patterns and legend
# ----------------------------
def plot_with_patterns_and_legend(clean_df, symbol, company_name, patterns):
    """Plot chart with colored pattern lines and custom x-axis"""
    addplots = []
    legend_items = []
    legend_colors = []
    
    for pattern in patterns:
        if pattern is None:
            continue
            
        if pattern['type'] == 'head_shoulders':
            # Draw neckline ONLY within the pattern range
            left_shoulder = pattern['left_shoulder']
            right_shoulder = pattern['right_shoulder']
            
            neckline = np.full(len(clean_df), np.nan)
            # Only draw within pattern range + small extension
            for i in range(max(0, left_shoulder - 5), min(right_shoulder + 15, len(clean_df))):
                neckline[i] = pattern['neckline']
            addplots.append(mpf.make_addplot(neckline, color='red', linestyle='--', width=2))
            legend_items.append("Head & Shoulders")
            legend_colors.append('red')
            
        elif pattern['type'] == 'double_top':
            # Draw resistance line ONLY within pattern range
            peak1, peak2 = pattern['peak1'], pattern['peak2']
            
            resistance_line = np.full(len(clean_df), np.nan)
            # Only draw within pattern range + small extension
            for i in range(max(0, peak1 - 5), min(peak2 + 15, len(clean_df))):
                resistance_line[i] = max(clean_df['High'].iloc[peak1], clean_df['High'].iloc[peak2])
            addplots.append(mpf.make_addplot(resistance_line, color='blue', linestyle='--', width=2))
            
            legend_items.append("Double Top")
            legend_colors.append('blue')
            
        elif pattern['type'] == 'double_bottom':
            # Draw support line ONLY within pattern range
            trough1, trough2 = pattern['trough1'], pattern['trough2']
            
            support_line = np.full(len(clean_df), np.nan)
            # Only draw within pattern range + small extension
            for i in range(max(0, trough1 - 5), min(trough2 + 15, len(clean_df))):
                support_line[i] = min(clean_df['Low'].iloc[trough1], clean_df['Low'].iloc[trough2])
            addplots.append(mpf.make_addplot(support_line, color='blue', linestyle='--', width=2))
            
            legend_items.append("Double Bottom")
            legend_colors.append('blue')
                
        elif 'triangle' in pattern['type']:
            # Draw triangle trendlines within pattern range only
            peaks, troughs = pattern['peaks'], pattern['troughs']
            triangle_name = pattern['type'].replace('_', ' ').title()
            
            # Calculate pattern range
            pattern_start = min(peaks[0] if len(peaks) > 0 else len(clean_df), 
                              troughs[0] if len(troughs) > 0 else len(clean_df))
            pattern_end = max(peaks[-1] if len(peaks) > 0 else 0, 
                            troughs[-1] if len(troughs) > 0 else 0)
            
            if len(peaks) >= 2:
                # Upper trendline - only within pattern range
                upper_line = np.full(len(clean_df), np.nan)
                slope = (clean_df['High'].iloc[peaks[-1]] - clean_df['High'].iloc[peaks[0]]) / (peaks[-1] - peaks[0])
                intercept = clean_df['High'].iloc[peaks[0]] - slope * peaks[0]
                
                for i in range(max(0, pattern_start - 5), min(pattern_end + 15, len(clean_df))):
                    upper_line[i] = slope * i + intercept
                addplots.append(mpf.make_addplot(upper_line, color='green', width=2))
                
            if len(troughs) >= 2:
                # Lower trendline - only within pattern range
                lower_line = np.full(len(clean_df), np.nan)
                slope = (clean_df['Low'].iloc[troughs[-1]] - clean_df['Low'].iloc[troughs[0]]) / (troughs[-1] - troughs[0])
                intercept = clean_df['Low'].iloc[troughs[0]] - slope * troughs[0]
                
                for i in range(max(0, pattern_start - 5), min(pattern_end + 15, len(clean_df))):
                    lower_line[i] = slope * i + intercept
                addplots.append(mpf.make_addplot(lower_line, color='green', width=2))
            
            legend_items.append(triangle_name)
            legend_colors.append('green')
        
        elif pattern['type'] in ['flag', 'bear_flag']:
            # Draw flag boundaries only within flag range
            flag_start = pattern['flag_start']
            
            flag_high = clean_df['High'].iloc[flag_start:].max()
            flag_low = clean_df['Low'].iloc[flag_start:].min()
            
            flag_top = np.full(len(clean_df), np.nan)
            flag_bottom = np.full(len(clean_df), np.nan)
            
            # Only draw within flag range + small extension
            for i in range(max(0, flag_start - 2), min(len(clean_df), flag_start + 20)):
                flag_top[i] = flag_high
                flag_bottom[i] = flag_low
            
            addplots.append(mpf.make_addplot(flag_top, color='orange', width=2, linestyle='--'))
            addplots.append(mpf.make_addplot(flag_bottom, color='orange', width=2, linestyle='--'))
            
            flag_type = "Bull Flag" if pattern['type'] == 'flag' else "Bear Flag"
            legend_items.append(flag_type)
            legend_colors.append('orange')
        
        elif pattern['type'] == 'cup_handle':
            # Draw rim level only within cup and handle range
            cup_start = pattern['cup_start']
            handle_end = pattern['handle_end']
            
            rim_line = np.full(len(clean_df), np.nan)
            for i in range(max(0, cup_start - 5), min(handle_end + 10, len(clean_df))):
                rim_line[i] = pattern['rim_level']
            addplots.append(mpf.make_addplot(rim_line, color='purple', linestyle='--', width=2))
            
            # Highlight cup region (orange)
            cup_end = pattern['cup_end']
            cup_highlight = np.full(len(clean_df), np.nan)
            cup_highlight[cup_start:cup_end] = clean_df['Close'].iloc[cup_start:cup_end]
            addplots.append(mpf.make_addplot(cup_highlight, color='orange', width=3, alpha=0.7))
            
            # Highlight handle region (red)
            handle_start = pattern['handle_start']
            handle_highlight = np.full(len(clean_df), np.nan)
            handle_highlight[handle_start:handle_end] = clean_df['Close'].iloc[handle_start:handle_end]
            addplots.append(mpf.make_addplot(handle_highlight, color='red', width=3, alpha=0.7))
            
            legend_items.append("Cup & Handle")
            legend_colors.append('purple')
        
        elif 'channel' in pattern['type']:
            # Draw channel trendlines within pattern range
            start_idx = pattern['start_idx']
            end_idx = min(pattern['end_idx'] + 20, len(clean_df) - 1)
            
            # Upper channel line
            upper_line = np.full(len(clean_df), np.nan)
            for i in range(start_idx, end_idx):
                upper_line[i] = pattern['upper_slope'] * i + pattern['upper_intercept']
            addplots.append(mpf.make_addplot(upper_line, color='cyan', width=2))
            
            # Lower channel line
            lower_line = np.full(len(clean_df), np.nan)
            for i in range(start_idx, end_idx):
                lower_line[i] = pattern['lower_slope'] * i + pattern['lower_intercept']
            addplots.append(mpf.make_addplot(lower_line, color='cyan', width=2))
            
            channel_name = pattern['type'].replace('_', ' ').title()
            legend_items.append(channel_name)
            legend_colors.append('cyan')

    # Create plot without saving yet
    fig, axes = mpf.plot(
        clean_df,
        type="candle",
        style="yahoo",
        addplot=addplots if addplots else None,
        title=f"{company_name} ({symbol}) — 1 Year Daily Chart",
        figsize=(16, 9),
        returnfig=True,  # This returns the figure so we can add legend
        tight_layout=True,
    )
    
    # Customize x-axis BEFORE adding legend
#    customize_date_axis(axes[0], clean_df)
#    customize_date_axis(axes[0])
    
    # Add legend only if patterns were detected - SEMI-TRANSPARENT OVERLAY
    if legend_items:
        # Create legend patches
        legend_patches = []
        for item, color in zip(legend_items, legend_colors):
            legend_patches.append(mpatches.Patch(color=color, label=item))
        
        # Semi-transparent overlay - data shows through
        axes[0].legend(
            handles=legend_patches, 
            loc='upper left',
            bbox_to_anchor=(0.02, 0.98),
            frameon=True,
            fancybox=True,
            shadow=False,        # No shadow for cleaner look
            fontsize=9,
            framealpha=0.7,      # Semi-transparent background
            edgecolor='gray',    # Subtle border
            facecolor='white'    # White background with transparency
        )
        
        print(f"{symbol}: {', '.join(legend_items)}")
    else:
        print(f"{symbol}: No patterns detected")
    
    # Save with patterns filename
    fig.savefig(f"charts/{symbol}_1y_patterns.png", dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close to free memory
    
    return legend_items


#-----------------------
#Commentary
#----------------------


def pattern_annotation(pattern):
    """
    Converts pattern dict into Substack-ready text.
    """

    score = pattern.get("score", 0)
    confidence = (
        "High" if score >= 0.8 else
        "Moderate" if score >= 0.65 else
        "Low"
    )

    text = f"**{pattern['type'].replace('_', ' ').title()}** detected "

    if "breakout" in pattern and pattern["breakout"]["confirmed"]:
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
# Main loop
# ----------------------------
for symbol in symbols:
    print(f"Processing {symbol}...")
    
    # Get company name
    company_name = get_company_name(symbol)
    print(f"Company: {company_name}")
    
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
    
    # Build mplfinance-safe DataFrame (guaranteed 1-D floats)
    clean_df = pd.DataFrame(
        {
            "Open":  df["Open"].to_numpy().astype("float64").ravel(),
            "High":  df["High"].to_numpy().astype("float64").ravel(),
            "Low":   df["Low"].to_numpy().astype("float64").ravel(),
            "Close": df["Close"].to_numpy().astype("float64").ravel(),
        },
        index=pd.to_datetime(df.index)
    )

    for enable_patterns in [True, False]:
        print(f"  Generating chart for enable_patterns={enable_patterns}...")
        # Generate charts based on pattern setting
        if enable_patterns:
            # Detect all patterns
            detector = PatternDetector(clean_df)
            patterns = [
                detector.detect_head_shoulders(),
                detector.detect_double_top_bottom(),
                detector.detect_triangle(),
                detector.detect_flag_pennant(),
                detector.detect_cup_handle(),
                detector.detect_price_channels()
            ]

            # Plot with patterns and company name - FIXED: Added company_name parameter
            legend_items = plot_with_patterns_and_legend(clean_df, symbol, company_name, patterns)
        else:
            # Plot simple chart with company name
            plot_simple_chart(clean_df, symbol, company_name)
            print(f"{symbol}: Simple chart generated")
