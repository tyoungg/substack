import os
import yaml
import pandas as pd
import yfinance as yf
from chart_utils import get_company_name, PatternDetector, plot_with_patterns_and_legend, plot_simple_chart

# ----------------------------
# Load symbols and config
# ----------------------------
with open("symbols.yaml", "r") as f:
    config = yaml.safe_load(f)
    symbols = config["symbols"]

os.makedirs("charts", exist_ok=True)

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

    if df.empty:
        print(f"  No data for {symbol}, skipping.")
        continue
    
    # Fix MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Build mplfinance-safe DataFrame (guaranteed 1-D floats)
    clean_df = pd.DataFrame(
        {
            "Open":   df["Open"].to_numpy().astype("float64").ravel(),
            "High":   df["High"].to_numpy().astype("float64").ravel(),
            "Low":    df["Low"].to_numpy().astype("float64").ravel(),
            "Close":  df["Close"].to_numpy().astype("float64").ravel(),
            "Volume": df["Volume"].to_numpy().astype("float64").ravel(),
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
                detector.detect_price_channels(),
                detector.detect_undercut_rally(),
                detector.detect_regime_start(),
                detector.detect_threat_line()
            ]

            # Plot with patterns and company name
            plot_with_patterns_and_legend(clean_df, symbol, company_name, patterns)
        else:
            # Plot simple chart with company name
            plot_simple_chart(clean_df, symbol, company_name)
            print(f"{symbol}: Simple chart generated")
