import os
import yaml
import pandas as pd
import yfinance as yf
from chart_utils import get_company_name, PatternDetector, plot_with_patterns_and_legend, plot_simple_chart, generate_html_file_list

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

    # Generate weekly chart with technical indicators
    print(f"  Fetching weekly data for {symbol}...")
    df_weekly = yf.download(
        symbol,
        period="2y",
        interval="1wk",
        auto_adjust=False,
        progress=False,
    )
    df_weekly = df_weekly.dropna()

    if not df_weekly.empty:
        if isinstance(df_weekly.columns, pd.MultiIndex):
            df_weekly.columns = df_weekly.columns.get_level_values(0)

        clean_df_weekly = pd.DataFrame(
            {
                "Open":   df_weekly["Open"].to_numpy().astype("float64").ravel(),
                "High":   df_weekly["High"].to_numpy().astype("float64").ravel(),
                "Low":    df_weekly["Low"].to_numpy().astype("float64").ravel(),
                "Close":  df_weekly["Close"].to_numpy().astype("float64").ravel(),
                "Volume": df_weekly["Volume"].to_numpy().astype("float64").ravel(),
            },
            index=pd.to_datetime(df_weekly.index)
        )

        print(f"  Generating weekly chart with technical indicators for {symbol}...")
        detector_weekly = PatternDetector(clean_df_weekly)
        patterns_weekly = [
            detector_weekly.detect_head_shoulders(),
            detector_weekly.detect_double_top_bottom(),
            detector_weekly.detect_triangle(),
            detector_weekly.detect_flag_pennant(),
            detector_weekly.detect_cup_handle(),
            detector_weekly.detect_price_channels(),
            detector_weekly.detect_undercut_rally(),
            detector_weekly.detect_regime_start(),
            detector_weekly.detect_threat_line()
        ]

        plot_with_patterns_and_legend(
            clean_df_weekly,
            symbol,
            company_name,
            patterns_weekly,
            chart_title=f"{company_name} ({symbol}) — Weekly Chart",
            filename=f"charts/{symbol}_weekly_patterns.png"
        )

# Generate HTML indices in docs folder
generate_html_file_list("charts", "docs/allcharts.html", exclude_str="weekly", page_title="substack-charts — All images", valid_symbols=symbols)
generate_html_file_list("charts", "docs/weeklies.html", filter_str="weekly", page_title="substack-charts — Weekly images", valid_symbols=symbols)
