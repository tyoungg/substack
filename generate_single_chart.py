import os
import yaml
import pandas as pd
import yfinance as yf
from chart_utils import get_company_name, PatternDetector, plot_with_patterns_and_legend

# ----------------------------
# Load symbol and config
# ----------------------------
with open("single_symbol.yaml", "r") as f:
    config = yaml.safe_load(f)
    symbol = config["symbol"]

os.makedirs("charts", exist_ok=True)

# ----------------------------
# Main execution
# ----------------------------
def main():
    print(f"Processing {symbol}...")
    company_name = get_company_name(symbol)
    print(f"Company: {company_name}")

    df = yf.download(
        symbol, period="1y", interval="1d",
        auto_adjust=False, progress=False,
    )
    df = df.dropna()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    clean_df = pd.DataFrame({
        "Open": df["Open"].to_numpy().astype("float64").ravel(),
        "High": df["High"].to_numpy().astype("float64").ravel(),
        "Low": df["Low"].to_numpy().astype("float64").ravel(),
        "Close": df["Close"].to_numpy().astype("float64").ravel(),
    }, index=pd.to_datetime(df.index))

    detector = PatternDetector(clean_df)
    patterns = [
        detector.detect_head_shoulders(),
        detector.detect_double_top_bottom(),
        detector.detect_triangle(),
        detector.detect_flag_pennant(),
        detector.detect_cup_handle(),
        detector.detect_price_channels(),
        detector.detect_regime_start(),
        detector.detect_threat_line()
    ]

    plot_with_patterns_and_legend(clean_df, symbol, company_name, patterns)

if __name__ == "__main__":
    main()
