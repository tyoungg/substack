# Financial Chart Generator

This project is a Python script that generates financial charts for a list of stock symbols. It uses `yfinance` to download historical data and `mplfinance` to create candlestick charts. The script also includes a `PatternDetector` class that can identify various technical analysis patterns in the data.

## Features

- **Data Fetching**: Downloads historical stock data from Yahoo Finance using the `yfinance` library.
- **Chart Generation**: Creates clean and informative candlestick charts using `mplfinance`.
- **Technical Pattern Detection**: Includes a `PatternDetector` class to identify common technical analysis patterns, such as:
  - Head and Shoulders
  - Double Top and Double Bottom
  - Ascending, Descending, and Symmetrical Triangles
  - Bull and Bear Flags
  - Cup and Handle
  - Price Channels
- **Configuration**: Easily configure the list of stock symbols to analyze by editing the `symbols.yaml` file.
- **Pattern Toggle**: Enable or disable pattern detection globally from the `symbols.yaml` file.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/tyoungg/substack.git
   cd substack
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```


## Usage

### Generating Charts for Multiple Symbols

1. **Run the script:**
   ```bash
   python generate_charts.py
   ```
   The generated charts will be saved in the `charts` directory.

2. **Customize the stock symbols:**
   - Open the `symbols.yaml` file.
   - Add or remove symbols from the `symbols` list.

3. **Enable or disable pattern detection:**
   - Open the `symbols.yaml` file.
   - Set `enable_patterns` to `true` to enable pattern detection, or `false` to disable it.
### Generating a Chart for a Single Symbol

1. **Specify the stock symbol:**
   - Open the `single_symbol.yaml` file.
   - Change the `symbol` to the stock you want to analyze.

2. **Run the script:**
   ```bash
   python generate_single_chart.py
   ```
   The generated chart will be saved in the `charts` directory.

### Generating a Chart via GitHub Actions

1.  Navigate to the **Actions** tab of the repository on GitHub.
2.  In the left sidebar, click on the **Single Symbol Chart Generation** workflow.
3.  Click the **Run workflow** dropdown button.
4.  Enter the stock symbol you want to analyze in the **Stock Symbol** input field.
5.  Click the **Run workflow** button to start the process.

The workflow will generate the chart and commit it to the `charts` directory in the repository.


## How It Works

The script uses the `yfinance` library to download historical stock data for the symbols defined in `symbols.yaml`. It then processes the data and uses the `mplfinance` library to generate candlestick charts.

When `enable_patterns` is set to `true` in `symbols.yaml`, the script uses the `PatternDetector` class to analyze the data for the following technical patterns:

### Detected Patterns

- **Head and Shoulders**: Identifies a specific chart formation that predicts a bullish-to-bearish trend reversal.
- **Double Top and Double Bottom**: Chart patterns that occur when a stock price reaches a certain level twice and is unable to break through.
- **Triangles**: Formations that occur when the price of a stock is consolidating. The script can identify ascending, descending, and symmetrical triangles.
- **Flags**: Short-term continuation patterns that occur after a sharp price movement.
- **Cup and Handle**: A bullish continuation pattern that resembles a cup with a handle.
- **Price Channels / Channel**: Two parallel trendlines that contain the price of a stock, representing a controlled trend.
- **Trendline**: Indicates directional pressure.
- **Horizontal Zone**: Shows price memory.
- **Shelf**: Indicates consolidation.
- **Regime Start**: Marks a context reset.
- **Threat Line**: Represents a risk boundary.


If any of these patterns are detected, they are drawn on the chart, and a legend is added to identify them.

