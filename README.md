<div align="center">

# LLM Trading Strategy Generator

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--5-green.svg)](https://platform.openai.com/)
[![Anthropic](https://img.shields.io/badge/Anthropic-Claude-purple.svg)](https://www.anthropic.com/)
[![Google](https://img.shields.io/badge/Google-Gemini-red.svg)](https://ai.google.dev/)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow.svg)](https://huggingface.co/)
[![CodeQwen](https://img.shields.io/badge/Qwen-CodeQwen1.5-cyan.svg)](https://huggingface.co/Qwen)
[![CodeGemma](https://img.shields.io/badge/Google-CodeGemma-gold.svg)](https://huggingface.co/google)
[![Gradio](https://img.shields.io/badge/Gradio-Interface-orange.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


**An AI-powered tool that generates algorithmic trading strategies using multiple frontier and open-source language models**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Models](#-supported-models) • [Documentation](#-architecture)

</div>

---

## ✨ Features

### 🤖 Multi-Model AI Integration
- **Frontier Models**: GPT-4o, Claude Sonnet 4.5, Gemini 2.5 Pro
- **Open-Source Models**: CodeQwen 1.5 7B, CodeGemma 7B
- **Real-time Streaming**: Progressive code generation with live output
- **Model Comparison**: Test strategies across different AI architectures

### 📊 Trading Strategy Generation
- Generates 5 unique trading functions from a single example
- Supports multiple strategy types:
  - Momentum and trend-following
  - Mean reversion
  - Moving average crossovers
  - RSI-based strategies
  - Volatility breakout strategies
  - Random/Monte Carlo approaches
- Realistic threshold conditions that trigger frequently on real market data
- Fallback mechanisms to ensure consistent signal generation

### 💹 Market Data Integration
- **Free Data Sources**: Yahoo Finance via `yfinance` (no API key required)
- **Historical OHLCV Data**: Open, High, Low, Close, Volume
- **46 Ticker Support**: Includes tech stocks, crypto-related stocks, and traditional companies
- **Data Validation**: Automatic handling of delisted/missing tickers

### 🎮 Interactive Gradio Interface
- **Code Generation**: Generate 5 trading strategies from example function
- **Strategy Viewer**: View individual strategy code with syntax highlighting
- **Trade Execution**: Run strategies and see actual trade results
- **Model Selection**: Easy dropdown to switch between AI models
- **Results Display**: Formatted output showing tickers, quantities, and actions

### 🔧 Advanced Features
- **Trade Splitting**: Automatically parse generated code into individual functions
- **Execution Engine**: Run strategies with real market data
- **Error Handling**: Robust retry mechanisms for strategy execution
- **Code Sanitization**: Automatic fixing of common Pandas DataFrame errors
- **Multiple Ticker Sets**: Standard (17 tickers) and extended (43 tickers) for different strategies

---

## 🤖 Supported Models

### Frontier AI Models

| Model | Provider | Version | Speed | Quality | Best For |
|-------|----------|---------|-------|---------|----------|
| GPT-4o | OpenAI | gpt-4o | ⚡⚡⚡ | ⭐⭐⭐⭐ | Fast generation with good quality |
| Claude Sonnet 4.5 | Anthropic | claude-sonnet-4-5-20250929 | ⚡⚡ | ⭐⭐⭐⭐⭐ | Highest quality strategies |
| Gemini 2.5 Pro | Google | gemini-2.5-pro | ⚡⚡ | ⭐⭐⭐⭐⭐ | Advanced strategy logic |

### Open-Source Models (HuggingFace)

| Model | Size | Quantization | Speed | Best For |
|-------|------|--------------|-------|----------|
| CodeQwen 1.5 | 7B | 4-bit | ⚡⚡⚡ | Algorithmic code generation |
| CodeGemma | 7B | N/A (Inference API) | ⚡⚡⚡ | Lightweight strategy creation |

---

## 📦 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended for open-source models)
- **GPU**: Optional (for faster CodeQwen inference with MPS/CUDA)

### Core Dependencies
```
openai>=1.0.0
anthropic>=0.7.0
google-generativeai>=0.3.0
gradio>=4.0.0
transformers>=4.30.0
torch>=2.0.0
yfinance>=0.2.0
pandas>=1.5.0
numpy>=1.23.0
huggingface-hub>=0.16.0
python-dotenv>=0.21.0
bitsandbytes>=0.41.0  # For model quantization
```

---

## 🔧 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/llm-trading-strategy-generator.git
cd llm-trading-strategy-generator
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the project root:
```env
# Frontier Models
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# HuggingFace
HF_API_KEY=your_huggingface_token_here
```

### 5. Launch Jupyter Notebook
```bash
jupyter notebook "Trading Project.ipynb"
```

---

## ⚙️ Configuration

### API Keys Setup

#### OpenAI API
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Create API key in account settings
3. Add to `.env` as `OPENAI_API_KEY`

#### Anthropic API
1. Visit [Anthropic Console](https://console.anthropic.com/)
2. Generate API key
3. Add to `.env` as `ANTHROPIC_API_KEY`

#### Google Gemini API
1. Visit [Google AI Studio](https://ai.google.dev/)
2. Create API key
3. Add to `.env` as `GOOGLE_API_KEY`

#### HuggingFace Token
1. Visit [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create token with read access
3. Add to `.env` as `HF_API_KEY`

### Model Configuration
The following models are pre-configured in the notebook:
```python
claude_model = 'claude-sonnet-4-5-20250929'
gpt_model = 'gpt-4o'
gemini_model = 'gemini-2.5-pro'
code_qwen = 'Qwen/CodeQwen1.5-7B-Chat'
code_gemma = 'google/codegemma-7b-it'
```

---

## 🎮 Usage

### Launch the Interface
1. Open `Trading Project.ipynb` in Jupyter
2. Run all cells to initialize models and data
3. The Gradio interface will launch automatically at `http://127.0.0.1:7865`

### Using the Web Interface

#### Step 1: Generate Strategies
1. **Input Example Function**: The default example is pre-loaded (trade1)
2. **Select Model**: Choose from GPT, Claude, Gemini, Qwen, or Gemma
3. **Click "Simulate Trading"**: Watch real-time code generation
4. **View Output**: See 5 unique trading strategies (trade2 through trade6)

#### Step 2: View Individual Strategy
1. **Select Trade Number**: Use slider (2-6) to choose which strategy
2. **Click "View Selected Trade Code"**: Display the specific function code
3. **Review Logic**: Examine strategy implementation and comments

#### Step 3: Execute Strategy
1. **Click "Run Trading Strategy"**: Execute the selected strategy
2. **View Results**: See trade output with tickers, quantities, and actions
3. **Analyze Performance**: Review which stocks were selected for trading

### Supported Tickers
The system fetches real-time data for 46 tickers including:
- **Tech Giants**: AAPL, MSFT, GOOGL, AMZN, META, NFLX
- **Growth Stocks**: TSLA, NVDA, AMD, UBER, PYPL
- **Crypto-Related**: COIN, RIOT, MARA
- **Meme Stocks**: GME, AMC
- **Enterprise**: IBM, ORCL, CRM, CSCO
- **Cloud/SaaS**: SNOW, DDOG, MDB, CRWD, ZM, OKTA
- **And many more...**

---

## 🏗️ Architecture

### System Overview
```
┌─────────────────────────────────────────────────────────┐
│               Gradio Web Interface                      │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  Generate   │  │ View Code    │  │ Run Strategy │    │
│  │  Strategies │  │              │  │              │    │
│  └─────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│           Multi-Model Strategy Generator                │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │   GPT    │  │  Claude  │  │  Gemini  │               │
│  │   4o     │  │ Sonnet4.5│  │ 2.5 Pro  │               │
│  └──────────┘  └──────────┘  └──────────┘               │
│                                                         │
│  ┌──────────┐  ┌──────────┐                             │
│  │CodeQwen  │  │CodeGemma │                             │
│  │ 1.5 7B   │  │   7B     │                             │
│  └──────────┘  └──────────┘                             │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│         Strategy Execution Engine                       │
│                                                         │
│  • Code Splitting & Parsing                             │
│  • Market Data Fetching (yfinance)                      │
│  • Trade Simulation                                     │
│  • Error Handling & Retries                             │
│  • Results Formatting                                   │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              Trade Results Display                      │
│                                                         │
│  Ticker    Quantity    Action                           │
│  ──────────────────────────────                         │
│  AAPL      100         BUY                              │
│  MSFT      -50         SELL                             │
│  ...                                                    │
└─────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. Strategy Generation
- **System Prompt**: Detailed instructions for AI models on generating trading strategies
- **User Prompt**: Example function + requirements for 5 additional strategies
- **Streaming**: Real-time code generation with progressive display

#### 2. Code Processing
- **`split_trades()`**: Parse generated code into individual function dictionaries
- **`sanitize_generated_code()`**: Fix common Pandas DataFrame errors
- **`display_splitted_trade()`**: Extract and display specific trade function

#### 3. Trade Execution
- **`execute_trade_nums_claude()`**: Run Claude-generated strategies
- **`execute_trade_nums_gemini()`**: Run Gemini-generated strategies with validation
- **`execute_trade_nums_openai()`**: Run GPT-generated strategies with helper functions
- **Retry Mechanisms**: Up to 10 attempts for strategies that return no trades

#### 4. Data Management
- **Price Fetching**: Download 60 days of historical data for all tickers
- **Price Storage**: Global `prices` dictionary with ticker -> price list mapping
- **Ticker Sets**: Standard (17) and extended (43) ticker lists for different strategies

---

## 💡 Example Workflow

### Input (Trade 1 Example)
```python
import numpy as np

def trade1():
    # Buy top performing stock in the last 5 days
    avg_prices = {ticker: np.mean(prices[ticker][:5]) for ticker in tickers}
    best_ticker = max(avg_prices, key=avg_prices.get)
    trade = Trade(best_ticker, 100)
    return [trade]
```

### Generated Output (Example)
```python
def trade2():
    # Mean Reversion: Buy stocks that dropped significantly
    # in last 5 days compared to 20-day average
    trades = []
    for ticker in tickers:
        recent_avg = np.mean(prices[ticker][:5])
        long_avg = np.mean(prices[ticker][:20])
        if recent_avg < long_avg * 0.95:  # 5% drop
            trades.append(Trade(ticker, 100))
    return trades

def trade3():
    # Volatility Breakout: Buy high volatility stocks
    # with positive momentum
    trades = []
    for ticker in tickers:
        volatility = np.std(prices[ticker][:10])
        momentum = prices[ticker][0] - prices[ticker][4]
        if volatility > 0.015 or momentum > 0:
            trades.append(Trade(ticker, 50))
    return trades

# ... trade4(), trade5(), trade6()
```

### Execution Results
```
Trade 4: 3 trades

Ticker     Quantity   Action    
───────────────────────────────
AAPL       100        BUY       
MSFT       100        BUY       
GOOGL      50         BUY       
```

---

## 📊 Trading Strategy Patterns

### Strategy Types Generated

1. **Momentum Trading**
   - Buy stocks with strong recent price increases
   - Use 5-day vs 20-day moving averages
   - Filter by volume and price change thresholds

2. **Mean Reversion**
   - Buy oversold stocks (RSI < 40)
   - Sell overbought stocks (RSI > 60)
   - Target price deviations from moving averages

3. **Volatility Breakout**
   - Trade stocks with high volatility (> 1.5%)
   - Combine with momentum filters
   - Scale position sizes by inverse volatility

4. **Moving Average Crossover**
   - 5-day MA crossing above 20-day MA (buy signal)
   - 5-day MA crossing below 20-day MA (sell signal)
   - Include fallback conditions for near-crossovers

5. **RSI-Based**
   - Calculate 14-period RSI
   - Buy when RSI < 40, sell when RSI > 60
   - Use OR logic for inclusive conditions

6. **Risk-Managed**
   - Position sizing based on volatility
   - Maximum position limits per stock
   - Diversification across multiple tickers

### Critical Design Principles

The system enforces **realistic threshold conditions** to ensure strategies generate trades regularly:

❌ **Bad (Too Strict)**
```python
if volatility > 0.03 and momentum > 0.02:  # Almost never triggers
    return [Trade(ticker, 100)]
```

✅ **Good (Realistic)**
```python
if momentum > 0.01 or (volatility > 0.015 and price_change > 0):
    return [Trade(ticker, 100)]
```

---

## 🔍 Trade Object Structure

### Trade Class Definition
```python
class Trade:
    def __init__(self, ticker, quantity):
        self.ticker = ticker      # Stock symbol (e.g., 'AAPL')
        self.quantity = quantity  # Positive for BUY, negative for SELL
```

### Usage Examples
```python
# Buy 100 shares of Apple
Trade('AAPL', 100)

# Sell/short 50 shares of Microsoft
Trade('MSFT', -50)

# Return multiple trades
return [
    Trade('AAPL', 100),
    Trade('GOOGL', 75),
    Trade('MSFT', -50)
]
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. API Key Errors
**Error**: `AuthenticationError: Invalid API key`

**Solution**: Verify `.env` file exists and contains valid keys
```bash
cat .env  # Check file exists
# Verify each key is on its own line without quotes
```

#### 2. No Trades Generated
**Error**: `Trade X: 0 trades` or empty results

**Solution**: Strategy conditions are too strict
- The system automatically retries up to 10 times
- Try different trade numbers (trade4 or trade6 usually more reliable)
- Check if market data was successfully downloaded

#### 3. Module Not Found
**Error**: `ModuleNotFoundError: No module named 'X'`

**Solution**: Install missing dependencies
```bash
pip install -r requirements.txt
# Or install specific package
pip install yfinance pandas numpy
```

#### 4. HuggingFace Login Failed
**Error**: `Token is invalid`

**Solution**: Re-authenticate with HuggingFace
```python
from huggingface_hub import login
login(token="your_token_here", add_to_git_credential=True)
```

#### 5. Memory Issues with CodeQwen
**Error**: `RuntimeError: CUDA out of memory` or system freezes

**Solution**: Using 4-bit quantization already reduces memory. If issues persist:
- Close other applications
- Restart Jupyter kernel
- Use frontier models (GPT/Claude/Gemini) instead

#### 6. Pandas DataFrame Ambiguity
**Error**: `ValueError: The truth value of a DataFrame is ambiguous`

**Solution**: Already handled by `sanitize_generated_code()`
- Automatically adds `.any()` or `.values[0]` where needed
- If persistent, try regenerating with a different model

#### 7. Delisted Ticker Errors
**Warning**: `possibly delisted; no price data found`

**Solution**: System automatically handles this
- SPLK, TWTR, FEYE are known delisted tickers
- They're automatically excluded from `available_tickers`
- No action needed - strategies run with remaining valid tickers

---

## 📈 Performance Tips

### For Best Results

1. **Model Selection**
   - **Claude Sonnet 4.5**: Best quality strategies, most consistent
   - **GPT-4o**: Fastest generation, good quality
   - **Gemini 2.5 Pro**: Strong logic, requires validation retries
   - **CodeQwen/Gemma**: Good for offline use, lower quality

2. **Strategy Execution**
   - Trade3 uses extended ticker set (43 stocks)
   - Other trades use standard set (17 stocks)
   - Check "Available tickers" count in console output

3. **Debugging**
   - Enable DEBUG output to see execution flow
   - Check console for ticker availability messages
   - Verify price data was downloaded successfully

---

## 📁 Project Structure

```
llm-trading-strategy-generator/
│
├── Trading Project.ipynb         # Main Jupyter notebook with all code
├── .env                          # API keys (not in repo)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── LICENSE                       # MIT License
```

---


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

**This tool is for educational and research purposes only.**

- Not financial advice
- No guarantee of profitability
- Past performance does not indicate future results
- Always do your own research before trading
- Use at your own risk

---

## 🎓 Citation

If you use LLM Trading Strategy Generator in research, please cite:
```bibtex
@software{LLM Trading Strategy Generator 2025,
  author = {Asutosha Nanda},
  title = {LLM Trading Strategy Generator},
  year = {2025},
  url = {https://github.com/AsutoshaNanda/llm-trading-strategy-generator}
}
```

---

<div align="center">

**[⬆ Back to Top](#llm-trading-strategy-generator)**

Made with 🤖 by AI Trading Research

</div>
