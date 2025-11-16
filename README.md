<div align="center">

# llm-trading-strategy-generator

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--5-green.svg)](https://platform.openai.com/)
[![Anthropic](https://img.shields.io/badge/Anthropic-Claude-purple.svg)](https://www.anthropic.com/)
[![Google](https://img.shields.io/badge/Google-Gemini-red.svg)](https://ai.google.dev/)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow.svg)](https://huggingface.co/)
[![CodeQwen](https://img.shields.io/badge/Qwen-CodeQwen1.5-cyan.svg)](https://huggingface.co/Qwen)
[![CodeGemma](https://img.shields.io/badge/Google-CodeGemma-gold.svg)](https://huggingface.co/google)
[![Gradio](https://img.shields.io/badge/Gradio-Interface-orange.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

<div align="center">
<h2>📋 Table of Contents</h2>
<table>
  <tr>
    <td><a href="#features">✨ Features</a></td>
    <td><a href="#supported-models">🤖 Supported Models</a></td>
    <td><a href="#requirements">📦 Requirements</a></td>
    <td><a href="#installation">🔧 Installation</a></td>
  </tr>
  <tr>
    <td><a href="#configuration">⚙️ Configuration</a></td>
    <td><a href="#usage">🎮 Usage</a></td>
    <td><a href="#architecture">🏗️ Architecture</a></td>
    <td><a href="#examples">💡 Examples</a></td>
  </tr>
  <tr>
    <td><a href="#trading-functions">📊 Trading Functions</a></td>
    <td><a href="#data-sources">📡 Data Sources</a></td>
    <td><a href="#troubleshooting">🐛 Troubleshooting</a></td>
    <td><a href="#license">📄 License</a></td>
  </tr>
</table>
</div>

---

## ✨ Features

### 🤖 **Multi-Model Support**

#### Frontier AI Models
- **OpenAI GPT-5** - Fast code generation for trading strategies
- **Anthropic Claude Sonnet 4** - Superior strategy logic and optimization
- **Google Gemini 2.5 Pro** - Advanced algorithmic understanding and generation

#### Open-Source Models (HuggingFace)
- **CodeQwen 1.5 7B** - Specialized for algorithmic code understanding
- **CodeGemma 7B** - Lightweight inference for strategy generation

### 🎯 **Algorithmic Trading Strategy Generation**
- Automatic generation of unique trading strategies
- Support for multiple strategy types:
  - Momentum / trend-following
  - Mean reversion
  - Moving average crossovers
  - RSI-based strategies
  - Volatility and risk-managed strategies
  - Random/Monte Carlo approaches
- Strategy comparison and backtesting

### 📊 **Market Data Integration**
- Real OHLCV (Open, High, Low, Close, Volume) data from:
  - Yahoo Finance via `yfinance` library (no API key required)
  - Stooq via direct CSV URLs (no API key required)
- Historical price data fetching and parsing
- Support for multiple stock tickers


### 🖥️ **Interactive Web UI**
- Beautiful Gradio interface with code input/output
- Model selection dropdown for easy switching
- Real-time strategy generation and streaming
- Clean, user-friendly design

### 🔀 **Streaming Response**
- Real-time streaming for all AI models
- Progressive code generation display
- Fast feedback during strategy generation

---

## 🤖 Supported Models

### Frontier AI Models (Cloud-Based)

| Model | Provider | Speed | Quality | Context |
|-------|----------|-------|---------|---------|
| GPT-5 | OpenAI | ⚡⚡⚡ | ⭐⭐⭐⭐ | 128K |
| Claude Sonnet 4 | Anthropic | ⚡⚡ | ⭐⭐⭐⭐⭐ | 200K |
| Gemini 2.5 Pro | Google | ⚡⚡ | ⭐⭐⭐⭐⭐ | 1M |

### Open-Source Models (HuggingFace)

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| CodeQwen 1.5 | 7B | ⚡⚡⚡ | ⭐⭐⭐⭐ | Balanced |
| CodeGemma | 7B | ⚡⚡⚡ | ⭐⭐⭐⭐ | Lightweight inference |

---

## 📦 Requirements

### System Requirements
- **Python 3.8+**
- **8GB+ RAM** (for data processing and model inference)

### Python Dependencies
```
openai>=1.0.0
anthropic>=0.7.0
google-generativeai>=0.3.0
python-dotenv>=0.21.0
gradio>=4.0.0
ipython>=8.0.0
huggingface-hub>=0.16.0
transformers>=4.30.0
torch>=2.0.0
yfinance>=0.2.0
```

---

## 🔧 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/trading-project.git
cd trading-project
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

### 4. Configure Environment Variables
Create a `.env` file in the project root:
```env
# Frontier Models
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Open-Source Models (HuggingFace)
HF_API_KEY=your_huggingface_token_here
```

---

## ⚙️ Configuration

### API Keys Setup

#### OpenAI API
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Navigate to API keys section
3. Create new secret key
4. Add to `.env` file as `OPENAI_API_KEY`

#### Anthropic API
1. Visit [Anthropic Console](https://console.anthropic.com/)
2. Generate API key
3. Add to `.env` file as `ANTHROPIC_API_KEY`

#### Google Gemini API
1. Visit [Google AI Studio](https://ai.google.dev/)
2. Create API key
3. Add to `.env` file as `GOOGLE_API_KEY`

#### HuggingFace Token
1. Visit [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create new token (read access)
3. Add to `.env` file as `HF_API_KEY`

### Model Configuration
```python
# Frontier Models
claude_model = 'claude-sonnet-4-20250514'
gpt_model = 'gpt-5-2025-08-07'
gemini_model = 'gemini-2.5-pro'

# Open-Source Models
code_qwen = 'Qwen/CodeQwen1.5-7B-Chat'
code_gemma = 'google/codegemma-7b-it'
```

---

## 🎮 Usage

### Quick Start

#### 1. Launch Gradio Interface
```bash
jupyter notebook Trading\ Project.ipynb
# Navigate to the final UI cell and run it
```

#### 2. Using the Web UI
- **Input Example Trading Function**: Paste your example trading function in the textbox (default example provided)
- **Select Model**: Choose from available models (GPT, Claude, Gemini, Qwen, Gemma)
- **Generate**: Click "Simulate Trading" button to generate five additional trading functions
- **View Output**: See generated trading strategies in the output textbox
- **Analyze**: Review the generated code for unique trading logic and implementation

### Model Selection Guide

**For Best Quality (Slowest):**
```
Claude Sonnet 4 > Gemini 2.5 Pro > GPT-5 
```

**For Fastest Speed (Good Quality):**
```
CodeQwen > GPT-5 > Gemini 2.5 Pro > Claude Sonnet 4
```

**For Balanced Performance:**
```
GPT-5 or Gemini 2.5 Pro
```

### Programmatic Usage

#### Generate Trading Functions with Claude
```python
from trading_project import stream_claude

example_function = """
def trade1():
    avg_prices = {ticker: np.mean(prices[ticker][:5]) for ticker in tickers}
    best_ticker = max(avg_prices, key=avg_prices.get)
    return [Trade(best_ticker, 100)]
"""

for chunk in stream_claude(example_function):
    print(chunk, end='', flush=True)
```

#### Generate Trading Functions with GPT
```python
from trading_project import stream_gpt

for chunk in stream_gpt(example_function):
    print(chunk, end='', flush=True)
```

#### Generate Trading Functions with Gemini
```python
from trading_project import stream_gemini

for chunk in stream_gemini(example_function):
    print(chunk, end='', flush=True)
```

#### Generate with Open-Source Models
```python
from trading_project import stream_qwen, stream_gemma

# CodeQwen
for chunk in stream_qwen(example_function):
    print(chunk, end='', flush=True)

# CodeGemma
for chunk in stream_gemma(example_function):
    print(chunk, end='', flush=True)
```

---

## 📊 System Prompt & Strategy Requirements

### System Prompt Overview
The AI system is configured as an advanced code-generation specialist for algorithmic trading systems with the following capabilities:

- Fetching historical market data from free, no-API-key sources
- Parsing OHLCV price data for analysis
- Implementing unique trading strategies
- Simulating trades and tracking PnL, equity curves
- Generating visualizations (price charts, buy/sell markers, equity curves)
- Returning Trade objects (ticker, quantity pairs)

### Trading Function Requirements

Each generated trading function must:
- Implement a **unique and distinct** trading strategy
- Use only free data sources (Yahoo Finance or Stooq)
- Include clear code comments explaining strategy logic
- Return a list of Trade objects
- Contain only Python code with no explanations

### Supported Strategy Types
- **Momentum/Trend-Following**: Buy rising assets, sell falling assets
- **Mean Reversion**: Buy oversold, sell overbought conditions
- **Moving Average Crossovers**: Use MA crossings as buy/sell signals
- **RSI-Based Strategies**: Leverage Relative Strength Index indicators
- **Volatility/Risk-Managed**: Scale position sizes based on volatility
- **Random/Monte Carlo**: Probabilistic or randomized approaches

---

## 📡 Data Sources

### Yahoo Finance
- **Access**: `import yfinance as yf`
- **No API Key Required**: Free to use
- **Data Type**: Historical OHLCV data
- **Best For**: Backtesting, charting, strategy validation
- **Usage Example**:
```python
import yfinance as yf
data = yf.download('AAPL', start='2023-01-01', end='2024-01-01')
```

### Stooq
- **Access**: Direct CSV URL format
- **Format**: `https://stooq.com/q/d/l/?s={TICKER}&i=d`
- **No API Key Required**: Free to use
- **Data Type**: Daily OHLCV data
- **Best For**: Historical data retrieval, backtesting

---

## 🏗️ Architecture

### Component Overview

```
┌──────────────────────────────────────────────────────────┐
│            Gradio Web Interface (UI Layer)               │
├──────────────────────────────────────────────────────────┤
│                                                            │
│  ┌────────────────────┐      ┌────────────────────┐     │
│  │  Example Function  │      │ Generated Functions│     │
│  │     Input          │      │    Output          │     │
│  └────────────────────┘      └────────────────────┘     │
├──────────────────────────────────────────────────────────┤
│        Multi-Model Trading Strategy Generation Layer     │
├────────────────┬──────────────────┬────────────────────┤
│                │                  │                    │
│   Frontier     │  Open-Source     │  System Prompts   │
│   Models       │  Models          │                   │
│   ──────       │  ────────        │                   │
│   • OpenAI     │  • CodeQwen      │  • User Prompt    │
│   • Anthropic  │  • Gemma         │  • System Msg     │
│   • Google     │                  │                   │
│                │                  │                   │
├────────────────┴──────────────────┴────────────────────┤
│           Strategy Streaming & Delivery Layer           │
├──────────────────────────────────────────────────────────┤
│                                                            │
│  Real-time Streaming Output & Code Generation           │
│                                                            │
├──────────────────────────────────────────────────────────┤
```

### Key Functions

| Function | Purpose | Input | Output |
|----------|---------|-------|--------|
| `optimize()` | Route to appropriate AI model | Example function, model name | Generator yielding strategy code |
| `stream_gpt()` | Stream OpenAI GPT response | Example trading function | Streaming Python code chunks |
| `stream_claude()` | Stream Anthropic Claude response | Example trading function | Streaming Python code chunks |
| `stream_gemini()` | Stream Google Gemini response | Example trading function | Streaming Python code chunks |
| `stream_qwen()` | CodeQwen strategy generation | Example trading function | Python strategy code |
| `stream_gemma()` | Gemma strategy generation | Example trading function | Streaming Python code chunks |
| `user_prompt_for()` | Build AI user prompt | Example trading function | Formatted prompt with instructions |
| `messages_for()` | Build message structure | Example trading function | Message array for API |

---

## 💡 Examples

### Example 1: Basic Momentum Strategy

#### Example Input Function
```python
import numpy as np

def trade1():
    # Buy top performing stock in the last 5 days
    avg_prices = {ticker: np.mean(prices[ticker][:5]) for ticker in tickers}
    best_ticker = max(avg_prices, key=avg_prices.get)
    trade = Trade(best_ticker, 100)
    return [trade]
```

#### Generated Strategy (Example Output)
```python
import numpy as np

def trade2():
    # Mean reversion strategy: Buy stocks with lowest prices
    # in the last 10 days, expecting price recovery
    min_prices = {ticker: min(prices[ticker][:10]) for ticker in tickers}
    worst_ticker = min(min_prices, key=min_prices.get)
    trade = Trade(worst_ticker, 100)
    return [trade]

def trade3():
    # Moving average crossover: Use 5-day and 20-day MA
    # Buy when 5-day MA crosses above 20-day MA
    ma_signals = []
    for ticker in tickers:
        ma5 = np.mean(prices[ticker][:5])
        ma20 = np.mean(prices[ticker][:20])
        if ma5 > ma20:
            ma_signals.append((ticker, ma5 - ma20))
    
    if ma_signals:
        best_ticker = max(ma_signals, key=lambda x: x[1])[0]
        return [Trade(best_ticker, 100)]
    return []

def trade4():
    # Random selection with fixed position size
    # For comparison and baseline strategy
    selected = np.random.choice(tickers)
    return [Trade(selected, 50)]

def trade5():
    # Volatility-based: Buy highest volatility stocks
    volatilities = {}
    for ticker in tickers:
        vol = np.std(prices[ticker][:10])
        volatilities[ticker] = vol
    
    most_volatile = max(volatilities, key=volatilities.get)
    return [Trade(most_volatile, 75)]

def trade6():
    # Risk-managed strategy: Scale positions by inverse volatility
    positions = []
    for ticker in tickers:
        vol = np.std(prices[ticker][:10])
        if vol > 0:
            size = int(100 / (1 + vol))
            positions.append((ticker, size))
    
    if positions:
        best_ticker = max(positions, key=lambda x: x[1])[0]
        return [Trade(best_ticker, positions[0][1])]
    return []
```

---

## 📊 Trading Functions

### Function Structure
All trading functions follow this pattern:
```python
def trade_n():
    # Strategy description as comment
    # Logic implementation
    # Buy/Sell decision making
    return [Trade(ticker, quantity), ...]  # Return list of Trade objects
```

### Trade Object Format
- **Trade(ticker, quantity)**: Represents a trading decision
  - `quantity > 0`: BUY signal (long position)
  - `quantity < 0`: SELL signal (short position or exit)
  - `ticker`: Stock symbol as string

### Generated Function Requirements
The system generates exactly 5 additional functions:
- `trade2()` through `trade6()`
- Each with unique strategy logic
- Each using real historical price data
- Each returning list of Trade objects

---

## 🐛 Troubleshooting

### Issue: "API Key not found"
**Solution**: Verify `.env` file exists in project root and contains correct keys
```bash
cat .env  # Check file contents
```

### Issue: "Module not found" errors
**Solution**: Install all dependencies from requirements.txt
```bash
pip install -r requirements.txt
```

### Issue: "HuggingFace authentication failed"
**Solution**: Verify HF_API_KEY in .env has correct token with read access
```bash
huggingface-cli login  # Interactive login
```

### Issue: Gradio interface won't launch
**Solution**: Ensure Gradio is installed and port 7864 is available
```bash
pip install gradio
# Check port availability or specify different port in launch()
```

### Issue: Memory issues during model inference
**Solution**: Reduce max_tokens or check available system memory
```bash
top -l 1 | grep Memory  # Monitor memory usage on macOS
```

### Issue: Data fetch fails from Yahoo Finance
**Solution**: Check internet connection and verify ticker symbols are valid
```python
import yfinance as yf
data = yf.download('AAPL', period='1y')  # Verify with simple test
```

### Issue: Generated code contains syntax errors
**Solution**: Try different model or provide clearer example function
- Claude typically produces cleaner code
- GPT is faster but may have formatting issues
- Adjust system prompt or user prompt if needed

---

## 📁 File Structure

```
trading-project/
├── Trading Project.ipynb              # Main Jupyter notebook
├── .env                               # Environment variables (git-ignored) (Locally Use)
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
└── LICENSE                            # MIT License
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**[⬆ Back to Top](#-trading-project)**

**Multi-Model AI Trading Strategy Generator**  
Made for algorithmic trading research and strategy development

</div>
