# International Bond Relative Value System

A comprehensive system for comparing sovereign bonds across countries on a risk-adjusted basis, incorporating credit default swap curves, currency overlay analysis, and central bank policy divergence.

## Features

- **Sovereign Bond Analysis**: Compare bonds across different countries with risk-adjusted metrics
- **Credit Default Swap Integration**: Incorporate CDS curves for enhanced credit risk assessment
- **Currency Overlay**: Advanced currency hedging and overlay strategies
- **Duration-Neutral Strategies**: Build portfolios with controlled interest rate risk
- **Central Bank Policy Analysis**: Factor in monetary policy divergence across regions
- **Risk-Adjusted Comparisons**: Comprehensive relative value framework
- **Cross-Market Analysis**: Identify arbitrage opportunities across different markets
- **Yield Curve Analytics**: Advanced yield curve fitting and analysis
- **Portfolio Optimization**: Risk-parity and duration-neutral portfolio construction

## Project Structure

```
international_bond/
├── src/
│   ├── models/           # Data models for bonds, CDS, currencies, portfolios
│   ├── pricing/          # Bond pricing and yield curve calculations
│   ├── analytics/        # Relative value and cross-market analysis
│   ├── risk/            # Risk management and scenario analysis
│   ├── cds/             # Credit default swap curve analysis
│   ├── currency/        # Currency overlay and hedging strategies
│   ├── policy/          # Central bank policy analysis
│   ├── portfolio/       # Portfolio optimization algorithms
│   ├── strategies/      # Trading strategies and signal generation
│   └── utils/           # Utility functions and helpers
├── data/                # Sample data and market feeds
├── tests/               # Comprehensive test suite
├── web/                 # Web interface components
└── docs/                # Documentation
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd international_bond

# Install dependencies
pip install -r requirements.txt

# Run tests to verify installation
python -m pytest tests/ -v
```

## Usage

### Command Line Interface

```bash
# View available options
python main.py --help

# Run demonstration analysis
python main.py --mode demo --verbose

# Run comprehensive analysis
python main.py --mode analysis --output results.json

# Run risk analysis
python main.py --mode risk --config config.yaml
```

### Python API

```python
from src.models.bond import SovereignBond
from src.models.portfolio import Portfolio
from src.pricing.bond_pricer import BondPricer
from src.analytics.relative_value_analyzer import RelativeValueAnalyzer

# Create bonds
us_bond = SovereignBond(
    isin="US912828XG55",
    country="US",
    currency="USD",
    face_value=1000,
    coupon_rate=0.025,
    maturity_date=date(2030, 5, 15),
    issuer="US Treasury"
)

# Analyze relative value
analyzer = RelativeValueAnalyzer()
results = analyzer.analyze_relative_value(inputs)

# Build portfolio
portfolio = Portfolio(name="International Bonds")
portfolio.add_position(us_bond, quantity=100)
```

## Key Components

### Bond Models
- **SovereignBond**: Comprehensive sovereign bond representation
- **Portfolio**: Multi-bond portfolio with risk analytics
- **CreditRating**: Credit rating system integration

### Analytics
- **RelativeValueAnalyzer**: Cross-bond comparison and spread analysis
- **CrossMarketAnalyzer**: Arbitrage opportunity identification
- **RiskAdjustedComparator**: Risk-adjusted performance metrics

### Pricing & Risk
- **BondPricer**: Advanced bond pricing with yield curve integration
- **YieldCurve**: Yield curve construction and interpolation
- **VaRCalculator**: Value-at-Risk and stress testing

## Testing

The project includes a comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_models.py -v
python -m pytest tests/test_demo.py -v
```

## Development Status

✅ **Core Models**: Bond, Portfolio, Currency models implemented  
✅ **Pricing Engine**: Bond pricing and yield curve analytics  
✅ **Analytics Framework**: Relative value and cross-market analysis  
✅ **Risk Management**: VaR calculation and stress testing  
✅ **Test Suite**: Comprehensive unit and integration tests  
✅ **CLI Interface**: Command-line interface for analysis modes  

## License

MIT License