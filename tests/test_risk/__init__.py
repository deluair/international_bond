"""
Risk Management Module Tests

This package contains comprehensive tests for the risk management modules,
ensuring robust risk measurement, attribution, and stress testing capabilities.

Test Modules:
- test_var_calculator: Value at Risk calculation tests
- test_stress_tester: Stress testing and scenario analysis tests  
- test_risk_attributor: Risk attribution and decomposition tests
- test_scenario_analyzer: Economic and market scenario analysis tests

Test Coverage:
- Unit tests for individual risk metrics
- Integration tests for risk system workflows
- Performance tests for large portfolios
- Stress tests for extreme market conditions
- Validation tests against known benchmarks
"""

__version__ = "1.0.0"
__author__ = "International Bond System"

# Test configuration
TEST_CONFIDENCE_LEVELS = [0.95, 0.99, 0.999]
TEST_TIME_HORIZONS = [1, 5, 10, 22]  # Days
TEST_PORTFOLIO_SIZES = [10, 50, 100, 500]  # Number of positions

# Risk test constants
MAX_VAR_CALCULATION_TIME = 5.0  # Seconds
MAX_STRESS_TEST_TIME = 10.0     # Seconds
MIN_COVERAGE_RATIO = 0.95       # For backtesting
MAX_ATTRIBUTION_ERROR = 0.001   # 10bps tolerance