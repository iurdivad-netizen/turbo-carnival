# EURUSD Monte Carlo Trading Simulator

A comprehensive Monte Carlo simulation engine for forecasting EURUSD trading performance based on historical pattern analysis.

## Overview

This simulator models trading performance across three pattern classifications:
- **Ultra-Momentum** (4.1% of trades)
- **Standard Best Pattern** (23.1% of trades)
- **Exit Zone** (72.8% of trades)

Based on 147-trade historical sample from Sep-Dec 2025.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Execution

Run the full simulation with default parameters (10,000 simulations):

```bash
python monte_carlo_simulator.py
```

### Custom Configuration

```python
from monte_carlo_simulator import MonteCarloSimulator, SimulationConfig

# Configure simulation parameters
config = SimulationConfig(
    n_simulations=5000,
    monthly_trades_mean=37.0,
    random_seed=42
)

# Run simulation
simulator = MonteCarloSimulator(config)
results = simulator.run_simulation()

# Display results
simulator.print_results()

# Generate visualizations
simulator.plot_results(save_path='results.png')
```

## Output

The simulation generates:

1. **Console Report** - Comprehensive statistics including:
   - Annual performance metrics
   - Probability analysis
   - Risk metrics (drawdown, consecutive losses)
   - Pattern contribution breakdown
   - Percentile distributions

2. **Visualization** - 12-panel chart (`monte_carlo_results.png`) showing:
   - Annual pips distribution
   - Win rate distribution
   - Maximum drawdown analysis
   - Cumulative probability curves
   - Monthly performance boxplots
   - Sample equity curves
   - Risk-return relationships

3. **Scenario Analysis** - Compares pessimistic, base case, and optimistic scenarios

## Key Metrics

- **Mean Annual Pips**: Expected annual performance
- **Win Rate**: Percentage of profitable trades
- **Max Drawdown**: Largest peak-to-trough decline
- **Probability Analysis**: P(Profit > X pips)
- **Pattern Contribution**: Pips attribution by pattern type

## Configuration Parameters

See `SimulationConfig` class for full parameter documentation including:
- Trade frequency distributions
- Pattern classification probabilities
- Outcome probabilities and pip ranges for each pattern type
- Risk parameters (exits, drawdowns)

## Requirements

- Python 3.8+
- NumPy >= 1.24.0
- Matplotlib >= 3.7.0
- SciPy >= 1.10.0
