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

### Option 1: Run on GitHub Actions (Recommended)

The easiest way to run simulations without installing anything locally:

1. Go to your repository on GitHub
2. Click the **Actions** tab
3. Select **Monte Carlo Trading Simulation** workflow
4. Click **Run workflow** button
5. Enter **custom simulation count** (1 to 100,000)
   - Common values: 1,000 | 5,000 | 10,000 | 25,000 | 50,000 | 100,000
   - Or enter any value between 1 and 100,000
6. Choose whether to include **scenario analysis** (optional, unchecking saves time)
7. Click **Run workflow**
8. Wait for completion (time varies by simulation count)
9. Download results from **Artifacts** section

The workflow generates:
- Full text output with all statistics
- 12-panel visualization chart (PNG)
- Results preserved for 30 days

**Estimated run times:**
- 1,000 sims: ~1 minute
- 10,000 sims: ~3-5 minutes
- 25,000 sims: ~8-12 minutes
- 50,000 sims: ~20-30 minutes
- 100,000 sims: ~40-60 minutes

### Option 2: Local Execution

Run the full simulation with default parameters (10,000 simulations):

```bash
python monte_carlo_simulator.py
```

**Command-line options:**

```bash
# Run with custom simulation count
python monte_carlo_simulator.py -n 25000

# Run 50k simulations without scenario analysis (faster)
python monte_carlo_simulator.py -n 50000 --no-scenario

# Custom output filename and random seed
python monte_carlo_simulator.py -n 10000 --seed 123 -o my_results.png

# Show all options
python monte_carlo_simulator.py --help
```

**Supported simulation counts:**
- `1,000` - Quick test (~30 seconds)
- `5,000` - Standard run (~1-2 minutes)
- `10,000` - Default/recommended (~2-3 minutes)
- `25,000` - High precision (~5-8 minutes)
- `100,000` - Maximum (~30+ minutes)

### Option 3: Custom Configuration

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
