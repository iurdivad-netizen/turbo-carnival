import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from dataclasses import dataclass
from typing import List, Dict, Tuple
import warnings
import argparse
import sys
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
#                         CONFIGURATION & PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimulationConfig:
    """Core simulation parameters from historical analysis"""

    # Trade Frequency Parameters
    monthly_trades_mean: float = 37.0
    monthly_trades_std: float = 7.8
    monthly_trades_min: int = 25
    monthly_trades_max: int = 50

    # Classification Probabilities
    prob_ultra_momentum: float = 0.041     # 4.1%
    prob_standard_best: float = 0.231      # 23.1%
    prob_exit_zone: float = 0.728          # 72.8%

    # Ultra-Momentum Outcomes
    ultra_both_tp_prob: float = 0.75
    ultra_both_tp_pips: float = 48.0
    ultra_both_tp_std: float = 4.0

    ultra_t1tp_t2sl_prob: float = 0.15
    ultra_t1tp_t2sl_pips: float = 18.0
    ultra_t1tp_t2sl_std: float = 2.0

    ultra_t1sl_t2tp_prob: float = 0.10
    ultra_t1sl_t2tp_pips: float = 18.0
    ultra_t1sl_t2tp_std: float = 2.0

    ultra_both_sl_prob: float = 0.00
    ultra_both_sl_pips: float = -14.0

    # Standard Best Pattern Outcomes
    standard_both_tp_prob: float = 0.647
    standard_both_tp_pips: float = 47.0
    standard_both_tp_std: float = 6.5

    standard_t1tp_t2sl_prob: float = 0.206
    standard_t1tp_t2sl_pips: float = 17.0
    standard_t1tp_t2sl_std: float = 3.2

    standard_t1sl_t2tp_prob: float = 0.118
    standard_t1sl_t2tp_pips: float = 18.0
    standard_t1sl_t2tp_std: float = 2.8

    standard_both_sl_prob: float = 0.029
    standard_both_sl_pips: float = -14.0

    # Exit Zone Parameters
    exit_mean: float = 0.89
    exit_std: float = 5.2
    exit_min: float = -8.0
    exit_max: float = 16.0

    # Simulation Settings
    n_simulations: int = 10000
    n_months: int = 12
    random_seed: int = 42


# ═══════════════════════════════════════════════════════════════════════════════
#                              SIMULATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class MonteCarloSimulator:
    """
    Monte Carlo simulation engine for EURUSD trading system
    Based on 147-trade historical sample (Sep-Dec 2025)
    """

    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
        self.results = None

    def generate_monthly_trades(self) -> int:
        """Generate number of trades for a month using truncated normal"""
        n_trades = np.random.normal(
            self.config.monthly_trades_mean,
            self.config.monthly_trades_std
        )
        return int(np.clip(
            n_trades,
            self.config.monthly_trades_min,
            self.config.monthly_trades_max
        ))

    def classify_trade(self) -> str:
        """Classify trade into one of three tiers"""
        return np.random.choice(
            ['ultra', 'standard', 'exit'],
            p=[
                self.config.prob_ultra_momentum,
                self.config.prob_standard_best,
                self.config.prob_exit_zone
            ]
        )

    def generate_ultra_outcome(self) -> Tuple[float, str]:
        """Generate outcome for Ultra-Momentum pattern"""
        rand = np.random.random()

        if rand < self.config.ultra_both_tp_prob:
            pips = np.random.normal(
                self.config.ultra_both_tp_pips,
                self.config.ultra_both_tp_std
            )
            return pips, 'both_tp'
        elif rand < self.config.ultra_both_tp_prob + self.config.ultra_t1tp_t2sl_prob:
            pips = np.random.normal(
                self.config.ultra_t1tp_t2sl_pips,
                self.config.ultra_t1tp_t2sl_std
            )
            return pips, 't1tp_t2sl'
        elif rand < 1.0 - self.config.ultra_both_sl_prob:
            pips = np.random.normal(
                self.config.ultra_t1sl_t2tp_pips,
                self.config.ultra_t1sl_t2tp_std
            )
            return pips, 't1sl_t2tp'
        else:
            return self.config.ultra_both_sl_pips, 'both_sl'

    def generate_standard_outcome(self) -> Tuple[float, str]:
        """Generate outcome for Standard Best Pattern"""
        rand = np.random.random()

        cumulative = 0.0

        cumulative += self.config.standard_both_tp_prob
        if rand < cumulative:
            pips = np.random.normal(
                self.config.standard_both_tp_pips,
                self.config.standard_both_tp_std
            )
            return pips, 'both_tp'

        cumulative += self.config.standard_t1tp_t2sl_prob
        if rand < cumulative:
            pips = np.random.normal(
                self.config.standard_t1tp_t2sl_pips,
                self.config.standard_t1tp_t2sl_std
            )
            return pips, 't1tp_t2sl'

        cumulative += self.config.standard_t1sl_t2tp_prob
        if rand < cumulative:
            pips = np.random.normal(
                self.config.standard_t1sl_t2tp_pips,
                self.config.standard_t1sl_t2tp_std
            )
            return pips, 't1sl_t2tp'

        return self.config.standard_both_sl_pips, 'both_sl'

    def generate_exit_outcome(self) -> Tuple[float, str]:
        """Generate outcome for Exit Zone trades"""
        pips = np.random.normal(
            self.config.exit_mean,
            self.config.exit_std
        )
        # Apply realistic bounds
        pips = np.clip(pips, self.config.exit_min, self.config.exit_max)
        return pips, 'exit_60min'

    def simulate_single_trade(self) -> Dict:
        """Simulate a single trade with full tracking"""
        classification = self.classify_trade()

        if classification == 'ultra':
            pips, outcome = self.generate_ultra_outcome()
        elif classification == 'standard':
            pips, outcome = self.generate_standard_outcome()
        else:
            pips, outcome = self.generate_exit_outcome()

        return {
            'classification': classification,
            'outcome': outcome,
            'pips': pips,
            'is_win': pips > 0,
            'is_best_pattern': classification in ['ultra', 'standard']
        }

    def simulate_single_year(self) -> Dict:
        """Simulate a complete trading year"""

        trades = []
        monthly_pips = []
        monthly_trades = []

        # Simulate each month
        for month in range(self.config.n_months):
            n_trades = self.generate_monthly_trades()
            month_results = []

            for _ in range(n_trades):
                trade = self.simulate_single_trade()
                trades.append(trade)
                month_results.append(trade['pips'])

            monthly_pips.append(sum(month_results))
            monthly_trades.append(n_trades)

        # Extract metrics
        all_pips = [t['pips'] for t in trades]

        # Calculate equity curve and drawdown
        equity_curve = np.cumsum(all_pips)
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = equity_curve - running_max
        max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0

        # Count consecutive losses
        max_consecutive_losses = self._count_max_consecutive_losses(trades)

        # Count negative months
        negative_months = sum(1 for p in monthly_pips if p < 0)

        # Classification breakdown
        ultra_count = sum(1 for t in trades if t['classification'] == 'ultra')
        standard_count = sum(1 for t in trades if t['classification'] == 'standard')
        exit_count = sum(1 for t in trades if t['classification'] == 'exit')

        # Pattern-specific stats
        ultra_pips = sum(t['pips'] for t in trades if t['classification'] == 'ultra')
        standard_pips = sum(t['pips'] for t in trades if t['classification'] == 'standard')
        exit_pips = sum(t['pips'] for t in trades if t['classification'] == 'exit')

        # Calculate new metrics: Expectancy, Profit Factor, Recovery Factor
        winning_trades = [t['pips'] for t in trades if t['is_win']]
        losing_trades = [t['pips'] for t in trades if not t['is_win']]

        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = abs(np.mean(losing_trades)) if losing_trades else 0
        win_rate = len(winning_trades) / len(trades) if trades else 0

        # Expectancy: (Avg Win × Win Rate) - (Avg Loss × Loss Rate)
        expectancy = (avg_win * win_rate) - (avg_loss * (1 - win_rate))

        # Profit Factor: Total Wins / Total Losses
        total_wins = sum(winning_trades) if winning_trades else 0
        total_losses = abs(sum(losing_trades)) if losing_trades else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Recovery Factor: Net Profit / Max Drawdown
        net_profit = sum(all_pips)
        recovery_factor = net_profit / max_drawdown if max_drawdown > 0 else float('inf')

        # Sortino Ratio: Return / Downside Deviation
        # Only penalize downside volatility (negative returns)
        negative_returns = [p for p in all_pips if p < 0]
        if negative_returns and len(negative_returns) > 1:
            downside_deviation = np.std(negative_returns)
            sortino_ratio = net_profit / downside_deviation if downside_deviation > 0 else float('inf')
        else:
            # Not enough negative data - use a placeholder
            sortino_ratio = 999.9  # Very high but not inf

        # Drawdown Duration: How long equity stays below previous peak
        drawdown_durations = []
        current_dd_duration = 0
        for dd in drawdowns:
            if dd < 0:  # In drawdown
                current_dd_duration += 1
            else:  # At new peak
                if current_dd_duration > 0:
                    drawdown_durations.append(current_dd_duration)
                    current_dd_duration = 0
        # Add final drawdown if still in one
        if current_dd_duration > 0:
            drawdown_durations.append(current_dd_duration)

        avg_dd_duration = np.mean(drawdown_durations) if drawdown_durations else 0
        max_dd_duration = max(drawdown_durations) if drawdown_durations else 0

        # Calmar Ratio: Annual Return / Max Drawdown
        # Industry standard for hedge fund performance
        calmar_ratio = net_profit / max_drawdown if max_drawdown > 0 else float('inf')

        # Consecutive Wins Analysis
        max_consecutive_wins = self._count_max_consecutive_wins(trades)
        avg_win_streak = self._calculate_avg_win_streak(trades)

        return {
            'annual_pips': sum(all_pips),
            'total_trades': len(trades),
            'win_rate': np.mean([t['is_win'] for t in trades]),
            'max_drawdown': max_drawdown,
            'max_consecutive_losses': max_consecutive_losses,
            'negative_months': negative_months,
            'monthly_pips': monthly_pips,
            'monthly_trades': monthly_trades,
            'ultra_count': ultra_count,
            'standard_count': standard_count,
            'exit_count': exit_count,
            'ultra_pips': ultra_pips,
            'standard_pips': standard_pips,
            'exit_pips': exit_pips,
            'best_pattern_rate': (ultra_count + standard_count) / len(trades),
            'equity_curve': equity_curve.tolist(),
            'drawdowns': drawdowns.tolist(),
            'all_trades': trades,
            # New metrics
            'expectancy': expectancy,
            'profit_factor': profit_factor,
            'recovery_factor': recovery_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sortino_ratio': sortino_ratio,
            'avg_dd_duration': avg_dd_duration,
            'max_dd_duration': max_dd_duration,
            'calmar_ratio': calmar_ratio,
            'max_consecutive_wins': max_consecutive_wins,
            'avg_win_streak': avg_win_streak,
            'drawdown_durations': drawdown_durations  # For histogram
        }

    def _count_max_consecutive_losses(self, trades: List[Dict]) -> int:
        """Count maximum consecutive losing trades"""
        max_streak = 0
        current_streak = 0

        for trade in trades:
            if not trade['is_win']:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

    def _count_max_consecutive_wins(self, trades: List[Dict]) -> int:
        """Count maximum consecutive winning trades"""
        max_streak = 0
        current_streak = 0

        for trade in trades:
            if trade['is_win']:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

    def _calculate_avg_win_streak(self, trades: List[Dict]) -> float:
        """Calculate average winning streak length"""
        win_streaks = []
        current_streak = 0

        for trade in trades:
            if trade['is_win']:
                current_streak += 1
            else:
                if current_streak > 0:
                    win_streaks.append(current_streak)
                    current_streak = 0

        # Add final streak if still in one
        if current_streak > 0:
            win_streaks.append(current_streak)

        return np.mean(win_streaks) if win_streaks else 0

    def run_simulation(self, n_simulations: int = None) -> Dict:
        """Run full Monte Carlo simulation"""

        if n_simulations is not None:
            self.config.n_simulations = n_simulations

        np.random.seed(self.config.random_seed)

        print("=" * 70)
        print("     MONTE CARLO SIMULATION: EURUSD 2026 TRADING FORECAST")
        print("=" * 70)
        print(f"\n🎲 Running {self.config.n_simulations:,} simulations...")

        # Run simulations
        simulations = []
        for i in range(self.config.n_simulations):
            if (i + 1) % 2500 == 0:
                print(f"   Progress: {i + 1:,}/{self.config.n_simulations:,} ({100*(i+1)/self.config.n_simulations:.0f}%)")
            simulations.append(self.simulate_single_year())

        print(f"   ✓ Completed {self.config.n_simulations:,} simulations\n")

        # Extract arrays for analysis
        self.results = {
            'simulations': simulations,
            'annual_pips': np.array([s['annual_pips'] for s in simulations]),
            'win_rates': np.array([s['win_rate'] for s in simulations]),
            'max_drawdowns': np.array([s['max_drawdown'] for s in simulations]),
            'max_consecutive_losses': np.array([s['max_consecutive_losses'] for s in simulations]),
            'negative_months': np.array([s['negative_months'] for s in simulations]),
            'total_trades': np.array([s['total_trades'] for s in simulations]),
            'best_pattern_rates': np.array([s['best_pattern_rate'] for s in simulations]),
            'ultra_pips': np.array([s['ultra_pips'] for s in simulations]),
            'standard_pips': np.array([s['standard_pips'] for s in simulations]),
            'exit_pips': np.array([s['exit_pips'] for s in simulations]),
            # New metrics
            'expectancy': np.array([s['expectancy'] for s in simulations]),
            'profit_factor': np.array([s['profit_factor'] for s in simulations]),
            'recovery_factor': np.array([s['recovery_factor'] for s in simulations]),
            'avg_wins': np.array([s['avg_win'] for s in simulations]),
            'avg_losses': np.array([s['avg_loss'] for s in simulations]),
            'sortino_ratio': np.array([s['sortino_ratio'] for s in simulations]),
            'avg_dd_duration': np.array([s['avg_dd_duration'] for s in simulations]),
            'max_dd_duration': np.array([s['max_dd_duration'] for s in simulations]),
            'calmar_ratio': np.array([s['calmar_ratio'] for s in simulations]),
            'max_consecutive_wins': np.array([s['max_consecutive_wins'] for s in simulations]),
            'avg_win_streak': np.array([s['avg_win_streak'] for s in simulations]),
        }

        return self.results

    def print_results(self):
        """Print comprehensive simulation results"""

        if self.results is None:
            raise ValueError("Run simulation first!")

        r = self.results

        print("=" * 70)
        print("                      SIMULATION RESULTS")
        print("=" * 70)

        # Annual Performance
        print("\n📊 ANNUAL PERFORMANCE (2026 Projection)")
        print("-" * 50)
        print(f"  Mean Annual Pips:      {np.mean(r['annual_pips']):,.0f}")
        print(f"  Median Annual Pips:    {np.median(r['annual_pips']):,.0f}")
        print(f"  Std Deviation:         {np.std(r['annual_pips']):,.0f}")
        print(f"  95% CI:                [{np.percentile(r['annual_pips'], 2.5):,.0f}, {np.percentile(r['annual_pips'], 97.5):,.0f}]")
        print(f"  99% CI:                [{np.percentile(r['annual_pips'], 0.5):,.0f}, {np.percentile(r['annual_pips'], 99.5):,.0f}]")

        # Monthly Performance
        monthly_mean = np.mean(r['annual_pips']) / 12
        monthly_std = np.std(r['annual_pips']) / np.sqrt(12)
        print(f"\n  Monthly Average:       {monthly_mean:,.0f} pips")
        print(f"  Monthly Std Dev:       {monthly_std:,.0f} pips")

        # Probability Analysis
        print("\n📈 PROBABILITY ANALYSIS")
        print("-" * 50)
        prob_profit = 100 * np.mean(r['annual_pips'] > 0)
        prob_1000 = 100 * np.mean(r['annual_pips'] > 1000)
        prob_2000 = 100 * np.mean(r['annual_pips'] > 2000)
        prob_3000 = 100 * np.mean(r['annual_pips'] > 3000)
        prob_4000 = 100 * np.mean(r['annual_pips'] > 4000)
        prob_5000 = 100 * np.mean(r['annual_pips'] > 5000)

        print(f"  P(Profit > 0):         {prob_profit:.1f}%")
        print(f"  P(> 1,000 pips):       {prob_1000:.1f}%")
        print(f"  P(> 2,000 pips):       {prob_2000:.1f}%")
        print(f"  P(> 3,000 pips):       {prob_3000:.1f}%")
        print(f"  P(> 4,000 pips):       {prob_4000:.1f}%")
        print(f"  P(> 5,000 pips):       {prob_5000:.1f}%")

        # Risk Metrics
        print("\n⚠️ RISK METRICS")
        print("-" * 50)
        print(f"  Mean Max Drawdown:     {np.mean(r['max_drawdowns']):,.0f} pips")
        print(f"  Median Max Drawdown:   {np.median(r['max_drawdowns']):,.0f} pips")
        print(f"  95th %ile Drawdown:    {np.percentile(r['max_drawdowns'], 95):,.0f} pips")
        print(f"  99th %ile Drawdown:    {np.percentile(r['max_drawdowns'], 99):,.0f} pips")
        print(f"  Max Observed DD:       {np.max(r['max_drawdowns']):,.0f} pips")

        print(f"\n  Avg Max Consec Losses: {np.mean(r['max_consecutive_losses']):.1f}")
        print(f"  95th %ile Consec Loss: {np.percentile(r['max_consecutive_losses'], 95):.0f}")
        print(f"  Max Observed Streak:   {np.max(r['max_consecutive_losses']):.0f}")

        print(f"\n  Avg Negative Months:   {np.mean(r['negative_months']):.1f}")
        print(f"  95th %ile Neg Months:  {np.percentile(r['negative_months'], 95):.0f}")

        # Trade Statistics
        print("\n📋 TRADE STATISTICS")
        print("-" * 50)
        print(f"  Mean Total Trades:     {np.mean(r['total_trades']):,.0f}")
        print(f"  Mean Win Rate:         {100*np.mean(r['win_rates']):.1f}%")
        print(f"  Win Rate 95% CI:       [{100*np.percentile(r['win_rates'], 2.5):.1f}%, {100*np.percentile(r['win_rates'], 97.5):.1f}%]")
        print(f"  Mean Best Pattern %:   {100*np.mean(r['best_pattern_rates']):.1f}%")

        # Pattern Contribution
        print("\n🎯 PATTERN CONTRIBUTION TO PIPS")
        print("-" * 50)
        ultra_contribution = np.mean(r['ultra_pips'])
        standard_contribution = np.mean(r['standard_pips'])
        exit_contribution = np.mean(r['exit_pips'])
        total = ultra_contribution + standard_contribution + exit_contribution

        print(f"  Ultra-Momentum:        {ultra_contribution:,.0f} pips ({100*ultra_contribution/total:.1f}%)")
        print(f"  Standard Best:         {standard_contribution:,.0f} pips ({100*standard_contribution/total:.1f}%)")
        print(f"  Exit Zone:             {exit_contribution:,.0f} pips ({100*exit_contribution/total:.1f}%)")

        # Percentile Table
        print("\n📊 PERCENTILE DISTRIBUTION")
        print("-" * 50)
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            val = np.percentile(r['annual_pips'], p)
            print(f"  {p:3d}th Percentile:      {val:,.0f} pips")

        # Sharpe-like Ratio (pips-based)
        sharpe_like = np.mean(r['annual_pips']) / np.std(r['annual_pips'])
        print(f"\n📐 RISK-ADJUSTED METRICS")
        print("-" * 50)
        print(f"  Pips Sharpe Ratio:     {sharpe_like:.2f}")
        print(f"  (Mean / Std Dev)")

        # New Advanced Metrics
        print(f"\n💡 ADVANCED PERFORMANCE METRICS")
        print("-" * 50)

        # Expectancy
        mean_expectancy = np.mean(r['expectancy'])
        median_expectancy = np.median(r['expectancy'])
        print(f"  Mean Expectancy:       {mean_expectancy:.2f} pips/trade")
        print(f"  Median Expectancy:     {median_expectancy:.2f} pips/trade")
        print(f"  Expectancy 95% CI:     [{np.percentile(r['expectancy'], 2.5):.2f}, {np.percentile(r['expectancy'], 97.5):.2f}]")

        # Profit Factor
        mean_pf = np.mean(r['profit_factor'][r['profit_factor'] < 1000])  # Filter out inf values
        median_pf = np.median(r['profit_factor'][r['profit_factor'] < 1000])
        print(f"\n  Mean Profit Factor:    {mean_pf:.2f}")
        print(f"  Median Profit Factor:  {median_pf:.2f}")
        print(f"  PF 95% CI:             [{np.percentile(r['profit_factor'][r['profit_factor'] < 1000], 2.5):.2f}, {np.percentile(r['profit_factor'][r['profit_factor'] < 1000], 97.5):.2f}]")
        pf_above_2 = 100 * np.mean(r['profit_factor'] >= 2.0)
        print(f"  P(PF ≥ 2.0):           {pf_above_2:.1f}% (excellent threshold)")

        # Recovery Factor
        mean_rf = np.mean(r['recovery_factor'][r['recovery_factor'] < 1000])  # Filter out inf values
        median_rf = np.median(r['recovery_factor'][r['recovery_factor'] < 1000])
        print(f"\n  Mean Recovery Factor:  {mean_rf:.1f}")
        print(f"  Median Recovery Factor:{median_rf:.1f}")
        print(f"  RF 95% CI:             [{np.percentile(r['recovery_factor'][r['recovery_factor'] < 1000], 2.5):.1f}, {np.percentile(r['recovery_factor'][r['recovery_factor'] < 1000], 97.5):.1f}]")

        # Win/Loss Ratio
        avg_win_loss_ratio = np.mean(r['avg_wins'] / r['avg_losses'])
        print(f"\n  Avg Win/Loss Ratio:    {avg_win_loss_ratio:.2f}")
        print(f"  Mean Win Size:         {np.mean(r['avg_wins']):.1f} pips")
        print(f"  Mean Loss Size:        {np.mean(r['avg_losses']):.1f} pips")

        # Sortino Ratio
        mean_sortino = np.mean(r['sortino_ratio'][r['sortino_ratio'] < 1000])
        median_sortino = np.median(r['sortino_ratio'][r['sortino_ratio'] < 1000])
        print(f"\n  Mean Sortino Ratio:    {mean_sortino:.2f}")
        print(f"  Median Sortino Ratio:  {median_sortino:.2f}")
        print(f"  (Better than Sharpe: only penalizes downside)")

        # Drawdown Duration
        print(f"\n  Avg DD Duration:       {np.mean(r['avg_dd_duration']):.1f} trades")
        print(f"  Median DD Duration:    {np.median(r['avg_dd_duration']):.1f} trades")
        print(f"  Max DD Duration (95th):{np.percentile(r['max_dd_duration'], 95):.0f} trades")
        print(f"  Worst Case DD Duration:{np.max(r['max_dd_duration']):.0f} trades")

        # Calmar Ratio
        mean_calmar = np.mean(r['calmar_ratio'][r['calmar_ratio'] < 1000])
        median_calmar = np.median(r['calmar_ratio'][r['calmar_ratio'] < 1000])
        print(f"\n  Mean Calmar Ratio:     {mean_calmar:.2f}")
        print(f"  Median Calmar Ratio:   {median_calmar:.2f}")
        print(f"  (Return/Drawdown - hedge fund standard)")

        # Consecutive Wins
        print(f"\n  Avg Max Win Streak:    {np.mean(r['max_consecutive_wins']):.1f}")
        print(f"  95th %ile Win Streak:  {np.percentile(r['max_consecutive_wins'], 95):.0f}")
        print(f"  Max Observed Streak:   {np.max(r['max_consecutive_wins']):.0f}")
        print(f"  Avg Win Streak Length: {np.mean(r['avg_win_streak']):.1f}")

        print("\n" + "=" * 70)

    def plot_results(self, save_path: str = None):
        """Generate comprehensive visualization"""

        if self.results is None:
            raise ValueError("Run simulation first!")

        r = self.results

        fig = plt.figure(figsize=(24, 30))

        # Title
        fig.suptitle('Monte Carlo Simulation: EURUSD 2026 Forecast\n(Based on 147-trade historical sample)',
                     fontsize=16, fontweight='bold', y=0.995)

        # 1. Annual Pips Distribution
        ax1 = fig.add_subplot(5, 4, 1)
        ax1.hist(r['annual_pips'], bins=60, density=True, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.axvline(np.mean(r['annual_pips']), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(r["annual_pips"]):,.0f}')
        ax1.axvline(np.median(r['annual_pips']), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(r["annual_pips"]):,.0f}')
        ax1.axvline(np.percentile(r['annual_pips'], 2.5), color='green', linestyle=':', linewidth=1.5, label=f'2.5th: {np.percentile(r["annual_pips"], 2.5):,.0f}')
        ax1.axvline(np.percentile(r['annual_pips'], 97.5), color='green', linestyle=':', linewidth=1.5, label=f'97.5th: {np.percentile(r["annual_pips"], 97.5):,.0f}')
        ax1.set_xlabel('Annual Pips', fontsize=10)
        ax1.set_ylabel('Density', fontsize=10)
        ax1.set_title('Annual Pips Distribution', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 2. Win Rate Distribution
        ax2 = fig.add_subplot(5, 4, 2)
        ax2.hist(100*r['win_rates'], bins=40, density=True, alpha=0.7, color='forestgreen', edgecolor='black')
        ax2.axvline(100*np.mean(r['win_rates']), color='red', linestyle='--', linewidth=2, label=f'Mean: {100*np.mean(r["win_rates"]):.1f}%')
        ax2.set_xlabel('Win Rate (%)', fontsize=10)
        ax2.set_ylabel('Density', fontsize=10)
        ax2.set_title('Win Rate Distribution', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # 3. Max Drawdown Distribution
        ax3 = fig.add_subplot(5, 4, 3)
        ax3.hist(r['max_drawdowns'], bins=50, density=True, alpha=0.7, color='crimson', edgecolor='black')
        ax3.axvline(np.mean(r['max_drawdowns']), color='black', linestyle='--', linewidth=2, label=f'Mean: {np.mean(r["max_drawdowns"]):,.0f}')
        ax3.axvline(np.percentile(r['max_drawdowns'], 95), color='orange', linestyle='--', linewidth=2, label=f"95th: {np.percentile(r['max_drawdowns'], 95):,.0f}")
        ax3.set_xlabel('Max Drawdown (pips)', fontsize=10)
        ax3.set_ylabel('Density', fontsize=10)
        ax3.set_title('Maximum Drawdown Distribution', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        # 4. Probability Curve (Cumulative)
        ax4 = fig.add_subplot(5, 4, 4)
        sorted_pips = np.sort(r['annual_pips'])
        cumulative_prob = np.arange(1, len(sorted_pips) + 1) / len(sorted_pips)
        ax4.plot(sorted_pips, 100 * cumulative_prob, linewidth=2, color='steelblue')
        ax4.axhline(50, color='gray', linestyle=':', alpha=0.5)
        ax4.axhline(95, color='green', linestyle='--', alpha=0.7)
        ax4.axhline(5, color='red', linestyle='--', alpha=0.7)
        ax4.axvline(0, color='black', linestyle='-', linewidth=1)
        ax4.fill_between(sorted_pips, 100 * cumulative_prob, alpha=0.3)
        ax4.set_xlabel('Annual Pips', fontsize=10)
        ax4.set_ylabel('Cumulative Probability (%)', fontsize=10)
        ax4.set_title('Cumulative Distribution Function', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        # 5. Box Plot of Monthly Pips
        ax5 = fig.add_subplot(5, 4, 5)
        monthly_data = np.array([s['monthly_pips'] for s in r['simulations']])
        bp = ax5.boxplot([monthly_data[:, i] for i in range(12)],
                         labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                         patch_artist=True)
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, 12))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax5.axhline(0, color='red', linestyle='--', linewidth=1)
        ax5.set_xlabel('Month', fontsize=10)
        ax5.set_ylabel('Pips', fontsize=10)
        ax5.set_title('Monthly Performance Distribution', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)

        # 6. Sample Equity Curves
        ax6 = fig.add_subplot(5, 4, 6)
        np.random.seed(42)
        sample_indices = np.random.choice(len(r['simulations']), size=min(100, len(r['simulations'])), replace=False)
        for idx in sample_indices:
            equity = r['simulations'][idx]['equity_curve']
            alpha = 0.1 if len(sample_indices) > 50 else 0.3
            ax6.plot(equity, alpha=alpha, linewidth=0.5, color='steelblue')

        # Plot mean equity curve
        all_curves = [s['equity_curve'] for s in r['simulations']]
        min_len = min(len(c) for c in all_curves)
        truncated_curves = np.array([c[:min_len] for c in all_curves])
        mean_curve = np.mean(truncated_curves, axis=0)
        ax6.plot(mean_curve, color='red', linewidth=2, label='Mean Path')
        ax6.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax6.set_xlabel('Trade Number', fontsize=10)
        ax6.set_ylabel('Cumulative Pips', fontsize=10)
        ax6.set_title('Sample Equity Curves (100 paths)', fontsize=12, fontweight='bold')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)

        # 7. Consecutive Losses Distribution
        ax7 = fig.add_subplot(5, 4, 7)
        unique_streaks, counts = np.unique(r['max_consecutive_losses'], return_counts=True)
        ax7.bar(unique_streaks, counts / len(r['max_consecutive_losses']) * 100,
                color='coral', edgecolor='black', alpha=0.7)
        ax7.set_xlabel('Max Consecutive Losses', fontsize=10)
        ax7.set_ylabel('Frequency (%)', fontsize=10)
        ax7.set_title('Maximum Losing Streak Distribution', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3)

        # 8. Negative Months Distribution
        ax8 = fig.add_subplot(5, 4, 8)
        unique_neg, counts_neg = np.unique(r['negative_months'], return_counts=True)
        ax8.bar(unique_neg, counts_neg / len(r['negative_months']) * 100,
                color='salmon', edgecolor='black', alpha=0.7)
        ax8.set_xlabel('Number of Negative Months', fontsize=10)
        ax8.set_ylabel('Frequency (%)', fontsize=10)
        ax8.set_title('Negative Months per Year Distribution', fontsize=12, fontweight='bold')
        ax8.grid(True, alpha=0.3)

        # 9. Pattern Contribution Pie Chart
        ax9 = fig.add_subplot(5, 4, 9)
        ultra_contribution = np.mean(r['ultra_pips'])
        standard_contribution = np.mean(r['standard_pips'])
        exit_contribution = np.mean(r['exit_pips'])

        sizes = [max(0, ultra_contribution), max(0, standard_contribution), max(0, exit_contribution)]
        labels = [f'Ultra-Momentum\n{ultra_contribution:,.0f} pips',
                  f'Standard Best\n{standard_contribution:,.0f} pips',
                  f'Exit Zone\n{exit_contribution:,.0f} pips']
        colors = ['gold', 'steelblue', 'lightgray']
        explode = (0.05, 0, 0)

        ax9.pie(sizes, labels=labels, colors=colors, explode=explode,
                autopct='%1.1f%%', startangle=90, shadow=True)
        ax9.set_title('Pattern Contribution to Total Pips', fontsize=12, fontweight='bold')

        # 10. Probability Threshold Chart
        ax10 = fig.add_subplot(5, 4, 10)
        thresholds = np.arange(0, 5500, 100)
        probabilities = [100 * np.mean(r['annual_pips'] > t) for t in thresholds]
        ax10.plot(thresholds, probabilities, linewidth=2, color='steelblue')
        ax10.fill_between(thresholds, probabilities, alpha=0.3)
        ax10.axhline(50, color='gray', linestyle='--', alpha=0.5)
        ax10.axhline(90, color='green', linestyle=':', alpha=0.7, label='90% threshold')
        ax10.axhline(10, color='red', linestyle=':', alpha=0.7, label='10% threshold')
        ax10.set_xlabel('Annual Pips Threshold', fontsize=10)
        ax10.set_ylabel('Probability of Exceeding (%)', fontsize=10)
        ax10.set_title('Probability of Exceeding Target', fontsize=12, fontweight='bold')
        ax10.legend(fontsize=8)
        ax10.grid(True, alpha=0.3)

        # 11. Trade Count Distribution
        ax11 = fig.add_subplot(5, 4, 11)
        ax11.hist(r['total_trades'], bins=30, density=True, alpha=0.7, color='mediumpurple', edgecolor='black')
        ax11.axvline(np.mean(r['total_trades']), color='red', linestyle='--', linewidth=2,
                     label=f'Mean: {np.mean(r["total_trades"]):,.0f}')
        ax11.set_xlabel('Total Annual Trades', fontsize=10)
        ax11.set_ylabel('Density', fontsize=10)
        ax11.set_title('Annual Trade Count Distribution', fontsize=12, fontweight='bold')
        ax11.legend(fontsize=8)
        ax11.grid(True, alpha=0.3)

        # 12. Risk-Return Scatter
        ax12 = fig.add_subplot(5, 4, 12)
        ax12.scatter(r['max_drawdowns'], r['annual_pips'], alpha=0.1, s=10, color='steelblue')
        ax12.axhline(0, color='red', linestyle='--', linewidth=1)
        ax12.set_xlabel('Max Drawdown (pips)', fontsize=10)
        ax12.set_ylabel('Annual Pips', fontsize=10)
        ax12.set_title('Risk-Return Relationship', fontsize=12, fontweight='bold')
        ax12.grid(True, alpha=0.3)

        # 13. Underwater Equity Chart
        ax13 = fig.add_subplot(5, 4, 13)
        # Use a representative simulation for the underwater chart
        sample_idx = len(r['simulations']) // 2  # Middle simulation
        sample_drawdowns = r['simulations'][sample_idx]['drawdowns']
        ax13.fill_between(range(len(sample_drawdowns)), sample_drawdowns, 0,
                          where=np.array(sample_drawdowns) < 0,
                          color='crimson', alpha=0.6, label='Drawdown')
        ax13.plot(sample_drawdowns, color='darkred', linewidth=1, alpha=0.8)
        ax13.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax13.set_xlabel('Trade Number', fontsize=10)
        ax13.set_ylabel('Drawdown (pips)', fontsize=10)
        ax13.set_title('Underwater Equity Chart (Sample Path)', fontsize=12, fontweight='bold')
        ax13.legend(fontsize=8)
        ax13.grid(True, alpha=0.3)

        # 14. Monthly Performance Heatmap
        ax14 = fig.add_subplot(5, 4, 14)
        monthly_data = np.array([s['monthly_pips'] for s in r['simulations']])
        # Calculate percentiles for each month
        month_percentiles = np.percentile(monthly_data, [10, 25, 50, 75, 90], axis=0)
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        percentile_labels = ['10th', '25th', '50th', '75th', '90th']

        # Create heatmap data
        heatmap_data = month_percentiles
        im = ax14.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', interpolation='nearest')

        # Set ticks and labels
        ax14.set_xticks(np.arange(len(months)))
        ax14.set_yticks(np.arange(len(percentile_labels)))
        ax14.set_xticklabels(months, fontsize=8)
        ax14.set_yticklabels(percentile_labels, fontsize=8)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax14)
        cbar.set_label('Pips', fontsize=8)

        # Add values to cells
        for i in range(len(percentile_labels)):
            for j in range(len(months)):
                text = ax14.text(j, i, f'{heatmap_data[i, j]:.0f}',
                               ha="center", va="center", color="black", fontsize=7)

        ax14.set_title('Monthly Performance Heatmap (Percentiles)', fontsize=12, fontweight='bold')

        # 15. Expectancy Distribution
        ax15 = fig.add_subplot(5, 4, 15)
        ax15.hist(r['expectancy'], bins=40, density=True, alpha=0.7, color='teal', edgecolor='black')
        ax15.axvline(np.mean(r['expectancy']), color='red', linestyle='--', linewidth=2,
                     label=f'Mean: {np.mean(r["expectancy"]):.2f}')
        ax15.axvline(np.median(r['expectancy']), color='orange', linestyle='--', linewidth=2,
                     label=f'Median: {np.median(r["expectancy"]):.2f}')
        ax15.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax15.set_xlabel('Expectancy (pips/trade)', fontsize=10)
        ax15.set_ylabel('Density', fontsize=10)
        ax15.set_title('Trade Expectancy Distribution', fontsize=12, fontweight='bold')
        ax15.legend(fontsize=8)
        ax15.grid(True, alpha=0.3)

        # 16. Profit Factor Distribution
        ax16 = fig.add_subplot(5, 4, 16)
        # Filter out infinite values for better visualization
        pf_filtered = r['profit_factor'][r['profit_factor'] < 20]  # Cap at 20 for visualization
        ax16.hist(pf_filtered, bins=40, density=True, alpha=0.7, color='darkgreen', edgecolor='black')
        ax16.axvline(np.mean(pf_filtered), color='red', linestyle='--', linewidth=2,
                     label=f'Mean: {np.mean(pf_filtered):.2f}')
        ax16.axvline(np.median(pf_filtered), color='orange', linestyle='--', linewidth=2,
                     label=f'Median: {np.median(pf_filtered):.2f}')
        ax16.axvline(2.0, color='green', linestyle=':', linewidth=2, alpha=0.7,
                     label='2.0 (Excellent)')
        ax16.axvline(1.0, color='black', linestyle='-', linewidth=1, alpha=0.5,
                     label='1.0 (Breakeven)')
        ax16.set_xlabel('Profit Factor', fontsize=10)
        ax16.set_ylabel('Density', fontsize=10)
        ax16.set_title('Profit Factor Distribution', fontsize=12, fontweight='bold')
        ax16.legend(fontsize=7)
        ax16.grid(True, alpha=0.3)

        # 17. Drawdown Duration Distribution
        ax17 = fig.add_subplot(5, 4, 17)
        # Collect all drawdown durations from all simulations
        all_dd_durations = []
        for sim in r['simulations']:
            if 'drawdown_durations' in sim and sim['drawdown_durations']:
                all_dd_durations.extend(sim['drawdown_durations'])

        if all_dd_durations:
            ax17.hist(all_dd_durations, bins=40, density=True, alpha=0.7, color='indianred', edgecolor='black')
            ax17.axvline(np.mean(all_dd_durations), color='red', linestyle='--', linewidth=2,
                         label=f'Mean: {np.mean(all_dd_durations):.1f}')
            ax17.axvline(np.median(all_dd_durations), color='orange', linestyle='--', linewidth=2,
                         label=f'Median: {np.median(all_dd_durations):.1f}')
            ax17.set_xlabel('Drawdown Duration (trades)', fontsize=10)
            ax17.set_ylabel('Density', fontsize=10)
            ax17.set_title('Drawdown Duration Distribution', fontsize=12, fontweight='bold')
            ax17.legend(fontsize=8)
            ax17.grid(True, alpha=0.3)
        else:
            ax17.text(0.5, 0.5, 'No Drawdown Data', ha='center', va='center', fontsize=12)
            ax17.set_title('Drawdown Duration Distribution', fontsize=12, fontweight='bold')

        # 18. Consecutive Wins Distribution
        ax18 = fig.add_subplot(5, 4, 18)
        unique_wins, counts_wins = np.unique(r['max_consecutive_wins'], return_counts=True)
        ax18.bar(unique_wins, counts_wins / len(r['max_consecutive_wins']) * 100,
                color='mediumseagreen', edgecolor='black', alpha=0.7)
        ax18.axvline(np.mean(r['max_consecutive_wins']), color='red', linestyle='--', linewidth=2,
                     label=f'Mean: {np.mean(r["max_consecutive_wins"]):.1f}')
        ax18.set_xlabel('Max Consecutive Wins', fontsize=10)
        ax18.set_ylabel('Frequency (%)', fontsize=10)
        ax18.set_title('Maximum Winning Streak Distribution', fontsize=12, fontweight='bold')
        ax18.legend(fontsize=8)
        ax18.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.01, 1, 0.99])

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Chart saved to: {save_path}")

        plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#                             SCENARIO ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def run_scenario_analysis(n_simulations: int = 5000):
    """Run simulations under different scenarios"""

    print("\n" + "=" * 70)
    print("                    SCENARIO ANALYSIS")
    print("=" * 70)

    scenarios = {
        'Pessimistic': SimulationConfig(
            monthly_trades_mean=28,
            prob_ultra_momentum=0.030,
            prob_standard_best=0.170,
            prob_exit_zone=0.800,
            standard_both_tp_prob=0.55,
            standard_both_sl_prob=0.05,
            n_simulations=n_simulations
        ),
        'Base Case': SimulationConfig(
            n_simulations=n_simulations
        ),
        'Optimistic': SimulationConfig(
            monthly_trades_mean=48,
            prob_ultra_momentum=0.055,
            prob_standard_best=0.265,
            prob_exit_zone=0.680,
            standard_both_tp_prob=0.70,
            standard_both_sl_prob=0.015,
            n_simulations=n_simulations
        )
    }

    scenario_results = {}

    for name, config in scenarios.items():
        print(f"\n📌 Running {name} scenario...")
        sim = MonteCarloSimulator(config)
        results = sim.run_simulation()
        scenario_results[name] = {
            'mean': np.mean(results['annual_pips']),
            'median': np.median(results['annual_pips']),
            'p5': np.percentile(results['annual_pips'], 5),
            'p95': np.percentile(results['annual_pips'], 95),
            'prob_profit': 100 * np.mean(results['annual_pips'] > 0),
            'max_dd': np.mean(results['max_drawdowns'])
        }

    print("\n" + "-" * 70)
    print(f"{'Scenario':<15} {'Mean':>10} {'Median':>10} {'5th':>10} {'95th':>10} {'P(Profit)':>10} {'Avg DD':>10}")
    print("-" * 70)

    for name, res in scenario_results.items():
        print(f"{name:<15} {res['mean']:>10,.0f} {res['median']:>10,.0f} {res['p5']:>10,.0f} {res['p95']:>10,.0f} {res['prob_profit']:>9.1f}% {res['max_dd']:>10,.0f}")

    print("-" * 70)

    return scenario_results


# ═══════════════════════════════════════════════════════════════════════════════
#                               MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Monte Carlo Trading Simulator for EURUSD',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python monte_carlo_simulator.py                    # Run with default 10,000 simulations
  python monte_carlo_simulator.py -n 25000           # Run with 25,000 simulations
  python monte_carlo_simulator.py -n 50000 --no-scenario  # Run 50k without scenario analysis
        """
    )

    parser.add_argument(
        '-n', '--simulations',
        type=int,
        default=10000,
        help='Number of simulations to run (default: 10000, max recommended: 100000)'
    )

    parser.add_argument(
        '--no-scenario',
        action='store_true',
        help='Skip scenario analysis (saves time for large simulations)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default='monte_carlo_results.png',
        help='Output filename for chart (default: monte_carlo_results.png)'
    )

    args = parser.parse_args()

    # Validate simulation count
    if args.simulations < 100:
        print("⚠️  Warning: Less than 100 simulations may produce unreliable results")
        print("   Minimum recommended: 1,000 simulations\n")

    if args.simulations > 100000:
        print("⚠️  Warning: More than 100,000 simulations may take significant time")
        response = input("   Continue? (y/n): ")
        if response.lower() != 'y':
            print("Simulation cancelled.")
            sys.exit(0)

    # Initialize simulator with configuration
    config = SimulationConfig(
        n_simulations=args.simulations,
        random_seed=args.seed
    )

    simulator = MonteCarloSimulator(config)

    # Run main simulation
    results = simulator.run_simulation()

    # Print comprehensive results
    simulator.print_results()

    # Generate visualizations
    simulator.plot_results(save_path=args.output)

    # Run scenario analysis (optional)
    if not args.no_scenario:
        # Use same simulation count as main analysis for consistency
        scenario_sims = args.simulations
        if scenario_sims > 25000:
            print(f"\n⚠️  Running scenario analysis with {scenario_sims:,} simulations per scenario")
            print(f"   This will run 3 scenarios totaling {scenario_sims * 3:,} simulations")
            print(f"   Consider using --no-scenario for faster execution")
        scenario_results = run_scenario_analysis(n_simulations=scenario_sims)
    else:
        print("\n⏭️  Skipping scenario analysis")

    print("\n" + "=" * 70)
    print("                    SIMULATION COMPLETE")
    print("=" * 70)
