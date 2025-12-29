import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from dataclasses import dataclass
from typing import List, Dict, Tuple
import warnings
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
            'all_trades': trades
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

        print("\n" + "=" * 70)

    def plot_results(self, save_path: str = None):
        """Generate comprehensive visualization"""

        if self.results is None:
            raise ValueError("Run simulation first!")

        r = self.results

        fig = plt.figure(figsize=(20, 24))

        # Title
        fig.suptitle('Monte Carlo Simulation: EURUSD 2026 Forecast\n(Based on 147-trade historical sample)',
                     fontsize=16, fontweight='bold', y=0.98)

        # 1. Annual Pips Distribution
        ax1 = fig.add_subplot(4, 3, 1)
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
        ax2 = fig.add_subplot(4, 3, 2)
        ax2.hist(100*r['win_rates'], bins=40, density=True, alpha=0.7, color='forestgreen', edgecolor='black')
        ax2.axvline(100*np.mean(r['win_rates']), color='red', linestyle='--', linewidth=2, label=f'Mean: {100*np.mean(r["win_rates"]):.1f}%')
        ax2.set_xlabel('Win Rate (%)', fontsize=10)
        ax2.set_ylabel('Density', fontsize=10)
        ax2.set_title('Win Rate Distribution', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # 3. Max Drawdown Distribution
        ax3 = fig.add_subplot(4, 3, 3)
        ax3.hist(r['max_drawdowns'], bins=50, density=True, alpha=0.7, color='crimson', edgecolor='black')
        ax3.axvline(np.mean(r['max_drawdowns']), color='black', linestyle='--', linewidth=2, label=f'Mean: {np.mean(r["max_drawdowns"]):,.0f}')
        ax3.axvline(np.percentile(r['max_drawdowns'], 95), color='orange', linestyle='--', linewidth=2, label=f'95th: {np.percentile(r["max_drawdowns"], 95):,.0f}')
        ax3.set_xlabel('Max Drawdown (pips)', fontsize=10)
        ax3.set_ylabel('Density', fontsize=10)
        ax3.set_title('Maximum Drawdown Distribution', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        # 4. Probability Curve (Cumulative)
        ax4 = fig.add_subplot(4, 3, 4)
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
        ax5 = fig.add_subplot(4, 3, 5)
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
        ax6 = fig.add_subplot(4, 3, 6)
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
        ax7 = fig.add_subplot(4, 3, 7)
        unique_streaks, counts = np.unique(r['max_consecutive_losses'], return_counts=True)
        ax7.bar(unique_streaks, counts / len(r['max_consecutive_losses']) * 100,
                color='coral', edgecolor='black', alpha=0.7)
        ax7.set_xlabel('Max Consecutive Losses', fontsize=10)
        ax7.set_ylabel('Frequency (%)', fontsize=10)
        ax7.set_title('Maximum Losing Streak Distribution', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3)

        # 8. Negative Months Distribution
        ax8 = fig.add_subplot(4, 3, 8)
        unique_neg, counts_neg = np.unique(r['negative_months'], return_counts=True)
        ax8.bar(unique_neg, counts_neg / len(r['negative_months']) * 100,
                color='salmon', edgecolor='black', alpha=0.7)
        ax8.set_xlabel('Number of Negative Months', fontsize=10)
        ax8.set_ylabel('Frequency (%)', fontsize=10)
        ax8.set_title('Negative Months per Year Distribution', fontsize=12, fontweight='bold')
        ax8.grid(True, alpha=0.3)

        # 9. Pattern Contribution Pie Chart
        ax9 = fig.add_subplot(4, 3, 9)
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
        ax10 = fig.add_subplot(4, 3, 10)
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
        ax11 = fig.add_subplot(4, 3, 11)
        ax11.hist(r['total_trades'], bins=30, density=True, alpha=0.7, color='mediumpurple', edgecolor='black')
        ax11.axvline(np.mean(r['total_trades']), color='red', linestyle='--', linewidth=2,
                     label=f'Mean: {np.mean(r["total_trades"]):,.0f}')
        ax11.set_xlabel('Total Annual Trades', fontsize=10)
        ax11.set_ylabel('Density', fontsize=10)
        ax11.set_title('Annual Trade Count Distribution', fontsize=12, fontweight='bold')
        ax11.legend(fontsize=8)
        ax11.grid(True, alpha=0.3)

        # 12. Risk-Return Scatter
        ax12 = fig.add_subplot(4, 3, 12)
        ax12.scatter(r['max_drawdowns'], r['annual_pips'], alpha=0.1, s=10, color='steelblue')
        ax12.axhline(0, color='red', linestyle='--', linewidth=1)
        ax12.set_xlabel('Max Drawdown (pips)', fontsize=10)
        ax12.set_ylabel('Annual Pips', fontsize=10)
        ax12.set_title('Risk-Return Relationship', fontsize=12, fontweight='bold')
        ax12.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.02, 1, 0.96])

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Chart saved to: {save_path}")

        plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#                             SCENARIO ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def run_scenario_analysis():
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
            n_simulations=5000
        ),
        'Base Case': SimulationConfig(
            n_simulations=5000
        ),
        'Optimistic': SimulationConfig(
            monthly_trades_mean=48,
            prob_ultra_momentum=0.055,
            prob_standard_best=0.265,
            prob_exit_zone=0.680,
            standard_both_tp_prob=0.70,
            standard_both_sl_prob=0.015,
            n_simulations=5000
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

    # Initialize simulator with default configuration
    config = SimulationConfig(
        n_simulations=10000,
        random_seed=42
    )

    simulator = MonteCarloSimulator(config)

    # Run main simulation
    results = simulator.run_simulation()

    # Print comprehensive results
    simulator.print_results()

    # Generate visualizations
    simulator.plot_results(save_path='monte_carlo_results.png')

    # Run scenario analysis
    scenario_results = run_scenario_analysis()

    print("\n" + "=" * 70)
    print("                    SIMULATION COMPLETE")
    print("=" * 70)
