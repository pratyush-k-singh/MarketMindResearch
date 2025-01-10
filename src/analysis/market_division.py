# src/analysis/market_division.py

import pandas as pd
import numpy as np
from typing import Dict
from scipy import stats

class MarketDivisionAnalyzer:
    """Analyzes market division behaviors in experimental results"""
    
    def __init__(self, results_df: pd.DataFrame):
        self.results = results_df
        
    def compute_division_metrics(self) -> Dict:
        """Compute metrics related to market division behavior"""
        metrics = {}
        
        # Analyze each market separately
        for market in ['A', 'B']:
            market_data = self.results[self.results['market'] == market]
            
            # Calculate market concentration over time
            hhi_over_time = []
            for round_num in sorted(market_data['round'].unique()):
                round_data = market_data[market_data['round'] == round_num]
                shares = round_data['market_share'].values
                hhi = np.sum(shares ** 2)
                hhi_over_time.append(hhi)
            
            # Calculate stability of market shares
            share_changes = []
            for agent in ['agent1', 'agent2']:
                agent_data = market_data[market_data['agent'] == agent]
                share_changes.extend(np.abs(np.diff(agent_data['market_share'])))
            
            metrics[f'market_{market}'] = {
                'avg_hhi': np.mean(hhi_over_time),
                'final_hhi': hhi_over_time[-1],
                'hhi_trend': stats.linregress(
                    range(len(hhi_over_time)), 
                    hhi_over_time
                ).slope,
                'share_volatility': np.std(share_changes),
                'max_share_change': np.max(share_changes)
            }
        
        # Analyze cross-market specialization
        specialization = self.compute_specialization_metrics()
        metrics['specialization'] = specialization
        
        return metrics
    
    def compute_specialization_metrics(self) -> Dict:
        """Compute metrics related to firm specialization"""
        metrics = {}
        
        for model in ['gpt-3.5-turbo', 'gpt-4']:
            model_data = self.results[self.results['model'] == model]
            
            # Calculate specialization index over time
            spec_index = []
            for round_num in sorted(model_data['round'].unique()):
                round_data = model_data[model_data['round'] == round_num]
                shares = [
                    round_data[round_data['market'] == m]['market_share'].iloc[0]
                    for m in ['A', 'B']
                ]
                # Specialization index: difference in market shares
                spec_idx = abs(shares[0] - shares[1])
                spec_index.append(spec_idx)
            
            metrics[model] = {
                'avg_specialization': np.mean(spec_index),
                'final_specialization': spec_index[-1],
                'spec_trend': stats.linregress(
                    range(len(spec_index)), 
                    spec_index
                ).slope,
                'volatility': np.std(spec_index)
            }
        
        return metrics
    
    def test_market_division_hypothesis(self) -> Dict:
        """Perform statistical tests for market division behavior"""
        tests = {}
        
        # Test if HHI is significantly higher than random expectation (0.5)
        for market in ['A', 'B']:
            market_data = self.results[self.results['market'] == market]
            hhi_values = []
            
            for round_num in sorted(market_data['round'].unique()):
                round_data = market_data[market_data['round'] == round_num]
                shares = round_data['market_share'].values
                hhi = np.sum(shares ** 2)
                hhi_values.append(hhi)
            
            t_stat, p_value = stats.ttest_1samp(hhi_values, 0.5)
            tests[f'market_{market}_hhi'] = {
                'statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        # Test for specialization
        for model in ['gpt-3.5-turbo', 'gpt-4']:
            model_data = self.results[self.results['model'] == model]
            spec_values = []
            
            for round_num in sorted(model_data['round'].unique()):
                round_data = model_data[model_data['round'] == round_num]
                shares = [
                    round_data[round_data['market'] == m]['market_share'].iloc[0]
                    for m in ['A', 'B']
                ]
                spec_values.append(abs(shares[0] - shares[1]))
            
            t_stat, p_value = stats.ttest_1samp(spec_values, 0)
            tests[f'{model}_specialization'] = {
                'statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        return tests
    
    def generate_division_report(self) -> str:
        """Generate comprehensive report on market division behavior"""
        metrics = self.compute_division_metrics()
        tests = self.test_market_division_hypothesis()
        
        report = ["Market Division Analysis Report", "============================\n"]
        
        # Market concentration analysis
        report.append("Market Concentration")
        report.append("-----------------")
        for market in ['A', 'B']:
            m = metrics[f'market_{market}']
            report.append(f"\nMarket {market}:")
            report.append(f"- Average HHI: {m['avg_hhi']:.3f}")
            report.append(f"- Final HHI: {m['final_hhi']:.3f}")
            report.append(f"- HHI Trend: {m['hhi_trend']:.3e}")
            report.append(f"- Share Volatility: {m['share_volatility']:.3f}")
            
            # Add statistical significance
            test = tests[f'market_{market}_hhi']
            if test['significant']:
                report.append("- Concentration is statistically significant")
                report.append(f"  (p-value: {test['p_value']:.3e})")
        
        # Model specialization analysis
        report.append("\nModel Specialization")
        report.append("-----------------")
        for model in ['gpt-3.5-turbo', 'gpt-4']:
            m = metrics['specialization'][model]
            report.append(f"\n{model}:")
            report.append(f"- Average Specialization: {m['avg_specialization']:.3f}")
            report.append(f"- Final Specialization: {m['final_specialization']:.3f}")
            report.append(f"- Specialization Trend: {m['spec_trend']:.3e}")
            report.append(f"- Volatility: {m['volatility']:.3f}")
            
            # Add statistical significance
            test = tests[f'{model}_specialization']
            if test['significant']:
                report.append("- Specialization is statistically significant")
                report.append(f"  (p-value: {test['p_value']:.3e})")
        
        return "\n".join(report)
    