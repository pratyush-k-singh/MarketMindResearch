import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Optional

class VisualizationSuite:
    """Visualization suite for experimental results"""
    
    def __init__(self, results_df: pd.DataFrame):
        self.results = results_df
        
        # Set style - using default matplotlib style
        plt.style.use('default')
        # Use a colorblind-friendly color cycle
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    
    def generate_plots(self, output_dir: str):
        """Generate all visualization plots"""
        os.makedirs(output_dir, exist_ok=True)
        
        self.plot_market_shares(os.path.join(output_dir, 'market_shares.png'))
        self.plot_hhi_evolution(os.path.join(output_dir, 'hhi_evolution.png'))
        self.plot_specialization(os.path.join(output_dir, 'specialization.png'))
        self.plot_price_evolution(os.path.join(output_dir, 'price_evolution.png'))
        self.plot_profit_evolution(os.path.join(output_dir, 'profit_evolution.png'))
        self.plot_location_evolution(os.path.join(output_dir, 'location_evolution.png'))
        self.plot_model_comparison(os.path.join(output_dir, 'model_comparison.png'))
    
    def plot_market_shares(self, save_path: str):
        """Plot evolution of market shares"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        for idx, market in enumerate(['A', 'B']):
            ax = axes[idx]
            market_data = self.results[self.results['market'] == market]
            
            for model in ['gpt-3.5-turbo', 'gpt-4']:
                agent_data = market_data[market_data['model'] == model]
                ax.plot(
                    agent_data['round'],
                    agent_data['market_share'],
                    label=f'{model}'
                )
            
            ax.set_title(f'Market {market} Shares')
            ax.set_xlabel('Round')
            ax.set_ylabel('Market Share')
            ax.grid(True)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def plot_hhi_evolution(self, save_path: str):
        """Plot evolution of market concentration (HHI)"""
        plt.figure(figsize=(12, 6))
        
        for market in ['A', 'B']:
            market_data = self.results[self.results['market'] == market]
            hhi_values = []
            
            for round_num in sorted(market_data['round'].unique()):
                round_shares = market_data[
                    market_data['round'] == round_num
                ]['market_share'].values
                hhi = np.sum(round_shares ** 2)
                hhi_values.append(hhi)
            
            plt.plot(
                range(len(hhi_values)),
                hhi_values,
                label=f'Market {market}'
            )
        
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, 
                   label='Competitive Market')
        plt.title('Market Concentration Evolution')
        plt.xlabel('Round')
        plt.ylabel('HHI')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(save_path)
        plt.close()
    
    def plot_specialization(self, save_path: str):
        """Plot specialization over time"""
        plt.figure(figsize=(12, 6))
        
        for model in ['gpt-3.5-turbo', 'gpt-4']:
            model_data = self.results[self.results['model'] == model]
            spec_values = []
            
            for round_num in sorted(model_data['round'].unique()):
                round_data = model_data[model_data['round'] == round_num]
                shares = [
                    round_data[round_data['market'] == m]['market_share'].iloc[0]
                    for m in ['A', 'B']
                ]
                spec_idx = abs(shares[0] - shares[1])
                spec_values.append(spec_idx)
            
            plt.plot(range(len(spec_values)), spec_values, label=model)
        
        plt.title('Model Specialization Evolution')
        plt.xlabel('Round')
        plt.ylabel('Specialization Index')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(save_path)
        plt.close()
    
    def plot_price_evolution(self, save_path: str):
        """Plot evolution of prices by model"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        for idx, market in enumerate(['A', 'B']):
            ax = axes[idx]
            market_data = self.results[self.results['market'] == market]
            
            for model in ['gpt-3.5-turbo', 'gpt-4']:
                model_data = market_data[market_data['model'] == model]
                ax.plot(
                    model_data['round'],
                    model_data['price'],
                    label=model
                )
            
            ax.set_title(f'Market {market} Prices')
            ax.set_xlabel('Round')
            ax.set_ylabel('Price')
            ax.grid(True)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
    def plot_location_evolution(self, save_path: str):
        """Plot evolution of locations in feature space by model"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        for idx, market in enumerate(['A', 'B']):
            ax = axes[idx]
            market_data = self.results[self.results['market'] == market]
            
            for model in ['gpt-3.5-turbo', 'gpt-4']:
                model_data = market_data[market_data['model'] == model]
                locations = np.array([
                    eval(loc) for loc in model_data['location']
                ])
                
                if locations.shape[1] == 2:
                    ax.plot(locations[:, 0], locations[:, 1], 
                           'o-', label=f'{model} path', alpha=0.5)
                    
                    ax.plot(locations[0, 0], locations[0, 1], 'go',
                           label=f'{model} start')
                    ax.plot(locations[-1, 0], locations[-1, 1], 'ro',
                           label=f'{model} end')
            
            ax.set_title(f'Market {market} Locations')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.grid(True)
            ax.legend()
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def plot_model_comparison(self, save_path: str):
        """Plot comparative performance metrics between models"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Average profits
        ax = axes[0, 0]
        for model in ['gpt-3.5-turbo', 'gpt-4']:
            model_data = self.results[self.results['model'] == model]
            ax.bar(model, model_data['profit'].mean())
        ax.set_title('Average Profit by Model')
        ax.set_ylabel('Profit')
        
        # Market share stability
        ax = axes[0, 1]
        for model in ['gpt-3.5-turbo', 'gpt-4']:
            model_data = self.results[self.results['model'] == model]
            ax.bar(model, model_data['market_share'].std())
        ax.set_title('Market Share Volatility by Model')
        ax.set_ylabel('Standard Deviation')
        
        # Price stability
        ax = axes[1, 0]
        for model in ['gpt-3.5-turbo', 'gpt-4']:
            model_data = self.results[self.results['model'] == model]
            ax.bar(model, model_data['price'].std())
        ax.set_title('Price Volatility by Model')
        ax.set_ylabel('Standard Deviation')
        
        # Location changes
        ax = axes[1, 1]
        for model in ['gpt-3.5-turbo', 'gpt-4']:
            model_data = self.results[self.results['model'] == model]
            locations = np.array([eval(loc) for loc in model_data['location']])
            movements = np.sqrt(np.sum(np.diff(locations, axis=0)**2, axis=1))
            ax.bar(model, np.mean(movements))
        ax.set_title('Average Location Movement by Model')
        ax.set_ylabel('Average Movement')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()