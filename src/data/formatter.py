import pandas as pd
import numpy as np
from typing import Dict, List
import os
import json

class DataFormatter:
    """Formats test data into LLM-readable format"""
    
    def __init__(self, 
                 test_data_dir: str = 'data/test_data',
                 output_dir: str = 'data/experiment_data'):
        self.test_data_dir = test_data_dir
        self.output_dir = output_dir
        
    def format_all_datasets(self):
        """Format all test datasets for experiments"""
        for filename in os.listdir(self.test_data_dir):
            if not filename.endswith('_data.csv'):
                continue
            
            experiment_name = filename.replace('_data.csv', '')
            input_path = os.path.join(self.test_data_dir, filename)
            df = pd.read_csv(input_path)
            self.format_dataset(experiment_name, df)
    
    def format_dataset(self, experiment_name: str, df: pd.DataFrame):
        """Format a single dataset"""
        exp_dir = os.path.join(self.output_dir, experiment_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        for agent_id in ['agent1', 'agent2']:
            # Generate history file
            history = self._format_market_history(df, agent_id)
            with open(os.path.join(exp_dir, f'{agent_id}_history.txt'), 'w') as f:
                f.write(history)
            
            # Generate initial plans and insights
            plans_insights = self._generate_initial_strategy(df, agent_id)
            
            with open(os.path.join(exp_dir, f'{agent_id}_PLANS.txt'), 'w') as f:
                f.write(plans_insights['plans'])
                
            with open(os.path.join(exp_dir, f'{agent_id}_INSIGHTS.txt'), 'w') as f:
                f.write(plans_insights['insights'])
    
    def _format_market_history(self, df: pd.DataFrame, agent_id: str) -> str:
        """Format market history for an agent"""
        formatted = ""
        
        for round_num in sorted(df['round'].unique()):
            round_data = df[df['round'] == round_num]
            
            formatted += f"\nRound {round_num}:\n"
            
            for market in ['A', 'B']:
                market_data = round_data[round_data['market'] == market]
                agent_data = market_data[market_data['agent'] == agent_id].iloc[0]
                competitor_data = market_data[market_data['agent'] != agent_id].iloc[0]
                
                formatted += f"* Market {market}:\n"
                formatted += f"- My location: {agent_data['location']}\n"
                formatted += f"- Competitor location: {competitor_data['location']}\n"
                formatted += f"- My price: ${agent_data['price']:.2f}\n"
                formatted += f"- Competitor price: ${competitor_data['price']:.2f}\n"
                formatted += f"- My market share: {agent_data['market_share']:.1%}\n"
                formatted += f"- Units sold: {agent_data['quantity_sold']:.0f}\n"
                formatted += f"- My profit: ${agent_data['profit']:.2f}\n"
        
        return formatted
    
    def _generate_initial_strategy(self, df: pd.DataFrame, agent_id: str) -> Dict[str, str]:
        """Generate initial plans and insights"""
        recent_data = df[df['round'] >= df['round'].max() - 5]  # Last 5 rounds
        
        plans = []
        insights = []
        
        # Analyze each market
        for market in ['A', 'B']:
            market_data = recent_data[recent_data['market'] == market]
            agent_data = market_data[market_data['agent'] == agent_id]
            
            avg_profit = agent_data['profit'].mean()
            avg_share = agent_data['market_share'].mean()
            price_trend = agent_data['price'].diff().mean()
            
            insights.append(f"Market {market}:")
            insights.append(f"- Average profit: ${avg_profit:.0f}")
            insights.append(f"- Market share: {avg_share:.1%}")
            insights.append(f"- Price trend: {'Increasing' if price_trend > 0 else 'Decreasing' if price_trend < 0 else 'Stable'}")
            insights.append("")
            
            if avg_share > 0.5:
                plans.append(f"1. Maintain strong position in Market {market}")
            else:
                plans.append(f"1. Improve competitive position in Market {market}")
        
        plans.append("2. Monitor competitor pricing and location strategies")
        plans.append("3. Adjust prices based on market share performance")
        
        return {
            'plans': "\n".join(plans),
            'insights': "\n".join(insights)
        }
