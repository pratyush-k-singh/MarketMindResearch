import numpy as np
import pandas as pd
from typing import Dict
from scipy import stats

class StrategyAnalyzer:
    """Analyzes strategic behavior of agents"""
    
    def __init__(self, results_df: pd.DataFrame):
        self.results = results_df
    
    def analyze_price_strategies(self) -> Dict:
        """Analyze pricing strategies"""
        strategies = {}
        
        for model in ['gpt-3.5-turbo', 'gpt-4']:
            agent_data = self.results[self.results['model'] == model]
            
            # Analyze pricing patterns for each market
            market_strategies = {}
            for market in ['A', 'B']:
                market_data = agent_data[agent_data['market'] == market]
                prices = market_data['price'].values
                
                # Compute price statistics
                price_stats = {
                    'mean': np.mean(prices),
                    'std': np.std(prices),
                    'min': np.min(prices),
                    'max': np.max(prices),
                    'trend': stats.linregress(
                        range(len(prices)),
                        prices
                    ).slope
                }
                
                # Identify pricing regime changes
                price_changes = np.diff(prices)
                significant_changes = np.where(
                    np.abs(price_changes) > np.std(prices)
                )[0]
                
                price_stats['num_strategy_changes'] = len(significant_changes)
                
                market_strategies[market] = price_stats
            
            strategies[model] = market_strategies
        
        return strategies
    
    def analyze_location_strategies(self) -> Dict:
        """Analyze location strategies"""
        strategies = {}
        
        for model in ['gpt-3.5-turbo', 'gpt-4']:
            agent_data = self.results[self.results['model'] == model]
            
            market_strategies = {}
            for market in ['A', 'B']:
                market_data = agent_data[agent_data['market'] == market]
                locations = np.array([
                    eval(loc) for loc in market_data['location']
                ])
                
                # Compute movement patterns
                movements = np.sqrt(np.sum(np.diff(locations, axis=0)**2, axis=1))
                
                location_stats = {
                    'total_movement': np.sum(movements),
                    'avg_movement': np.mean(movements),
                    'std_movement': np.std(movements),
                    'final_position': locations[-1].tolist(),
                    'position_changes': len(np.where(movements > 0.1)[0])
                }
                
                market_strategies[market] = location_stats
            
            strategies[model] = market_strategies
        
        return strategies
    
    def analyze_competitive_responses(self) -> Dict:
        """Analyze how agents respond to each other"""
        responses = {}
        
        for market in ['A', 'B']:
            market_data = self.results[self.results['market'] == market]
            
            # Get data for each model
            gpt35_data = market_data[market_data['model'] == 'gpt-3.5-turbo']
            gpt4_data = market_data[market_data['model'] == 'gpt-4']
            
            # Analyze price responses
            gpt35_prices = gpt35_data['price'].values[:-1]  # Previous rounds
            gpt4_prices = gpt4_data['price'].values[1:]     # Next rounds
            
            price_corr = np.corrcoef(gpt35_prices, gpt4_prices)[0, 1]
            
            # Analyze location responses
            gpt35_locs = np.array([
                eval(loc) for loc in gpt35_data['location']
            ])[:-1]
            gpt4_locs = np.array([
                eval(loc) for loc in gpt4_data['location']
            ])[1:]
            
            # Compute distances between locations
            distances = np.sqrt(np.sum((gpt35_locs - gpt4_locs)**2, axis=1))
            
            responses[market] = {
                'price_correlation': float(price_corr),
                'avg_location_distance': float(np.mean(distances)),
                'min_location_distance': float(np.min(distances)),
                'max_location_distance': float(np.max(distances))
            }
        
        return responses
    
    def generate_strategy_report(self) -> str:
        """Generate comprehensive strategy analysis report"""
        price_strategies = self.analyze_price_strategies()
        location_strategies = self.analyze_location_strategies()
        competitive_responses = self.analyze_competitive_responses()
        
        report = ["Strategic Behavior Analysis", "=========================\n"]
        
        # Price strategy analysis
        report.append("Price Strategies")
        report.append("---------------")
        for model, markets in price_strategies.items():
            report.append(f"\n{model}:")
            for market, stats in markets.items():
                report.append(f"\nMarket {market}:")
                report.append(f"- Average price: ${stats['mean']:.2f}")
                report.append(f"- Price volatility: ${stats['std']:.2f}")
                report.append(f"- Price range: ${stats['min']:.2f} - ${stats['max']:.2f}")
                report.append(f"- Price trend: {stats['trend']:.2e}")
                report.append(f"- Strategy changes: {stats['num_strategy_changes']}")
        
        # Location strategy analysis
        report.append("\nLocation Strategies")
        report.append("------------------")
        for model, markets in location_strategies.items():
            report.append(f"\n{model}:")
            for market, stats in markets.items():
                report.append(f"\nMarket {market}:")
                report.append(f"- Total movement: {stats['total_movement']:.3f}")
                report.append(f"- Average movement: {stats['avg_movement']:.3f}")
                report.append(f"- Position changes: {stats['position_changes']}")
                report.append(f"- Final position: {stats['final_position']}")
        
        # Competitive response analysis
        report.append("\nCompetitive Responses")
        report.append("--------------------")
        for market, stats in competitive_responses.items():
            report.append(f"\nMarket {market}:")
            report.append(f"- Price correlation: {stats['price_correlation']:.3f}")
            report.append(f"- Average location distance: {stats['avg_location_distance']:.3f}")
            report.append(f"- Location distance range: {stats['min_location_distance']:.3f} - {stats['max_location_distance']:.3f}")
        
        return "\n".join(report)