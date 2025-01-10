# src/runners/main.py

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from ..core.market import HotellingMarket, MarketConfig
from ..core.agent import LLMAgent
from ..core.experiment import ExperimentState
from ..data.generator import TestDataGenerator
from ..data.formatter import DataFormatter
from ..data.validator import DataValidator
from ..analysis.market_division import MarketDivisionAnalyzer
from ..analysis.visualization import VisualizationSuite
from ..analysis.strategy import StrategyAnalyzer

def load_api_key() -> str:
    """Load API key from environment or .env file"""
    # Try to load from .env file in project root
    root_dir = Path(__file__).parent.parent.parent
    env_path = root_dir / '.env'
    
    if env_path.exists():
        load_dotenv(env_path)
    
    # Try to get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Please either:\n"
            "1. Set it as an environment variable, or\n"
            "2. Create a .env file in the project root directory with:\n"
            "   OPENAI_API_KEY=your-api-key-here"
        )
    
    return api_key

class ExperimentRunner:
    """Main experiment runner comparing different LLM models"""
    
    def __init__(self, 
                 experiment_name: str,
                 api_key: str):
        self.experiment_name = experiment_name
        self.api_key = api_key
        self.setup_logging()
        
        # Create necessary directories
        os.makedirs('data/test_data', exist_ok=True)
        os.makedirs('data/experiment_data', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    os.path.join(log_dir, f'{self.experiment_name}.log')
                ),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def prepare_data(self):
        """Prepare and validate test data"""
        self.logger.info("Generating test data...")
        generator = TestDataGenerator()
        datasets = generator.generate_all_datasets()
        
        self.logger.info("Formatting data for experiments...")
        formatter = DataFormatter()
        formatter.format_all_datasets()
        
        self.logger.info("Validating data...")
        validator = DataValidator()
        validation_results = validator.validate_all()
        
        # Check validation results
        all_valid = all(result['success'] for result in validation_results.values())
        if not all_valid:
            validator.print_validation_report(validation_results)
            raise ValueError("Data validation failed")
        
        self.logger.info("Data preparation complete")
        return datasets
    
    def run_experiment(self, market_configs: List[MarketConfig], num_rounds: int = 50):
        """Run experiment comparing GPT-3.5 vs GPT-4"""
        self.logger.info(f"Starting experiment: {self.experiment_name}")
        
        # Initialize markets
        markets = [HotellingMarket(config) for config in market_configs]
        
        # Initialize agents with different models
        agents = {
            'agent1': LLMAgent(
                'agent1', 
                model_name="gpt-3.5-turbo",
                api_key=self.api_key
            ),
            'agent2': LLMAgent(
                'agent2', 
                model_name="gpt-4-0125-preview",  # Using latest GPT-4
                api_key=self.api_key
            )
        }
        
        # Initialize experiment state
        state = ExperimentState(markets)
        
        # Run experiment rounds
        for round_num in range(num_rounds):
            self.logger.info(f"Running round {round_num}")
            
            # Get agent decisions
            decisions = {}
            for agent_id, agent in agents.items():
                try:
                    locations, prices = agent.make_decision(
                        self.experiment_name,
                        market_configs
                    )
                    decisions[agent_id] = (locations, prices)
                    self.logger.info(
                        f"{agent_id} ({agent.model_name}) decisions - "
                        f"locations: {locations}, prices: {prices}"
                    )
                except Exception as e:
                    self.logger.error(
                        f"Error in agent {agent_id} ({agent.model_name}) "
                        f"decision: {e}"
                    )
                    raise
            
            # Compute outcomes
            market_results = self._compute_round_results(markets, decisions)
            
            # Log round results
            self.logger.info(
                f"Round {round_num} results - "
                f"shares: {market_results['shares']}, "
                f"profits: {market_results['profits']}"
            )
            
            # Record results
            state.record_round(
                locations={
                    agent_id: dec[0] for agent_id, dec in decisions.items()
                },
                prices={
                    agent_id: dec[1] for agent_id, dec in decisions.items()
                },
                shares=market_results['shares'],
                profits=market_results['profits']
            )
        
        self.logger.info("Experiment complete")
        return state
    
    def _compute_round_results(self, 
                             markets: List[HotellingMarket],
                             decisions: Dict) -> Dict:
        """Compute results for a single round"""
        results = {
            'shares': {agent_id: [] for agent_id in decisions},
            'profits': {agent_id: [] for agent_id in decisions}
        }
        
        for market_idx, market in enumerate(markets):
            # Get decisions for this market
            loc1 = decisions['agent1'][0][market_idx]
            loc2 = decisions['agent2'][0][market_idx]
            price1 = decisions['agent1'][1][market_idx]
            price2 = decisions['agent2'][1][market_idx]
            
            # Compute demands
            demand1, demand2 = market.compute_demand(
                loc1, loc2, price1, price2
            )
            
            # Compute market shares
            total_demand = demand1 + demand2
            if total_demand > 0:
                share1 = demand1 / total_demand
                share2 = demand2 / total_demand
            else:
                share1 = share2 = 0.5
            
            results['shares']['agent1'].append(share1)
            results['shares']['agent2'].append(share2)
            
            # Compute profits
            profit1 = demand1 * price1
            profit2 = demand2 * price2
            
            results['profits']['agent1'].append(profit1)
            results['profits']['agent2'].append(profit2)
        
        return results
    
    def analyze_results(self, state: ExperimentState):
        """Analyze experiment results with model comparison"""
        self.logger.info("Analyzing results...")
        
        # Convert results to DataFrame
        results_df = state.get_history_dataframe()
        results_df['model'] = results_df['agent'].map({
            'agent1': 'gpt-3.5-turbo',
            'agent2': 'gpt-4-0125-preview'
        })
        
        # Save raw results
        results_dir = os.path.join('results', self.experiment_name)
        os.makedirs(results_dir, exist_ok=True)
        results_df.to_csv(os.path.join(results_dir, 'results.csv'), index=False)
        
        # Create analysis components
        market_analyzer = MarketDivisionAnalyzer(results_df)
        strategy_analyzer = StrategyAnalyzer(results_df)
        visualizer = VisualizationSuite(results_df)
        
        # Generate analyses
        division_metrics = market_analyzer.compute_division_metrics()
        division_tests = market_analyzer.test_market_division_hypothesis()
        strategy_analysis = strategy_analyzer.analyze_price_strategies()
        
        # Generate visualizations
        visualizer.generate_plots(results_dir)
        
        # Save analysis results
        with open(os.path.join(results_dir, 'analysis.json'), 'w') as f:
            json.dump({
                'division_metrics': division_metrics,
                'division_tests': division_tests,
                'strategy_analysis': strategy_analysis
            }, f, indent=2)
        
        # Generate reports
        with open(os.path.join(results_dir, 'market_division_report.md'), 'w') as f:
            f.write(market_analyzer.generate_division_report())
            
        with open(os.path.join(results_dir, 'strategy_report.md'), 'w') as f:
            f.write(strategy_analyzer.generate_strategy_report())
        
        self.logger.info("Analysis complete")
        
        return {
            'division_metrics': division_metrics,
            'division_tests': division_tests,
            'strategy_analysis': strategy_analysis
        }
    
    def run(self):
        """Run complete experiment pipeline"""
        try:
            # Prepare data
            datasets = self.prepare_data()
            
            # Get market configurations based on experiment type
            market_configs = self.get_market_configs()
            
            # Run experiment
            state = self.run_experiment(market_configs)
            
            # Analyze results
            analysis = self.analyze_results(state)
            
            self.logger.info(f"Experiment {self.experiment_name} pipeline complete")
            return state, analysis
            
        except Exception as e:
            self.logger.error(f"Error in experiment pipeline: {e}", exc_info=True)
            raise
    
    def get_market_configs(self) -> List[MarketConfig]:
        """Get market configurations based on experiment type"""
        configs = {
            'symmetric': [
                MarketConfig(
                    transport_cost=1.0,
                    market_size=1000,
                    min_price=0,
                    max_price=100,
                    feature_dimensions=2
                ),
                MarketConfig(
                    transport_cost=1.0,
                    market_size=1000,
                    min_price=0,
                    max_price=100,
                    feature_dimensions=2
                )
            ],
            'asymmetric_transport': [
                MarketConfig(
                    transport_cost=0.5,
                    market_size=1000,
                    min_price=0,
                    max_price=100,
                    feature_dimensions=2
                ),
                MarketConfig(
                    transport_cost=2.0,
                    market_size=1000,
                    min_price=0,
                    max_price=100,
                    feature_dimensions=2
                )
            ],
            'asymmetric_size': [
                MarketConfig(
                    transport_cost=1.0,
                    market_size=1500,
                    min_price=0,
                    max_price=100,
                    feature_dimensions=2
                ),
                MarketConfig(
                    transport_cost=1.0,
                    market_size=500,
                    min_price=0,
                    max_price=100,
                    feature_dimensions=2
                )
            ],
            'high_dim': [
                MarketConfig(
                    transport_cost=1.0,
                    market_size=1000,
                    min_price=0,
                    max_price=100,
                    feature_dimensions=4
                ),
                MarketConfig(
                    transport_cost=1.0,
                    market_size=1000,
                    min_price=0,
                    max_price=100,
                    feature_dimensions=4
                )
            ],
            'constrained_price': [
                MarketConfig(
                    transport_cost=1.0,
                    market_size=1000,
                    min_price=40,
                    max_price=60,
                    feature_dimensions=2
                ),
                MarketConfig(
                    transport_cost=1.0,
                    market_size=1000,
                    min_price=40,
                    max_price=60,
                    feature_dimensions=2
                )
            ]
        }
        
        if self.experiment_name not in configs:
            raise ValueError(f"Unknown experiment type: {self.experiment_name}")
        
        return configs[self.experiment_name]

def main():
    """Main entry point"""
    try:
        # Get API key
        api_key = load_api_key()
        
        # Define experiments to run
        experiments = [
            'symmetric',
            'asymmetric_transport',
            'asymmetric_size',
            'high_dim',
            'constrained_price'
        ]
        
        # Run each experiment
        for experiment_name in experiments:
            try:
                logging.info(f"\nStarting experiment: {experiment_name}")
                logging.info("=" * 50)
                
                runner = ExperimentRunner(experiment_name, api_key)
                runner.run()
                
                logging.info(f"Completed experiment: {experiment_name}")
                logging.info("-" * 50)
                
            except Exception as e:
                logging.error(
                    f"Error in experiment {experiment_name}: {str(e)}", 
                    exc_info=True
                )
                continue
        
    except Exception as e:
        logging.error(f"Fatal error in main: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
    