import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os
import logging

class DataValidator:
    """Validates test data and formatted experiment data"""
    
    def __init__(self, 
                 test_data_dir: str = 'data/test_data',
                 experiment_data_dir: str = 'data/experiment_data'):
        self.test_data_dir = test_data_dir
        self.experiment_data_dir = experiment_data_dir
        self.logger = logging.getLogger(__name__)
    
    def validate_all(self) -> Dict[str, Dict]:
        """Validate all datasets"""
        results = {}
        
        # Validate raw test data
        for filename in os.listdir(self.test_data_dir):
            if not filename.endswith('_data.csv'):
                continue
                
            experiment_name = filename.replace('_data.csv', '')
            df = pd.read_csv(os.path.join(self.test_data_dir, filename))
            
            success, issues = self.validate_test_data(df)
            results[f"raw_{experiment_name}"] = {
                'success': success,
                'issues': issues
            }
        
        # Validate formatted data
        for experiment_name in os.listdir(self.experiment_data_dir):
            exp_dir = os.path.join(self.experiment_data_dir, experiment_name)
            if not os.path.isdir(exp_dir):
                continue
                
            success, issues = self.validate_formatted_data(exp_dir)
            results[f"formatted_{experiment_name}"] = {
                'success': success,
                'issues': issues
            }
        
        return results
    
    def validate_test_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate raw test data"""
        issues = []
        
        # Check required columns
        required_cols = ['date', 'round', 'market', 'agent', 'location',
                        'price', 'market_share', 'quantity_sold', 'profit']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        # Check for null values
        if df.isnull().any().any():
            null_counts = df.isnull().sum()
            null_cols = null_counts[null_counts > 0]
            issues.append(f"Found null values in columns: {null_cols.index.tolist()}")
        
        # Check round continuity
        rounds = sorted(df['round'].unique())
        if len(rounds) < 30:
            issues.append(f"Insufficient rounds: {len(rounds)} (need 30)")
        
        expected_rounds = list(range(min(rounds), max(rounds) + 1))
        missing_rounds = set(expected_rounds) - set(rounds)
        if missing_rounds:
            issues.append(f"Missing rounds: {missing_rounds}")
        
        # Check market shares
        for round_num in rounds:
            for market in ['A', 'B']:
                shares = df[
                    (df['round'] == round_num) & 
                    (df['market'] == market)
                ]['market_share']
                share_sum = shares.sum()
                if not np.isclose(share_sum, 1.0, atol=1e-3):
                    issues.append(
                        f"Market shares don't sum to 1 in round {round_num}, "
                        f"market {market}: {share_sum:.3f}"
                    )
        
        # Validate location format and values
        try:
            locations = df['location'].apply(eval)
            invalid_locs = []
            for idx, loc in enumerate(locations):
                if not isinstance(loc, (list, tuple)):
                    invalid_locs.append(idx)
                elif not all(isinstance(x, (int, float)) for x in loc):
                    invalid_locs.append(idx)
                elif not all(0 <= x <= 1 for x in loc):
                    invalid_locs.append(idx)
            
            if invalid_locs:
                issues.append(f"Invalid locations at indices: {invalid_locs}")
        except Exception as e:
            issues.append(f"Error parsing locations: {str(e)}")
        
        # Check price ranges
        if (df['price'] <= 0).any():
            issues.append("Found non-positive prices")
        
        # Validate profit calculations
        price_margin = 0.2  # Allow for some variation in profit calculations
        for idx, row in df.iterrows():
            expected_revenue = row['price'] * row['quantity_sold']
            if not row['profit'] <= expected_revenue * (1 + price_margin):
                issues.append(
                    f"Profit ({row['profit']}) exceeds possible revenue "
                    f"({expected_revenue}) at index {idx}"
                )
        
        return len(issues) == 0, issues
    
    def validate_formatted_data(self, exp_dir: str) -> Tuple[bool, Dict[str, List[str]]]:
        """Validate formatted experiment data"""
        issues = {}
        
        for agent_id in ['agent1', 'agent2']:
            agent_issues = []
            
            # Check history file
            history_path = os.path.join(exp_dir, f'{agent_id}_history.txt')
            if not os.path.exists(history_path):
                agent_issues.append("Missing history file")
            else:
                with open(history_path, 'r') as f:
                    history = f.read()
                    
                    # Validate history content
                    if not history.strip():
                        agent_issues.append("Empty history file")
                    if "Round" not in history:
                        agent_issues.append("Missing round information")
                    if "Market A" not in history or "Market B" not in history:
                        agent_issues.append("Missing market information")
                    if "location" not in history.lower():
                        agent_issues.append("Missing location information")
                    if "profit" not in history.lower():
                        agent_issues.append("Missing profit information")
            
            # Check plans file
            plans_path = os.path.join(exp_dir, f'{agent_id}_PLANS.txt')
            if not os.path.exists(plans_path):
                agent_issues.append("Missing plans file")
            else:
                with open(plans_path, 'r') as f:
                    plans = f.read()
                    if not plans.strip():
                        agent_issues.append("Empty plans file")
                    elif len(plans.split('\n')) < 3:
                        agent_issues.append("Insufficient plan content")
            
            # Check insights file
            insights_path = os.path.join(exp_dir, f'{agent_id}_INSIGHTS.txt')
            if not os.path.exists(insights_path):
                agent_issues.append("Missing insights file")
            else:
                with open(insights_path, 'r') as f:
                    insights = f.read()
                    if not insights.strip():
                        agent_issues.append("Empty insights file")
                    elif len(insights.split('\n')) < 3:
                        agent_issues.append("Insufficient insights content")
            
            if agent_issues:
                issues[agent_id] = agent_issues
        
        return len(issues) == 0, issues
    
    def print_validation_report(self, results: Dict):
        """Print formatted validation report"""
        self.logger.info("\nValidation Report")
        self.logger.info("================\n")
        
        all_valid = True
        
        for dataset, result in results.items():
            success = result['success']
            all_valid &= success
            
            self.logger.info(f"{dataset}:")
            self.logger.info(f"Status: {'✓ PASS' if success else '✗ FAIL'}")
            
            if not success:
                self.logger.info("Issues found:")
                issues = result['issues']
                if isinstance(issues, dict):
                    for agent, agent_issues in issues.items():
                        self.logger.info(f"\n{agent}:")
                        for issue in agent_issues:
                            self.logger.info(f"  - {issue}")
                else:
                    for issue in issues:
                        self.logger.info(f"  - {issue}")
            self.logger.info("")
        
        self.logger.info(f"\nOverall validation: {'✓ PASS' if all_valid else '✗ FAIL'}")
        return all_valid