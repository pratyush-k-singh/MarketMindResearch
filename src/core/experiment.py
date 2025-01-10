import pandas as pd
import numpy as np
from typing import Dict, List
from .market import HotellingMarket

class ExperimentState:
    """Maintains the state of an ongoing experiment"""
    
    def __init__(self, markets: List[HotellingMarket]):
        self.markets = markets
        self.history = []
        self.round = 0
        
    def record_round(self, 
                    locations: Dict[str, List[np.ndarray]],
                    prices: Dict[str, List[float]],
                    shares: Dict[str, List[float]],
                    profits: Dict[str, List[float]]):
        """Record results of a round"""
        self.history.append({
            'round': self.round,
            'locations': locations,
            'prices': prices,
            'market_shares': shares,
            'profits': profits
        })
        self.round += 1
    
    def get_history_dataframe(self) -> pd.DataFrame:
        """Convert history to DataFrame format"""
        records = []
        
        for round_data in self.history:
            round_num = round_data['round']
            
            for market_idx in range(len(self.markets)):
                for agent_id in ['agent1', 'agent2']:
                    record = {
                        'round': round_num,
                        'market': market_idx,
                        'agent': agent_id,
                        'location': str(round_data['locations'][agent_id][market_idx]),
                        'price': round_data['prices'][agent_id][market_idx],
                        'market_share': round_data['market_shares'][agent_id][market_idx],
                        'profit': round_data['profits'][agent_id][market_idx]
                    }
                    records.append(record)
        
        return pd.DataFrame(records)