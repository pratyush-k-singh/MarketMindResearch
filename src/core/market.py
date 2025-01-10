import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class MarketConfig:
    """Configuration for a single market"""
    transport_cost: float  # Cost per unit distance
    market_size: int      # Total market size
    min_price: float     # Minimum allowable price
    max_price: float     # Maximum allowable price
    feature_dimensions: int  # Number of feature dimensions

class HotellingMarket:
    """Implementation of multi-dimensional Hotelling competition"""
    
    def __init__(self, config: MarketConfig):
        self.config = config
        self.transport_cost = config.transport_cost
        self.market_size = config.market_size
        
    def compute_demand(self, 
                      locations1: np.ndarray,
                      locations2: np.ndarray,
                      price1: float,
                      price2: float) -> Tuple[float, float]:
        """
        Compute demand for each firm given their locations and prices.
        Uses multi-dimensional Hotelling model with linear transport costs.
        """
        # Distance between firms
        dist = np.linalg.norm(locations1 - locations2)
        
        if dist == 0:  # If firms are at same location
            if price1 == price2:
                return self.market_size/2, self.market_size/2
            elif price1 < price2:
                return self.market_size, 0
            else:
                return 0, self.market_size
        
        # Point of indifference
        mid_point = (price2 - price1 + self.transport_cost * dist) / (2 * self.transport_cost * dist)
        
        # Compute market shares
        share1 = np.clip(mid_point, 0, 1)
        share2 = 1 - share1
        
        # Return actual quantities
        return share1 * self.market_size, share2 * self.market_size
    
    def compute_profits(self,
                       locations1: np.ndarray,
                       locations2: np.ndarray,
                       price1: float,
                       price2: float,
                       costs1: float,
                       costs2: float) -> Tuple[float, float]:
        """Compute profits for both firms"""
        demand1, demand2 = self.compute_demand(locations1, locations2, price1, price2)
        profit1 = (price1 - costs1) * demand1
        profit2 = (price2 - costs2) * demand2
        return profit1, profit2