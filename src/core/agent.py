import os
import json
import numpy as np
import openai
import logging
from typing import Dict, List, Tuple
from .market import MarketConfig

class LLMAgent:
    """LLM-based decision making agent"""
    
    def __init__(self, 
                 agent_id: str,
                 model_name: str,
                 api_key: str,
                 temperature: float = 1.0):
        self.agent_id = agent_id
        self.model_name = model_name
        self.temperature = temperature
        self.data_dir = os.path.join('data', 'experiment_data')
        self.client = openai.OpenAI(api_key=api_key)
        self.logger = logging.getLogger(__name__)
        
    def _load_files(self, experiment_name: str) -> Tuple[str, str, str]:
        """Load history and strategy files"""
        exp_dir = os.path.join(self.data_dir, experiment_name)
        
        with open(os.path.join(exp_dir, f'{self.agent_id}_history.txt'), 'r') as f:
            history = f.read()
            
        with open(os.path.join(exp_dir, f'{self.agent_id}_PLANS.txt'), 'r') as f:
            plans = f.read()
            
        with open(os.path.join(exp_dir, f'{self.agent_id}_INSIGHTS.txt'), 'r') as f:
            insights = f.read()
            
        return history, plans, insights
    
    def _create_prompt(self, 
                      history: str,
                      plans: str,
                      insights: str,
                      market_configs: List[MarketConfig]) -> str:
        """Create the decision prompt for the LLM"""
        
        prompt = f"""You are an AI agent competing in multiple markets using the Hotelling spatial competition model.
                Each market has consumers distributed uniformly along a feature space.

                Your task is to choose your location in the feature space and set prices to maximize profits.
                Transportation costs (representing how far your offering is from consumer preferences) are factored into consumer decisions.

                Market configurations:
                """
        
        for i, config in enumerate(market_configs):
            prompt += f"""
                    Market {i}:
                    - Transport cost: {config.transport_cost}
                    - Market size: {config.market_size}
                    - Price range: ${config.min_price} to ${config.max_price}
                    - Feature dimensions: {config.feature_dimensions}
                    """

        prompt += f"""
                Current market conditions:
                {history}

                Your current plans:
                {plans}

                Your current insights:
                {insights}

                Please provide your decisions for each market in the following JSON format:
                {{
                    "market_0": {{
                        "location": [x1, x2, ...],
                        "price": float
                    }},
                    "market_1": {{
                        "location": [x1, x2, ...],
                        "price": float
                    }}
                }}

                Ensure that:
                - Locations are within [0,1] for each dimension
                - Prices are within specified min/max for each market
                - Your decisions account for competitor behavior and market dynamics
                - You consider long-term strategic positioning
                """
        return prompt

    def make_decision(self, 
                     experiment_name: str,
                     market_configs: List[MarketConfig]) -> Tuple[List[np.ndarray], List[float]]:
        """Make location and pricing decisions for all markets"""
        try:
            # Load historical data and create prompt
            history, plans, insights = self._load_files(experiment_name)
            prompt = self._create_prompt(history, plans, insights, market_configs)
            
            # Get LLM response using new OpenAI client
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature
            )
            
            # Parse response
            decisions = json.loads(response.choices[0].message.content)
            
            # Extract decisions
            locations = []
            prices = []
            for market_id in range(len(market_configs)):
                market_key = f"market_{market_id}"
                loc = np.array(decisions[market_key]["location"])
                price = float(decisions[market_key]["price"])
                
                # Validate decisions
                if not all(0 <= x <= 1 for x in loc):
                    raise ValueError(f"Invalid location coordinates in {loc}")
                
                config = market_configs[market_id]
                if not config.min_price <= price <= config.max_price:
                    raise ValueError(f"Price {price} outside allowed range")
                
                locations.append(loc)
                prices.append(price)
            
            return locations, prices
            
        except Exception as e:
            self.logger.error(f"Error in LLM decision: {e}")
            # Return random valid decisions as fallback
            return self._generate_fallback_decisions(market_configs)
    
    def _generate_fallback_decisions(self, 
                                   market_configs: List[MarketConfig]) -> Tuple[List[np.ndarray], List[float]]:
        """Generate random valid decisions as fallback"""
        locations = []
        prices = []
        
        for config in market_configs:
            # Random location in feature space
            loc = np.random.rand(config.feature_dimensions)
            # Random price within allowed range
            price = np.random.uniform(config.min_price, config.max_price)
            
            locations.append(loc)
            prices.append(price)
        
        return locations, prices