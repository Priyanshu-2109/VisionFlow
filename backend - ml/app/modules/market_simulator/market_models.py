# app/modules/market_simulator/market_models.py
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import math
import random
from scipy import stats

class MarketModel:
    """Base class for different market models"""
    
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Predict market outcomes based on inputs"""
        raise NotImplementedError("Subclasses must implement predict method")

class PriceElasticityModel(MarketModel):
    """Model for price elasticity of demand"""
    
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.own_elasticity = params.get('own_elasticity', -1.5)
        self.cross_elasticities = params.get('cross_elasticities', {})
        self.market_growth = params.get('market_growth', 0.03)
        self.seasonality = params.get('seasonality', {})
        
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict demand changes based on price changes
        
        Args:
            inputs: Dictionary with price_change_pct, competitor_price_changes, period
            
        Returns:
            Dictionary with predicted demand change and contributing factors
        """
        price_change_pct = inputs.get('price_change_pct', 0)
        competitor_price_changes = inputs.get('competitor_price_changes', {})
        period = inputs.get('period')
        
        # Calculate own price effect
        own_price_effect = 1 + (price_change_pct/100) * self.own_elasticity
        
        # Calculate cross-price effects from competitors
        cross_price_effect = 1.0
        for competitor, change_pct in competitor_price_changes.items():
            if competitor in self.cross_elasticities:
                cross_elasticity = self.cross_elasticities[competitor]
                cross_price_effect *= (1 + (change_pct/100) * cross_elasticity)
        
        # Calculate seasonality effect
        seasonality_effect = 1.0
        if period and self.seasonality:
            month = period.split('-')[1] if '-' in period else None
            if month in self.seasonality:
                seasonality_effect = self.seasonality[month]
        
        # Calculate market growth effect
        market_growth_effect = 1 + self.market_growth/12  # Monthly growth rate
        
        # Calculate total demand change
        total_effect = own_price_effect * cross_price_effect * seasonality_effect * market_growth_effect
        demand_change_pct = (total_effect - 1) * 100
        
        return {
            'demand_change_pct': float(demand_change_pct),
            'factors': {
                'own_price_effect': float(own_price_effect),
                'cross_price_effect': float(cross_price_effect),
                'seasonality_effect': float(seasonality_effect),
                'market_growth_effect': float(market_growth_effect)
            }
        }

class MarketingResponseModel(MarketModel):
    """Model for marketing spend response"""
    
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.effectiveness = params.get('effectiveness', 0.3)
        self.saturation_point = params.get('saturation_point', 500000)
        self.decay_rate = params.get('decay_rate', 0.9)
        self.competitor_impact = params.get('competitor_impact', 0.5)
        
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict demand changes based on marketing spend changes
        
        Args:
            inputs: Dictionary with spend_change_pct, base_spend, period_index, competitor_spend_changes
            
        Returns:
            Dictionary with predicted demand change and contributing factors
        """
        spend_change_pct = inputs.get('spend_change_pct', 0)
        base_spend = inputs.get('base_spend', 10000)
        period_index = inputs.get('period_index', 0)  # Time period (for decay)
        competitor_spend_changes = inputs.get('competitor_spend_changes', {})
        
        # Calculate new spend
        new_spend = base_spend * (1 + spend_change_pct/100)
        
        # Calculate marketing effect with diminishing returns
        # For increases in spend
        if spend_change_pct > 0:
            marketing_effect = 1 + self.effectiveness * (self.decay_rate ** period_index) * \
                              np.log(1 + (new_spend - base_spend) / self.saturation_point)
        # For decreases in spend
        else:
            marketing_effect = 1 + self.effectiveness * (self.decay_rate ** period_index) * \
                              (spend_change_pct / 100)  # Linear decrease for spend reduction
        
        # Calculate competitor effect
        competitor_effect = 1.0
        for competitor, change_pct in competitor_spend_changes.items():
            # Competitors increasing spend reduces our effectiveness
            if change_pct > 0:
                competitor_effect *= (1 - self.competitor_impact * change_pct / 1000)
        
        # Calculate total demand change
        total_effect = marketing_effect * competitor_effect
        demand_change_pct = (total_effect - 1) * 100
        
        return {
            'demand_change_pct': float(demand_change_pct),
            'factors': {
                'marketing_effect': float(marketing_effect),
                'competitor_effect': float(competitor_effect),
                'decay_factor': float(self.decay_rate ** period_index)
            }
        }

class ProductAdoptionModel(MarketModel):
    """Bass diffusion model for new product adoption"""
    
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.innovation_factor = params.get('innovation_factor', 0.03)  # p
        self.imitation_factor = params.get('imitation_factor', 0.38)    # q
        self.market_potential = params.get('market_potential', 100000)  # m
        self.marketing_impact = params.get('marketing_impact', 0.2)
        self.price_sensitivity = params.get('price_sensitivity', -0.5)
        
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict new product adoption
        
        Args:
            inputs: Dictionary with period, cumulative_adopters, marketing_spend, price
            
        Returns:
            Dictionary with predicted new adopters and adoption rate
        """
        period = inputs.get('period', 1)  # Time period
        cumulative_adopters = inputs.get('cumulative_adopters', 0)
        marketing_spend = inputs.get('marketing_spend', 50000)
        price = inputs.get('price', 100)
        reference_price = inputs.get('reference_price', 100)
        seasonality = inputs.get('seasonality', 1.0)
        
        # Adjust innovation factor based on marketing spend
        adjusted_p = self.innovation_factor * (1 + self.marketing_impact * np.log(1 + marketing_spend / 50000))
        
        # Adjust for price
        price_effect = (price / reference_price) ** self.price_sensitivity
        
        # Calculate adoption for this period using Bass model
        m = self.market_potential
        p = adjusted_p
        q = self.imitation_factor
        
        if period == 1:
            new_adopters = p * m * price_effect * seasonality
        else:
            new_adopters = (p + q * cumulative_adopters / m) * (m - cumulative_adopters) * price_effect * seasonality
        
        # Add random variation
        noise_factor = np.random.normal(1, 0.1)  # 10% standard deviation
        new_adopters = max(0, new_adopters * noise_factor)
        
        # Calculate adoption rate
        adoption_rate = (cumulative_adopters + new_adopters) / m * 100
        
        return {
            'new_adopters': float(new_adopters),
            'adoption_rate': float(adoption_rate),
            'factors': {
                'innovation_effect': float(p * (m - cumulative_adopters)),
                'imitation_effect': float(q * cumulative_adopters * (m - cumulative_adopters) / m) if period > 1 else 0,
                'price_effect': float(price_effect),
                'seasonality': float(seasonality)
            }
        }

class CompetitorResponseModel(MarketModel):
    """Model for predicting competitor responses to market actions"""
    
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.competitors = params.get('competitors', [])
        self.response_probabilities = params.get('response_probabilities', {})
        self.response_delays = params.get('response_delays', {})
        self.response_strategies = params.get('response_strategies', {})
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict competitor responses to market actions
        
        Args:
            inputs: Dictionary with action_type, action_magnitude, period
            
        Returns:
            Dictionary with predicted competitor responses
        """
        action_type = inputs.get('action_type', 'price_change')  # price_change, marketing_increase, new_product
        action_magnitude = inputs.get('action_magnitude', 0)     # Percentage change or absolute value
        period = inputs.get('period', 1)                         # Time period since action
        
        responses = []
        
        for competitor in self.competitors:
            competitor_name = competitor.get('name', 'Unknown')
            
            # Get competitor-specific parameters or use defaults
            response_prob = self.response_probabilities.get(competitor_name, {}).get(action_type, 0.3)
            response_delay = self.response_delays.get(competitor_name, {}).get(action_type, 2)
            strategies = self.response_strategies.get(competitor_name, {}).get(action_type, 
                                                                           ['match', 'undercut', 'ignore'])
            
            # Adjust response probability based on action magnitude and time
            adjusted_prob = min(0.9, response_prob * (1 + abs(action_magnitude)/100) * min(3, period/response_delay))
            
            # Determine if competitor responds
            if random.random() < adjusted_prob:
                # Select response strategy
                if action_type == 'price_change':
                    strategy = np.random.choice(strategies, p=[0.4, 0.4, 0.2] if len(strategies) == 3 else None)
                    
                    if strategy == 'match':
                        response_magnitude = action_magnitude
                    elif strategy == 'undercut':
                        response_magnitude = action_magnitude * 1.2 if action_magnitude < 0 else action_magnitude * 0.8
                    else:  # ignore or other
                        response_magnitude = 0
                        
                elif action_type == 'marketing_increase':
                    strategy = np.random.choice(strategies, p=[0.3, 0.5, 0.2] if len(strategies) == 3 else None)
                    
                    if strategy == 'match':
                        response_magnitude = action_magnitude
                    elif strategy == 'outspend':
                        response_magnitude = action_magnitude * 1.2
                    else:  # ignore or other
                        response_magnitude = 0
                        
                elif action_type == 'new_product':
                    strategy = np.random.choice(strategies)
                    response_magnitude = 1 if strategy == 'launch_competing' else 0
                    
                responses.append({
                    'competitor': competitor_name,
                    'strategy': strategy,
                    'magnitude': float(response_magnitude),
                    'delay': int(response_delay),
                    'probability': float(adjusted_prob)
                })
        
        return {
            'action_type': action_type,
            'period': period,
            'responses': responses,
            'response_count': len(responses),
            'response_rate': len(responses) / len(self.competitors) if self.competitors else 0
        }
    