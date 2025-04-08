import os
import random
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketSimulator:
    """
    Advanced real-time market simulator for business decision impact analysis
    with integrated machine learning capabilities
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the market simulator with optional pre-trained models
        
        Args:
            model_path: Path to pre-trained models for prediction enhancement
        """
        self.market_data = None
        self.company_data = None
        self.competitors = []
        self.elasticity_matrix = {}
        self.market_growth_rate = 0.03  # Default annual market growth rate
        self.seasonality_factors = {}
        self.simulation_results = []
        self.current_period = None
        self.industry = None
        
        # ML model components
        self.price_impact_model = None
        self.marketing_impact_model = None
        self.adoption_model = None
        self.competitor_response_model = None
        
        # Load pre-trained models if provided
        if model_path and os.path.exists(model_path):
            self._load_models(model_path)
        else:
            # Initialize with default models
            self._initialize_default_models()
            
    def _load_models(self, model_path: str) -> None:
        """Load pre-trained models from the specified path"""
        try:
            models = joblib.load(model_path)
            self.price_impact_model = models.get('price_impact_model')
            self.marketing_impact_model = models.get('marketing_impact_model')
            self.adoption_model = models.get('adoption_model')
            self.competitor_response_model = models.get('competitor_response_model')
            logger.info(f"Loaded pre-trained models from {model_path}")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            self._initialize_default_models()
            
    def _initialize_default_models(self) -> None:
        """Initialize default machine learning models"""
        # Price impact model: Gradient Boosting for better accuracy with price elasticity
        self.price_impact_model = Pipeline([
            ('scaler', StandardScaler()),
            ('model', GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            ))
        ])
        
        # Marketing impact model: Random Forest for capturing non-linear marketing effects
        self.marketing_impact_model = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(
                n_estimators=100,
                max_depth=4,
                random_state=42
            ))
        ])
        
        # Product adoption model: Gradient Boosting for predicting adoption rates
        self.adoption_model = Pipeline([
            ('scaler', StandardScaler()),
            ('model', GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                random_state=42
            ))
        ])
        
        # Competitor response model: Random Forest for predicting competitor behavior
        self.competitor_response_model = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(
                n_estimators=100,
                max_depth=3,
                random_state=42
            ))
        ])
        
        logger.info("Initialized default machine learning models")
        
    def save_models(self, save_path: str) -> None:
        """Save the trained models to the specified path"""
        models = {
            'price_impact_model': self.price_impact_model,
            'marketing_impact_model': self.marketing_impact_model,
            'adoption_model': self.adoption_model,
            'competitor_response_model': self.competitor_response_model
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        joblib.dump(models, save_path)
        logger.info(f"Saved models to {save_path}")
        
    def train_models(self, historical_data: Dict[str, pd.DataFrame]) -> None:
        """
        Train the machine learning models using historical data
        
        Args:
            historical_data: Dictionary containing historical data for training
                - 'price_data': DataFrame with price changes and outcomes
                - 'marketing_data': DataFrame with marketing spend changes and outcomes
                - 'product_data': DataFrame with product launches and adoption
                - 'competitor_data': DataFrame with competitor responses
        """
        # Train price impact model
        if 'price_data' in historical_data and not historical_data['price_data'].empty:
            price_df = historical_data['price_data']
            X_price = price_df.drop(['revenue_change', 'volume_change', 'market_share_change'], axis=1)
            y_price_revenue = price_df['revenue_change']
            
            self.price_impact_model.fit(X_price, y_price_revenue)
            logger.info("Trained price impact model")
            
        # Train marketing impact model
        if 'marketing_data' in historical_data and not historical_data['marketing_data'].empty:
            marketing_df = historical_data['marketing_data']
            X_marketing = marketing_df.drop(['revenue_change', 'roi'], axis=1)
            y_marketing_roi = marketing_df['roi']
            
            self.marketing_impact_model.fit(X_marketing, y_marketing_roi)
            logger.info("Trained marketing impact model")
            
        # Train product adoption model
        if 'product_data' in historical_data and not historical_data['product_data'].empty:
            product_df = historical_data['product_data']
            X_product = product_df.drop(['adoption_rate', 'time_to_breakeven'], axis=1)
            y_product_adoption = product_df['adoption_rate']
            
            self.adoption_model.fit(X_product, y_product_adoption)
            logger.info("Trained product adoption model")
            
        # Train competitor response model
        if 'competitor_data' in historical_data and not historical_data['competitor_data'].empty:
            competitor_df = historical_data['competitor_data']
            X_competitor = competitor_df.drop(['response_type', 'response_magnitude'], axis=1)
            y_competitor_response = competitor_df['response_magnitude']
            
            self.competitor_response_model.fit(X_competitor, y_competitor_response)
            logger.info("Trained competitor response model")
        
    def load_company_data(self, company_data: Dict[str, Any]) -> None:
        """Load company data for simulation"""
        self.company_data = company_data
        self.industry = company_data.get('industry')
        
        # Set default elasticity if not provided
        if 'price_elasticity' not in self.company_data:
            self.company_data['price_elasticity'] = -1.5  # Default price elasticity
            
        logger.info(f"Loaded company data for {company_data.get('name')} in {self.industry} industry")
        
    def load_market_data(self, market_data: Dict[str, Any]) -> None:
        """Load market data for simulation"""
        self.market_data = market_data
        self.competitors = market_data.get('competitors', [])
        self.market_growth_rate = market_data.get('growth_rate', 0.03)
        self.current_period = market_data.get('current_period', datetime.now().strftime('%Y-%m'))
        
        # Load seasonality factors if provided
        if 'seasonality_factors' in market_data:
            self.seasonality_factors = market_data['seasonality_factors']
        else:
            # Default seasonality (slight increase in Q4, decrease in Q1)
            self.seasonality_factors = {
                '01': 0.9, '02': 0.92, '03': 0.95,
                '04': 1.0, '05': 1.0, '06': 1.02,
                '07': 1.0, '08': 0.98, '09': 1.05,
                '10': 1.1, '11': 1.15, '12': 1.2
            }
            
        # Load elasticity matrix if provided
        if 'elasticity_matrix' in market_data:
            self.elasticity_matrix = market_data['elasticity_matrix']
        else:
            # Create default elasticity matrix
            self._create_default_elasticity_matrix()
            
        logger.info(f"Loaded market data with {len(self.competitors)} competitors")
    
    def _create_default_elasticity_matrix(self) -> None:
        """Create default elasticity matrix based on competitors"""
        # Include our company in the list
        all_companies = [self.company_data['name']] + [comp['name'] for comp in self.competitors]
        
        # Create matrix: how company in row is affected by price change of company in column
        self.elasticity_matrix = {}
        
        for company1 in all_companies:
            self.elasticity_matrix[company1] = {}
            
            for company2 in all_companies:
                if company1 == company2:
                    # Own price elasticity
                    if company1 == self.company_data['name']:
                        self.elasticity_matrix[company1][company2] = self.company_data.get('price_elasticity', -1.5)
                    else:
                        # Competitor's own price elasticity
                        comp_data = next((c for c in self.competitors if c['name'] == company1), None)
                        elasticity = comp_data.get('price_elasticity', -1.3) if comp_data else -1.3
                        self.elasticity_matrix[company1][company2] = elasticity
                else:
                    # Cross-price elasticity (positive: substitutes, negative: complements)
                    # Default assumption: products are substitutes
                    self.elasticity_matrix[company1][company2] = random.uniform(0.2, 0.7)
    
    def simulate_price_change(self, price_change_pct: float, 
                             periods: int = 12) -> Dict[str, Any]:
        """
        Simulate the impact of a price change on market share and revenue
        
        Args:
            price_change_pct: Percentage change in price (e.g., 10 for 10% increase)
            periods: Number of periods to simulate
            
        Returns:
            Dict containing simulation results
        """
        if not self.company_data or not self.market_data:
            raise ValueError("Company and market data must be loaded before simulation")
            
        # Initialize simulation
        start_date = datetime.strptime(self.current_period, '%Y-%m')
        simulation_periods = [
            (start_date + timedelta(days=30*i)).strftime('%Y-%m')
            for i in range(periods)
        ]
        
        # Get initial values
        initial_price = self.company_data.get('price', 100)
        initial_market_share = self.company_data.get('market_share', 10)  # Percentage
        initial_volume = self.company_data.get('volume', 1000)
        initial_revenue = initial_price * initial_volume
        
        # Calculate new price
        new_price = initial_price * (1 + price_change_pct/100)
        
        # Initialize results
        results = []
        
        current_market_share = initial_market_share
        current_volume = initial_volume
        current_revenue = initial_revenue
        
        # Competitor response probability increases over time
        competitor_response_prob = 0.0
        
        # Use ML model for enhanced predictions if available
        use_ml_price_model = self.price_impact_model is not None
        use_ml_competitor_model = self.competitor_response_model is not None
        
        for i, period in enumerate(simulation_periods):
            period_month = period.split('-')[1]
            seasonality = self.seasonality_factors.get(period_month, 1.0)
            
            # Market growth factor (monthly)
            market_growth = (1 + self.market_growth_rate/12)
            
            # Calculate price effect on demand
            if use_ml_price_model:
                # Prepare features for ML model
                features = {
                    'price_change_pct': price_change_pct,
                    'initial_price': initial_price,
                    'initial_market_share': initial_market_share,
                    'period_num': i,
                    'market_growth': self.market_growth_rate,
                    'seasonality': seasonality,
                    'competitor_count': len(self.competitors),
                    'industry': self._encode_industry(self.industry)
                }
                
                # Convert to DataFrame for prediction
                X_pred = pd.DataFrame([features])
                
                # Predict revenue change
                predicted_revenue_change = self.price_impact_model.predict(X_pred)[0]
                price_effect = 1 + (predicted_revenue_change / 100)
            else:
                # Traditional elasticity-based calculation
                own_elasticity = self.elasticity_matrix[self.company_data['name']][self.company_data['name']]
                price_effect = (1 + (price_change_pct/100) * own_elasticity)
            
            # Competitor responses
            competitor_effects = 1.0
            competitor_responses = []
            
            # Probability of competitor response increases over time
            competitor_response_prob = min(0.8, competitor_response_prob + 0.1)
            
            for competitor in self.competitors:
                # Check if competitor responds in this period
                will_respond = random.random() < competitor_response_prob
                
                if will_respond:
                    if use_ml_competitor_model:
                        # Prepare features for competitor response prediction
                        comp_features = {
                            'our_price_change': price_change_pct,
                            'our_market_share': current_market_share,
                            'competitor_market_share': competitor.get('market_share', 5),
                            'period_num': i,
                            'competitor_aggression': competitor.get('aggression', 0.5),
                            'price_difference': (new_price / competitor.get('price', initial_price) - 1) * 100
                        }
                        
                        # Convert to DataFrame for prediction
                        X_comp = pd.DataFrame([comp_features])
                        
                        # Predict competitor response magnitude
                        response_magnitude = self.competitor_response_model.predict(X_comp)[0]
                        
                        # Determine response type based on magnitude
                        if response_magnitude > 0.7 * price_change_pct:
                            response_type = 'match'
                            comp_price_change = price_change_pct
                        elif response_magnitude > 0:
                            response_type = 'partial_match'
                            comp_price_change = response_magnitude
                        elif response_magnitude < -0.2 * price_change_pct:
                            response_type = 'undercut'
                            comp_price_change = price_change_pct * 1.2
                        else:
                            response_type = 'ignore'
                            comp_price_change = 0
                    else:
                        # Traditional probability-based competitor response
                        response_type = np.random.choice(
                            ['match', 'undercut', 'ignore', 'partial_match'],
                            p=[0.3, 0.3, 0.2, 0.2]
                        )
                        
                        if response_type == 'match':
                            comp_price_change = price_change_pct
                        elif response_type == 'undercut':
                            comp_price_change = price_change_pct * 1.2
                        elif response_type == 'partial_match':
                            comp_price_change = price_change_pct * 0.5
                        else:  # ignore
                            comp_price_change = 0
                    
                    # Record competitor response
                    competitor_responses.append({
                        'competitor': competitor['name'],
                        'response_type': response_type,
                        'price_change_pct': float(comp_price_change)
                    })
                    
                    # Calculate cross-price effect
                    cross_elasticity = self.elasticity_matrix[self.company_data['name']][competitor['name']]
                    competitor_effects *= (1 + (comp_price_change/100) * cross_elasticity)
            
            # Calculate new volume considering all factors
            new_volume = current_volume * price_effect * competitor_effects * seasonality * market_growth
            
            # Calculate new revenue
            new_revenue = new_price * new_volume
            
            # Calculate new market share (simplified model)
            market_share_change = (price_effect * competitor_effects - 1) * 100
            new_market_share = max(0.1, current_market_share + market_share_change * 0.1)
            
            # Record results for this period
            period_result = {
                'period': period,
                'price': float(new_price),
                'volume': float(new_volume),
                'revenue': float(new_revenue),
                'market_share': float(new_market_share),
                'revenue_change_pct': float((new_revenue - initial_revenue) / initial_revenue * 100),
                'volume_change_pct': float((new_volume - initial_volume) / initial_volume * 100),
                'market_share_change_pct': float((new_market_share - initial_market_share)),
                'seasonality_factor': float(seasonality),
                'competitor_responses': competitor_responses
            }
            
            results.append(period_result)
            
            # Update current values for next period
            current_volume = new_volume
            current_revenue = new_revenue
            current_market_share = new_market_share
        
        # Calculate summary statistics
        avg_revenue = sum(r['revenue'] for r in results) / len(results)
        total_revenue = sum(r['revenue'] for r in results)
        peak_revenue = max(r['revenue'] for r in results)
        min_revenue = min(r['revenue'] for r in results)
        
        final_market_share = results[-1]['market_share']
        market_share_change = final_market_share - initial_market_share
        
        summary = {
            'initial_price': float(initial_price),
            'new_price': float(new_price),
            'price_change_pct': float(price_change_pct),
            'simulation_periods': periods,
            'avg_monthly_revenue': float(avg_revenue),
            'total_revenue': float(total_revenue),
            'peak_revenue': float(peak_revenue),
            'min_revenue': float(min_revenue),
            'initial_market_share': float(initial_market_share),
            'final_market_share': float(final_market_share),
            'market_share_change': float(market_share_change),
            'recommendation': self._generate_price_recommendation(results, price_change_pct)
        }
        
        simulation_result = {
            'simulation_type': 'price_change',
            'timestamp': datetime.now().isoformat(),
            'input_parameters': {
                'price_change_pct': price_change_pct,
                'periods': periods
            },
            'period_results': results,
            'summary': summary
        }
        
        self.simulation_results.append(simulation_result)
        return simulation_result
    
    def _encode_industry(self, industry: str) -> int:
        """Simple encoding of industry for ML models"""
        industry_map = {
            'technology': 1,
            'retail': 2,
            'manufacturing': 3,
            'healthcare': 4,
            'finance': 5,
            'services': 6
        }
        return industry_map.get(industry.lower() if industry else '', 0)
    
    def _generate_price_recommendation(self, results: List[Dict[str, Any]], 
                                      price_change_pct: float) -> Dict[str, Any]:
        """Generate recommendation based on simulation results"""
        # Calculate key metrics
        avg_revenue_change = sum(r['revenue_change_pct'] for r in results) / len(results)
        final_market_share_change = results[-1]['market_share_change_pct']
        
        # Determine if price change is favorable
        if avg_revenue_change > 5 and final_market_share_change > -2:
            recommendation = "Implement the price change"
            confidence = "High"
            rationale = "The price change is projected to increase revenue without significant market share loss."
        elif avg_revenue_change > 0:
            recommendation = "Consider implementing the price change"
            confidence = "Medium"
            rationale = "The price change is projected to increase revenue, but monitor market share closely."
        elif price_change_pct < 0 and final_market_share_change > 3:
            recommendation = "Consider implementing the price reduction"
            confidence = "Medium"
            rationale = "The price reduction may lead to market share gains that could offset revenue loss in the long term."
        else:
            recommendation = "Avoid the price change"
            confidence = "High"
            rationale = "The price change is projected to decrease revenue and/or significantly reduce market share."
        
        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'rationale': rationale
        }
    
    def simulate_marketing_spend_change(self, spend_change_pct: float, 
                                       periods: int = 12) -> Dict[str, Any]:
        """
        Simulate the impact of a marketing spend change on market share and revenue
        
        Args:
            spend_change_pct: Percentage change in marketing spend (e.g., 20 for 20% increase)
            periods: Number of periods to simulate
            
        Returns:
            Dict containing simulation results
        """
        if not self.company_data or not self.market_data:
            raise ValueError("Company and market data must be loaded before simulation")
            
        # Initialize simulation
        start_date = datetime.strptime(self.current_period, '%Y-%m')
        simulation_periods = [
            (start_date + timedelta(days=30*i)).strftime('%Y-%m')
            for i in range(periods)
        ]
        
        # Get initial values
        initial_spend = self.company_data.get('marketing_spend', 10000)
        initial_market_share = self.company_data.get('market_share', 10)  # Percentage
        initial_volume = self.company_data.get('volume', 1000)
        initial_price = self.company_data.get('price', 100)
        initial_revenue = initial_price * initial_volume
        
        # Calculate new marketing spend
        new_spend = initial_spend * (1 + spend_change_pct/100)
        
        # Marketing effectiveness parameters
        # Diminishing returns modeled with logarithmic function
        marketing_effectiveness = self.company_data.get('marketing_effectiveness', 0.3)
        saturation_point = self.company_data.get('marketing_saturation', 500000)
        
        # Initialize results
        results = []
        
        current_market_share = initial_market_share
        current_volume = initial_volume
        current_revenue = initial_revenue
        
        # Decay factor for marketing impact (diminishes over time)
        decay_factor = 1.0
        
        # Use ML model for enhanced predictions if available
        use_ml_marketing_model = self.marketing_impact_model is not None
        
        for i, period in enumerate(simulation_periods):
            period_month = period.split('-')[1]
            seasonality = self.seasonality_factors.get(period_month, 1.0)
            
            # Market growth factor (monthly)
            market_growth = (1 + self.market_growth_rate/12)
            
            # Calculate marketing effect on demand
            if use_ml_marketing_model:
                # Prepare features for ML model
                features = {
                    'spend_change_pct': spend_change_pct,
                    'initial_spend': initial_spend,
                    'new_spend': new_spend,
                    'initial_market_share': initial_market_share,
                    'current_market_share': current_market_share,
                    'period_num': i,
                    'decay_factor': decay_factor,
                    'seasonality': seasonality,
                    'industry': self._encode_industry(self.industry)
                }
                
                # Convert to DataFrame for prediction
                X_pred = pd.DataFrame([features])
                
                # Predict marketing ROI
                predicted_roi = self.marketing_impact_model.predict(X_pred)[0]
                
                # Calculate marketing effect based on predicted ROI
                monthly_spend_increase = (new_spend - initial_spend) / 12
                expected_revenue_increase = predicted_roi * monthly_spend_increase
                marketing_effect = 1 + (expected_revenue_increase / current_revenue)
            else:
                # Traditional calculation with diminishing returns
                if spend_change_pct > 0:
                    marketing_effect = 1 + marketing_effectiveness * decay_factor * \
                                    np.log(1 + (new_spend - initial_spend) / saturation_point)
                else:
                    marketing_effect = 1 + marketing_effectiveness * decay_factor * \
                                    (spend_change_pct / 100)  # Linear decrease for spend reduction
            
            # Decay the marketing impact over time
            decay_factor = max(0.2, decay_factor * 0.95)
            
            # Competitor responses (simplified)
            competitor_effect = 1.0
            competitor_responses = []
            
            # Only model competitor responses after a few periods
            if i >= 2:
                for competitor in self.competitors:
                    # Competitors may increase their marketing in response
                    if random.random() < 0.3:
                        comp_response_pct = random.uniform(10, spend_change_pct * 0.8 if spend_change_pct > 0 else 15)
                        
                        competitor_responses.append({
                            'competitor': competitor['name'],
                            'marketing_increase_pct': float(comp_response_pct)
                        })
                        
                        # Reduce our marketing effectiveness due to competitor response
                        competitor_effect *= (1 - 0.05 * comp_response_pct / 100)
            
            # Calculate new volume considering all factors
            new_volume = current_volume * marketing_effect * competitor_effect * seasonality * market_growth
            
            # Calculate new revenue
            new_revenue = initial_price * new_volume
            
            # Calculate new market share
            market_share_change = (marketing_effect * competitor_effect - 1) * 2  # Amplify effect
            new_market_share = max(0.1, current_market_share + market_share_change)
            
            # Calculate ROI for this period
            period_marketing_roi = (new_revenue - current_revenue) / (new_spend / 12)
            
            # Record results for this period
            period_result = {
                'period': period,
                'marketing_spend': float(new_spend / 12),  # Monthly spend
                'volume': float(new_volume),
                'revenue': float(new_revenue),
                'market_share': float(new_market_share),
                'revenue_change_pct': float((new_revenue - initial_revenue) / initial_revenue * 100),
                'volume_change_pct': float((new_volume - initial_volume) / initial_volume * 100),
                'market_share_change_pct': float((new_market_share - initial_market_share)),
                'marketing_roi': float(period_marketing_roi),
                'seasonality_factor': float(seasonality),
                'marketing_effect': float(marketing_effect),
                'competitor_responses': competitor_responses
            }
            
            results.append(period_result)
            
            # Update current values for next period
            current_volume = new_volume
            current_revenue = new_revenue
            current_market_share = new_market_share
        
        # Calculate summary statistics
        avg_revenue = sum(r['revenue'] for r in results) / len(results)
        total_revenue = sum(r['revenue'] for r in results)
        total_marketing_spend = new_spend
        total_profit = total_revenue - total_marketing_spend
        roi = (total_revenue - (initial_revenue * periods)) / total_marketing_spend
        
        final_market_share = results[-1]['market_share']
        market_share_change = final_market_share - initial_market_share
        
        summary = {
            'initial_marketing_spend': float(initial_spend),
            'new_marketing_spend': float(new_spend),
            'spend_change_pct': float(spend_change_pct),
            'simulation_periods': periods,
            'avg_monthly_revenue': float(avg_revenue),
            'total_revenue': float(total_revenue),
            'total_marketing_spend': float(total_marketing_spend),
            'total_profit': float(total_profit),
            'marketing_roi': float(roi),
            'initial_market_share': float(initial_market_share),
            'final_market_share': float(final_market_share),
            'market_share_change': float(market_share_change),
            'recommendation': self._generate_marketing_recommendation(results, spend_change_pct, roi)
        }
        
        simulation_result = {
            'simulation_type': 'marketing_spend_change',
            'timestamp': datetime.now().isoformat(),
            'input_parameters': {
                'spend_change_pct': spend_change_pct,
                'periods': periods
            },
            'period_results': results,
            'summary': summary
        }
        
        self.simulation_results.append(simulation_result)
        return simulation_result
    
    def _generate_marketing_recommendation(self, results: List[Dict[str, Any]], 
                                         spend_change_pct: float, roi: float) -> Dict[str, Any]:
        """Generate recommendation based on marketing simulation results"""
        # Calculate key metrics
        avg_roi = sum(r.get('marketing_roi', 0) for r in results) / len(results)
        final_market_share_change = results[-1]['market_share_change_pct']
        
        # Determine if marketing spend change is favorable
        if roi > 2:
            recommendation = "Implement the marketing spend increase"
            confidence = "High"
            rationale = f"The marketing spend change is projected to deliver an excellent ROI of {roi:.2f} with market share gains."
        elif roi > 1:
            recommendation = "Consider implementing the marketing spend change"
            confidence = "Medium"
            rationale = f"The marketing spend change is projected to be profitable with an ROI of {roi:.2f}."
        elif spend_change_pct < 0 and roi > 0:
            recommendation = "Consider implementing the marketing spend reduction"
            confidence = "Medium"
            rationale = "Reducing marketing spend may improve overall profitability while having limited impact on revenue."
        else:
            recommendation = "Avoid the marketing spend change"
            confidence = "High"
            rationale = f"The marketing spend change is projected to have a poor ROI of {roi:.2f}."
        
        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'rationale': rationale
        }
    
    def simulate_product_launch(self, product_data: Dict[str, Any], 
                               periods: int = 24) -> Dict[str, Any]:
        """
        Simulate the impact of launching a new product
        
        Args:
            product_data: Dictionary containing new product data
            periods: Number of periods to simulate
            
        Returns:
            Dict containing simulation results
        """
        if not self.company_data or not self.market_data:
            raise ValueError("Company and market data must be loaded before simulation")
            
        # Initialize simulation
        start_date = datetime.strptime(self.current_period, '%Y-%m')
        simulation_periods = [
            (start_date + timedelta(days=30*i)).strftime('%Y-%m')
            for i in range(periods)
        ]
        
        # Get product data
        product_name = product_data.get('name', 'New Product')
        product_price = product_data.get('price', 100)
        product_cost = product_data.get('cost', 60)
        product_marketing = product_data.get('marketing_spend', 50000)
        target_market_size = product_data.get('target_market_size', 100000)
        cannibalization_rate = product_data.get('cannibalization_rate', 0.2)  # 20% cannibalization of existing products
        
        # Use ML model for enhanced adoption prediction if available
        use_ml_adoption_model = self.adoption_model is not None
        
        # Adoption curve parameters (Bass diffusion model)
        innovation_factor = product_data.get('innovation_factor', 0.03)  # p
        imitation_factor = product_data.get('imitation_factor', 0.38)   # q
        
        # Initialize results
        results = []
        
        # Tracking variables
        cumulative_adopters = 0
        previous_adopters = 0
        
        for i, period in enumerate(simulation_periods):
            period_month = period.split('-')[1]
            seasonality = self.seasonality_factors.get(period_month, 1.0)
            
            if use_ml_adoption_model and i > 0:  # Use ML model after first period to have some data
                # Prepare features for ML model
                features = {
                    'product_price': product_price,
                    'price_to_cost_ratio': product_price / product_cost,
                    'marketing_spend': product_marketing / periods,
                    'target_market_size': target_market_size,
                    'current_penetration': (cumulative_adopters / target_market_size),
                    'previous_adopters': previous_adopters,
                    'period_num': i,
                    'seasonality': seasonality,
                    'industry': self._encode_industry(self.industry),
                    'is_innovative': product_data.get('is_innovative', 0),
                    'competition_level': len(self.competitors)
                }
                
                # Convert to DataFrame for prediction
                X_pred = pd.DataFrame([features])
                
                # Predict adoption rate for this period
                predicted_adoption_rate = self.adoption_model.predict(X_pred)[0]
                
                # Calculate new adopters based on prediction
                new_adopters = predicted_adoption_rate * target_market_size - cumulative_adopters
                new_adopters = max(0, new_adopters) * seasonality
            else:
                # Calculate new adopters using Bass diffusion model
                t = i + 1  # Time period
                
                # Basic Bass model formula
                p = innovation_factor
                q = imitation_factor
                m = target_market_size
                
                # Adjust for marketing spend
                p_adjusted = p * (1 + 0.1 * np.log(1 + product_marketing / 50000))
                
                # Calculate adoption for this period
                if i == 0:
                    new_adopters = p_adjusted * m * seasonality
                else:
                    new_adopters = (p_adjusted + q * cumulative_adopters / m) * (m - cumulative_adopters) * seasonality
            
            # Add random noise
            noise_factor = np.random.normal(1, 0.1)  # 10% standard deviation
            new_adopters = max(0, new_adopters * noise_factor)
            
            # Update cumulative adopters
            previous_adopters = new_adopters
            cumulative_adopters += new_adopters
            
            # Calculate revenue, costs and profit
            period_revenue = new_adopters * product_price
            period_cost = new_adopters * product_cost
            period_marketing = product_marketing / periods  # Distribute marketing over periods
            period_profit = period_revenue - period_cost - period_marketing
            
            # Calculate cannibalization effect
            cannibalization_volume = new_adopters * cannibalization_rate
            cannibalization_revenue = cannibalization_volume * self.company_data.get('price', 100)
            
            # Calculate market share of new product
            market_share = (cumulative_adopters / target_market_size) * 100
            
            # Record results for this period
            period_result = {
                'period': period,
                'new_adopters': float(new_adopters),
                'cumulative_adopters': float(cumulative_adopters),
                'adoption_rate': float(cumulative_adopters / target_market_size * 100),
                'revenue': float(period_revenue),
                'cost': float(period_cost),
                'marketing': float(period_marketing),
                'profit': float(period_profit),
                'cannibalization_volume': float(cannibalization_volume),
                'cannibalization_revenue': float(cannibalization_revenue),
                'market_share': float(market_share),
                'seasonality_factor': float(seasonality)
            }
            
            results.append(period_result)
        
        # Calculate summary statistics
        total_revenue = sum(r['revenue'] for r in results)
        total_cost = sum(r['cost'] for r in results) + product_marketing
        total_profit = total_revenue - total_cost
        roi = total_profit / total_cost if total_cost > 0 else 0
        
        breakeven_period = None
        cumulative_profit = 0
        for i, r in enumerate(results):
            cumulative_profit += r['profit']
            if cumulative_profit >= 0 and breakeven_period is None:
                breakeven_period = i + 1
        
        final_market_share = results[-1]['market_share']
        total_cannibalization = sum(r['cannibalization_revenue'] for r in results)
        
        summary = {
            'product_name': product_name,
            'simulation_periods': periods,
            'total_adopters': float(cumulative_adopters),
            'market_penetration': float(cumulative_adopters / target_market_size * 100),
            'total_revenue': float(total_revenue),
            'total_cost': float(total_cost),
            'total_profit': float(total_profit),
            'roi': float(roi),
            'breakeven_period': breakeven_period,
            'final_market_share': float(final_market_share),
            'total_cannibalization': float(total_cannibalization),
            'net_revenue_impact': float(total_revenue - total_cannibalization),
            'recommendation': self._generate_product_recommendation(results, roi, breakeven_period)
        }
        
        simulation_result = {
            'simulation_type': 'product_launch',
            'timestamp': datetime.now().isoformat(),
            'input_parameters': product_data,
            'period_results': results,
            'summary': summary
        }
        
        self.simulation_results.append(simulation_result)
        return simulation_result
    
    def _generate_product_recommendation(self, results: List[Dict[str, Any]], 
                                        roi: float, breakeven_period: Optional[int]) -> Dict[str, Any]:
        """Generate recommendation based on product launch simulation results"""
        # Calculate key metrics
        final_adoption_rate = results[-1]['adoption_rate']
        
        # Determine if product launch is favorable
        if roi > 1.5 and breakeven_period and breakeven_period <= 12:
            recommendation = "Proceed with product launch"
            confidence = "High"
            rationale = f"The product is projected to be highly profitable with ROI of {roi:.2f} and breakeven in {breakeven_period} months."
        elif roi > 1:
            recommendation = "Consider proceeding with product launch"
            confidence = "Medium"
            rationale = f"The product is projected to be profitable with ROI of {roi:.2f}, but carefully monitor adoption rates."
        elif roi > 0.5:
            recommendation = "Proceed with caution"
            confidence = "Low"
            rationale = f"The product may be marginally profitable. Consider adjusting pricing or reducing costs."
        else:
            recommendation = "Reconsider product launch"
            confidence = "High"
            rationale = f"The product is projected to have poor ROI of {roi:.2f} and may not reach breakeven within the simulation period."
        
        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'rationale': rationale
        }
    
    def get_simulation_results(self) -> List[Dict[str, Any]]:
        """Get all simulation results"""
        return self.simulation_results
    
    def get_latest_simulation(self) -> Optional[Dict[str, Any]]:
        """Get the most recent simulation result"""
        if not self.simulation_results:
            return None
        return self.simulation_results[-1]
    
    def generate_monte_carlo_simulation(self, simulation_type: str, 
                                       parameters: Dict[str, Any], 
                                       iterations: int = 100) -> Dict[str, Any]:
        """
        Run a Monte Carlo simulation by varying input parameters
        
        Args:
            simulation_type: 'price_change', 'marketing_spend_change', or 'product_launch'
            parameters: Base parameters for the simulation
            iterations: Number of Monte Carlo iterations
            
        Returns:
            Dict containing Monte Carlo simulation results
        """
        if not self.company_data or not self.market_data:
            raise ValueError("Company and market data must be loaded before simulation")
            
        # Store original parameters
        original_params = parameters.copy()
        
        # Results storage
        mc_results = []
        
        for i in range(iterations):
            # Create variation of parameters
            varied_params = self._create_parameter_variation(simulation_type, original_params)
            
            # Run appropriate simulation
            if simulation_type == 'price_change':
                result = self.simulate_price_change(
                    varied_params.get('price_change_pct'),
                    varied_params.get('periods', 12)
                )
            elif simulation_type == 'marketing_spend_change':
                result = self.simulate_marketing_spend_change(
                    varied_params.get('spend_change_pct'),
                    varied_params.get('periods', 12)
                )
            elif simulation_type == 'product_launch':
                result = self.simulate_product_launch(
                    varied_params,
                    varied_params.get('periods', 24)
                )
            else:
                raise ValueError(f"Unknown simulation type: {simulation_type}")
            
            # Extract key metrics for Monte Carlo analysis
            key_metrics = self._extract_key_metrics(simulation_type, result)
            key_metrics['iteration'] = i
            key_metrics['varied_parameters'] = varied_params
            
            mc_results.append(key_metrics)
            
        # Analyze Monte Carlo results
        mc_analysis = self._analyze_monte_carlo_results(simulation_type, mc_results)
        
        monte_carlo_result = {
            'simulation_type': f'monte_carlo_{simulation_type}',
            'timestamp': datetime.now().isoformat(),
            'base_parameters': original_params,
            'iterations': iterations,
            'mc_results': mc_results,
            'analysis': mc_analysis
        }
        
        return monte_carlo_result
    
    def _create_parameter_variation(self, simulation_type: str, 
                                   original_params: Dict[str, Any]) -> Dict[str, Any]:
        """Create variation of parameters for Monte Carlo simulation"""
        varied_params = original_params.copy()
        
        if simulation_type == 'price_change':
            # Vary price change percentage
            base_price_change = original_params.get('price_change_pct', 0)
            varied_params['price_change_pct'] = base_price_change + np.random.normal(0, 2)
            
            # Vary market conditions
            self.market_growth_rate = max(0, self.market_data.get('growth_rate', 0.03) + np.random.normal(0, 0.01))
            
        elif simulation_type == 'marketing_spend_change':
            # Vary marketing spend change percentage
            base_spend_change = original_params.get('spend_change_pct', 0)
            varied_params['spend_change_pct'] = base_spend_change + np.random.normal(0, 5)
            
            # Vary marketing effectiveness
            if 'marketing_effectiveness' in self.company_data:
                self.company_data['marketing_effectiveness'] = max(0.1, 
                    self.company_data.get('marketing_effectiveness', 0.3) + np.random.normal(0, 0.05))
            
        elif simulation_type == 'product_launch':
            # Vary product parameters
            varied_params['price'] = original_params.get('price', 100) * (1 + np.random.normal(0, 0.05))
            varied_params['cost'] = original_params.get('cost', 60) * (1 + np.random.normal(0, 0.03))
            varied_params['target_market_size'] = original_params.get('target_market_size', 100000) * (1 + np.random.normal(0, 0.1))
            
            # Vary adoption parameters
            varied_params['innovation_factor'] = max(0.01, original_params.get('innovation_factor', 0.03) + np.random.normal(0, 0.005))
            varied_params['imitation_factor'] = max(0.1, original_params.get('imitation_factor', 0.38) + np.random.normal(0, 0.05))
            varied_params['cannibalization_rate'] = max(0, min(0.5, original_params.get('cannibalization_rate', 0.2) + np.random.normal(0, 0.03)))
        
        return varied_params
    
    def _extract_key_metrics(self, simulation_type: str, 
                            result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from simulation results for Monte Carlo analysis"""
        summary = result.get('summary', {})
        key_metrics = {}
        
        if simulation_type == 'price_change':
            key_metrics = {
                'total_revenue': summary.get('total_revenue', 0),
                'final_market_share': summary.get('final_market_share', 0),
                'market_share_change': summary.get('market_share_change', 0)
            }
        
        elif simulation_type == 'marketing_spend_change':
            key_metrics = {
                'total_revenue': summary.get('total_revenue', 0),
                'total_profit': summary.get('total_profit', 0),
                'marketing_roi': summary.get('marketing_roi', 0),
                'market_share_change': summary.get('market_share_change', 0)
            }
        
        elif simulation_type == 'product_launch':
            key_metrics = {
                'total_revenue': summary.get('total_revenue', 0),
                'total_profit': summary.get('total_profit', 0),
                'roi': summary.get('roi', 0),
                'breakeven_period': summary.get('breakeven_period'),
                'market_penetration': summary.get('market_penetration', 0)
            }
        
        return key_metrics
    
    def _analyze_monte_carlo_results(self, simulation_type: str, 
                                    mc_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze Monte Carlo simulation results"""
        # Extract key metrics into separate lists
        metrics = {}
        
        # Determine which metrics to analyze based on simulation type
        if simulation_type == 'price_change':
            metric_keys = ['total_revenue', 'final_market_share', 'market_share_change']
        elif simulation_type == 'marketing_spend_change':
            metric_keys = ['total_revenue', 'total_profit', 'marketing_roi', 'market_share_change']
        elif simulation_type == 'product_launch':
            metric_keys = ['total_revenue', 'total_profit', 'roi', 'breakeven_period', 'market_penetration']
        
        # Extract metrics
        for key in metric_keys:
            metrics[key] = [r.get(key, 0) for r in mc_results if key in r]
        
        # Calculate statistics for each metric
        analysis = {}
        for key, values in metrics.items():
            if key == 'breakeven_period':
                # Filter out None values
                valid_values = [v for v in values if v is not None]
                
                if valid_values:
                    analysis[key] = {
                        'mean': float(np.mean(valid_values)),
                        'median': float(np.median(valid_values)),
                        'std': float(np.std(valid_values)),
                        'min': float(np.min(valid_values)),
                        'max': float(np.max(valid_values)),
                        'percentile_10': float(np.percentile(valid_values, 10)),
                        'percentile_90': float(np.percentile(valid_values, 90)),
                        'no_breakeven_pct': (len(values) - len(valid_values)) / len(values) * 100 if values else 0
                    }
                else:
                    analysis[key] = {
                        'mean': None,
                        'median': None,
                        'std': None,
                        'min': None,
                        'max': None,
                        'percentile_10': None,
                        'percentile_90': None,
                        'no_breakeven_pct': 100.0
                    }
            else:
                analysis[key] = {
                    'mean': float(np.mean(values)),
                    'median': float(np.median(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'percentile_10': float(np.percentile(values, 10)),
                    'percentile_90': float(np.percentile(values, 90))
                }
        
        # Calculate probability of success
        success_probability = self._calculate_success_probability(simulation_type, mc_results)
        
        # Generate risk assessment
        risk_assessment = self._generate_risk_assessment(simulation_type, analysis)
        
        return {
            'metric_statistics': analysis,
            'success_probability': success_probability,
            'risk_assessment': risk_assessment
        }
    
    def _calculate_success_probability(self, simulation_type: str, 
                                      mc_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate probability of success based on Monte Carlo results"""
        probabilities = {}
        
        if simulation_type == 'price_change':
            # Success: Revenue increase
            revenue_increase_count = sum(1 for r in mc_results if r.get('total_revenue', 0) > 0)
            probabilities['revenue_increase'] = revenue_increase_count / len(mc_results) if mc_results else 0
            
            # Success: Market share maintained or increased
            market_share_maintained = sum(1 for r in mc_results if r.get('market_share_change', -1) >= 0)
            probabilities['market_share_maintained'] = market_share_maintained / len(mc_results) if mc_results else 0
            
            # Overall success: Revenue up and market share not significantly down
            overall_success = sum(1 for r in mc_results if r.get('total_revenue', 0) > 0 and r.get('market_share_change', -100) > -3)
            probabilities['overall_success'] = overall_success / len(mc_results) if mc_results else 0
            
        elif simulation_type == 'marketing_spend_change':
            # Success: Positive ROI
            positive_roi = sum(1 for r in mc_results if r.get('marketing_roi', 0) > 0)
            probabilities['positive_roi'] = positive_roi / len(mc_results) if mc_results else 0
            
            # Success: ROI > 1
            roi_above_one = sum(1 for r in mc_results if r.get('marketing_roi', 0) > 1)
            probabilities['roi_above_one'] = roi_above_one / len(mc_results) if mc_results else 0
            
            # Success: Market share increase
            market_share_increase = sum(1 for r in mc_results if r.get('market_share_change', 0) > 0)
            probabilities['market_share_increase'] = market_share_increase / len(mc_results) if mc_results else 0
            
            # Overall success: ROI > 1 and market share up
            overall_success = sum(1 for r in mc_results if r.get('marketing_roi', 0) > 1 and r.get('market_share_change', 0) > 0)
            probabilities['overall_success'] = overall_success / len(mc_results) if mc_results else 0
            
        elif simulation_type == 'product_launch':
            # Success: Positive ROI
            positive_roi = sum(1 for r in mc_results if r.get('roi', 0) > 0)
            probabilities['positive_roi'] = positive_roi / len(mc_results) if mc_results else 0
            
            # Success: ROI > 1
            roi_above_one = sum(1 for r in mc_results if r.get('roi', 0) > 1)
            probabilities['roi_above_one'] = roi_above_one / len(mc_results) if mc_results else 0
            
            # Success: Breakeven within 18 months
            breakeven_18_months = sum(1 for r in mc_results if r.get('breakeven_period') is not None and r.get('breakeven_period', 100) <= 18)
            probabilities['breakeven_18_months'] = breakeven_18_months / len(mc_results) if mc_results else 0
            
            # Overall success: ROI > 1 and breakeven within 18 months
            overall_success = sum(1 for r in mc_results if r.get('roi', 0) > 1 and r.get('breakeven_period') is not None and r.get('breakeven_period', 100) <= 18)
            probabilities['overall_success'] = overall_success / len(mc_results) if mc_results else 0
        
        return probabilities
    
    def _generate_risk_assessment(self, simulation_type: str, 
                                 analysis: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Generate risk assessment based on Monte Carlo analysis"""
        risk_assessment = {
            'risk_level': None,
            'key_risks': [],
            'upside_potential': [],
            'recommendation': None
        }
        
        if simulation_type == 'price_change':
            # Assess revenue risk
            revenue_mean = analysis.get('total_revenue', {}).get('mean', 0)
            revenue_std = analysis.get('total_revenue', {}).get('std', 0)
            revenue_downside = analysis.get('total_revenue', {}).get('percentile_10', 0)
            
            market_share_mean = analysis.get('market_share_change', {}).get('mean', 0)
            market_share_downside = analysis.get('market_share_change', {}).get('percentile_10', 0)
            
            # Determine risk level
            if revenue_downside < 0 and market_share_downside < -5:
                risk_level = "High"
                key_risks = [
                    f"90% chance of revenue loss exceeding {abs(revenue_downside):.2f}",
                    f"90% chance of market share loss exceeding {abs(market_share_downside):.2f}%"
                ]
            elif revenue_downside < 0:
                risk_level = "Medium"
                key_risks = [f"90% chance of revenue loss exceeding {abs(revenue_downside):.2f}"]
            elif market_share_downside < -3:
                risk_level = "Medium"
                key_risks = [f"90% chance of market share loss exceeding {abs(market_share_downside):.2f}%"]
            else:
                risk_level = "Low"
                key_risks = []
            
            # Determine upside potential
            revenue_upside = analysis.get('total_revenue', {}).get('percentile_90', 0)
            market_share_upside = analysis.get('market_share_change', {}).get('percentile_90', 0)
            
            upside_potential = []
            if revenue_upside > 0:
                upside_potential.append(f"10% chance of revenue gain exceeding {revenue_upside:.2f}")
            if market_share_upside > 0:
                upside_potential.append(f"10% chance of market share gain exceeding {market_share_upside:.2f}%")
            
            # Generate recommendation
            if risk_level == "Low" and revenue_mean > 0:
                recommendation = "Proceed with the price change - low risk with positive expected revenue impact"
            elif risk_level == "Medium" and revenue_mean > 0:
                recommendation = "Consider proceeding with caution - medium risk but positive expected revenue impact"
            else:
                recommendation = "Reconsider the price change - high risk or negative expected revenue impact"
            
        elif simulation_type == 'marketing_spend_change':
            # Assess ROI risk
            roi_mean = analysis.get('marketing_roi', {}).get('mean', 0)
            roi_std = analysis.get('marketing_roi', {}).get('std', 0)
            roi_downside = analysis.get('marketing_roi', {}).get('percentile_10', 0)
            
            profit_mean = analysis.get('total_profit', {}).get('mean', 0)
            profit_downside = analysis.get('total_profit', {}).get('percentile_10', 0)
            
            # Determine risk level
            if roi_downside < 0 and profit_downside < 0:
                risk_level = "High"
                key_risks = [
                    f"90% chance of negative ROI below {roi_downside:.2f}",
                    f"90% chance of profit loss exceeding {abs(profit_downside):.2f}"
                ]
            elif roi_downside < 0:
                risk_level = "Medium"
                key_risks = [f"90% chance of negative ROI below {roi_downside:.2f}"]
            elif profit_downside < 0:
                risk_level = "Medium"
                key_risks = [f"90% chance of profit loss exceeding {abs(profit_downside):.2f}"]
            else:
                risk_level = "Low"
                key_risks = []
            
            # Determine upside potential
            roi_upside = analysis.get('marketing_roi', {}).get('percentile_90', 0)
            profit_upside = analysis.get('total_profit', {}).get('percentile_90', 0)
            
            upside_potential = []
            if roi_upside > 1:
                upside_potential.append(f"10% chance of ROI exceeding {roi_upside:.2f}")
            if profit_upside > 0:
                upside_potential.append(f"10% chance of profit gain exceeding {profit_upside:.2f}")
            
            # Generate recommendation
            if risk_level == "Low" and roi_mean > 1:
                recommendation = "Proceed with the marketing spend change - low risk with positive expected ROI"
            elif risk_level == "Medium" and roi_mean > 0:
                recommendation = "Consider proceeding with caution - medium risk but positive expected ROI"
            else:
                recommendation = "Reconsider the marketing spend change - high risk or negative expected ROI"
            
        elif simulation_type == 'product_launch':
            # Assess ROI risk
            roi_mean = analysis.get('roi', {}).get('mean', 0)
            roi_std = analysis.get('roi', {}).get('std', 0)
            roi_downside = analysis.get('roi', {}).get('percentile_10', 0)
            
            breakeven_mean = analysis.get('breakeven_period', {}).get('mean')
            no_breakeven_pct = analysis.get('breakeven_period', {}).get('no_breakeven_pct', 0)
            
            # Determine risk level
            if roi_downside < 0 or no_breakeven_pct > 50:
                risk_level = "High"
                key_risks = [
                    f"90% chance of ROI below {roi_downside:.2f}",
                    f"{no_breakeven_pct:.1f}% chance of never reaching breakeven"
                ]
            elif roi_downside < 0.5:
                risk_level = "Medium"
                key_risks = [f"90% chance of ROI below {roi_downside:.2f}"]
            elif no_breakeven_pct > 20:
                risk_level = "Medium"
                key_risks = [f"{no_breakeven_pct:.1f}% chance of never reaching breakeven"]
            else:
                risk_level = "Low"
                key_risks = []
            
            # Determine upside potential
            roi_upside = analysis.get('roi', {}).get('percentile_90', 0)
            market_penetration_upside = analysis.get('market_penetration', {}).get('percentile_90', 0)
            
            upside_potential = []
            if roi_upside > 1.5:
                upside_potential.append(f"10% chance of ROI exceeding {roi_upside:.2f}")
            if market_penetration_upside > 0:
                upside_potential.append(f"10% chance of market penetration exceeding {market_penetration_upside:.2f}%")
            
            # Generate recommendation
            if risk_level == "Low" and roi_mean > 1:
                recommendation = "Proceed with the product launch - low risk with positive expected ROI"
            elif risk_level == "Medium" and roi_mean > 0.7:
                recommendation = "Consider proceeding with caution - medium risk but potentially positive ROI"
            else:
                recommendation = "Reconsider the product launch - high risk or low expected ROI"
        
        risk_assessment['risk_level'] = risk_level
        risk_assessment['key_risks'] = key_risks
        risk_assessment['upside_potential'] = upside_potential
        risk_assessment['recommendation'] = recommendation
        
        return risk_assessment
    def generate_scenario_comparison(self, scenario_type: str, 
                                    scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple scenarios to identify optimal business decisions
        
        Args:
            scenario_type: 'price_change', 'marketing_spend_change', or 'product_launch'
            scenarios: List of scenario parameters to compare
            
        Returns:
            Dict containing comparison results and recommendations
        """
        if not self.company_data or not self.market_data:
            raise ValueError("Company and market data must be loaded before simulation")
        
        # Results storage
        scenario_results = []
        
        # Run simulations for each scenario
        for i, scenario in enumerate(scenarios):
            scenario_name = scenario.get('name', f"Scenario {i+1}")
            
            # Run appropriate simulation
            if scenario_type == 'price_change':
                price_change_pct = scenario.get('price_change_pct', 0)
                periods = scenario.get('periods', 12)
                
                result = self.simulate_price_change(price_change_pct, periods)
                
            elif scenario_type == 'marketing_spend_change':
                spend_change_pct = scenario.get('spend_change_pct', 0)
                periods = scenario.get('periods', 12)
                
                result = self.simulate_marketing_spend_change(spend_change_pct, periods)
                
            elif scenario_type == 'product_launch':
                periods = scenario.get('periods', 24)
                
                result = self.simulate_product_launch(scenario, periods)
                
            else:
                raise ValueError(f"Unknown scenario type: {scenario_type}")
            
            # Add scenario name to result
            result['scenario_name'] = scenario_name
            scenario_results.append(result)
        
        # Compare scenarios
        comparison = self._compare_scenarios(scenario_type, scenario_results)
        
        return {
            'scenario_type': scenario_type,
            'timestamp': datetime.now().isoformat(),
            'scenarios': scenario_results,
            'comparison': comparison
        }
    
    def _compare_scenarios(self, scenario_type: str, 
                          scenario_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare scenario results and identify the best option"""
        # Extract key metrics for comparison
        comparison_metrics = []
        
        for result in scenario_results:
            summary = result.get('summary', {})
            scenario_name = result.get('scenario_name', 'Unnamed Scenario')
            
            if scenario_type == 'price_change':
                metrics = {
                    'scenario_name': scenario_name,
                    'price_change_pct': summary.get('price_change_pct', 0),
                    'total_revenue': summary.get('total_revenue', 0),
                    'market_share_change': summary.get('market_share_change', 0),
                    'recommendation': summary.get('recommendation', {}).get('recommendation', '')
                }
            
            elif scenario_type == 'marketing_spend_change':
                metrics = {
                    'scenario_name': scenario_name,
                    'spend_change_pct': summary.get('spend_change_pct', 0),
                    'total_revenue': summary.get('total_revenue', 0),
                    'total_profit': summary.get('total_profit', 0),
                    'marketing_roi': summary.get('marketing_roi', 0),
                    'market_share_change': summary.get('market_share_change', 0),
                    'recommendation': summary.get('recommendation', {}).get('recommendation', '')
                }
            
            elif scenario_type == 'product_launch':
                metrics = {
                    'scenario_name': scenario_name,
                    'product_name': summary.get('product_name', 'New Product'),
                    'total_revenue': summary.get('total_revenue', 0),
                    'total_profit': summary.get('total_profit', 0),
                    'roi': summary.get('roi', 0),
                    'breakeven_period': summary.get('breakeven_period'),
                    'market_penetration': summary.get('market_penetration', 0),
                    'recommendation': summary.get('recommendation', {}).get('recommendation', '')
                }
            
            comparison_metrics.append(metrics)
        
        # Determine the best scenario based on key metrics
        best_scenario = self._identify_best_scenario(scenario_type, comparison_metrics)
        
        # Generate comparison insights
        insights = self._generate_comparison_insights(scenario_type, comparison_metrics, best_scenario)
        
        return {
            'metrics': comparison_metrics,
            'best_scenario': best_scenario,
            'insights': insights
        }
    
    def _identify_best_scenario(self, scenario_type: str, 
                               comparison_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify the best scenario based on key metrics"""
        if not comparison_metrics:
            return None
        
        # Determine the best scenario based on scenario type
        if scenario_type == 'price_change':
            # Sort by total revenue (descending)
            sorted_scenarios = sorted(comparison_metrics, key=lambda x: x.get('total_revenue', 0), reverse=True)
            
            # Best scenario is the one with highest revenue that doesn't significantly reduce market share
            for scenario in sorted_scenarios:
                if scenario.get('market_share_change', -100) > -5:  # Threshold: -5% market share change
                    return scenario
            
            # If all scenarios reduce market share significantly, return the one with highest revenue
            return sorted_scenarios[0] if sorted_scenarios else None
            
        elif scenario_type == 'marketing_spend_change':
            # Sort by marketing ROI (descending)
            return max(comparison_metrics, key=lambda x: x.get('marketing_roi', 0), default=None)
            
        elif scenario_type == 'product_launch':
            # Sort by ROI (descending)
            sorted_by_roi = sorted(comparison_metrics, key=lambda x: x.get('roi', 0), reverse=True)
            
            # Best scenario is the one with highest ROI that breaks even within a reasonable time
            for scenario in sorted_by_roi:
                breakeven = scenario.get('breakeven_period')
                if breakeven is not None and breakeven <= 18:  # Threshold: 18 months
                    return scenario
            
            # If no scenario breaks even within threshold, return the one with highest ROI
            return sorted_by_roi[0] if sorted_by_roi else None
    
    def _generate_comparison_insights(self, scenario_type: str, 
                                     comparison_metrics: List[Dict[str, Any]],
                                     best_scenario: Dict[str, Any]) -> List[str]:
        """Generate insights based on scenario comparison"""
        insights = []
        
        if not comparison_metrics or len(comparison_metrics) < 2:
            return ["Not enough scenarios to generate meaningful comparison insights."]
        
        if scenario_type == 'price_change':
            # Find the range of price changes
            price_changes = [s.get('price_change_pct', 0) for s in comparison_metrics]
            min_price_change = min(price_changes)
            max_price_change = max(price_changes)
            
            # Analyze price elasticity
            if min_price_change < 0 and max_price_change > 0:
                # Find scenarios with price increase and decrease
                price_increase_scenarios = [s for s in comparison_metrics if s.get('price_change_pct', 0) > 0]
                price_decrease_scenarios = [s for s in comparison_metrics if s.get('price_change_pct', 0) < 0]
                
                # Compare average revenue impact
                if price_increase_scenarios and price_decrease_scenarios:
                    avg_revenue_increase = sum(s.get('total_revenue', 0) for s in price_increase_scenarios) / len(price_increase_scenarios)
                    avg_revenue_decrease = sum(s.get('total_revenue', 0) for s in price_decrease_scenarios) / len(price_decrease_scenarios)
                    
                    if avg_revenue_increase > avg_revenue_decrease:
                        insights.append("Price increases generally lead to higher revenue than price decreases, suggesting low price elasticity.")
                    else:
                        insights.append("Price decreases generally lead to higher revenue than price increases, suggesting high price elasticity.")
            
            # Analyze best scenario
            if best_scenario:
                price_change = best_scenario.get('price_change_pct', 0)
                if price_change > 0:
                    insights.append(f"The optimal price change appears to be an increase of {price_change:.1f}%, suggesting potential for premium positioning.")
                elif price_change < 0:
                    insights.append(f"The optimal price change appears to be a decrease of {abs(price_change):.1f}%, suggesting potential to gain market share through competitive pricing.")
                else:
                    insights.append("Maintaining current pricing appears to be the optimal strategy based on the scenarios analyzed.")
        
        elif scenario_type == 'marketing_spend_change':
            # Find the range of marketing spend changes
            spend_changes = [s.get('spend_change_pct', 0) for s in comparison_metrics]
            min_spend_change = min(spend_changes)
            max_spend_change = max(spend_changes)
            
            # Analyze diminishing returns
            if len(comparison_metrics) >= 3:
                # Sort by spend change
                sorted_scenarios = sorted(comparison_metrics, key=lambda x: x.get('spend_change_pct', 0))
                
                # Check if ROI decreases as spend increases
                rois = [s.get('marketing_roi', 0) for s in sorted_scenarios]
                if all(rois[i] >= rois[i+1] for i in range(len(rois)-1)):
                    insights.append("Marketing shows clear diminishing returns as spend increases. Consider smaller, targeted increases.")
            
            # Analyze best scenario
            if best_scenario:
                spend_change = best_scenario.get('spend_change_pct', 0)
                roi = best_scenario.get('marketing_roi', 0)
                
                if spend_change > 20 and roi > 1.5:
                    insights.append(f"Significant marketing investment ({spend_change:.1f}% increase) shows strong potential with an estimated ROI of {roi:.2f}.")
                elif spend_change > 0 and roi > 1:
                    insights.append(f"Moderate marketing investment ({spend_change:.1f}% increase) appears optimal with a positive ROI of {roi:.2f}.")
                elif spend_change < 0 and roi > 0:
                    insights.append(f"Reducing marketing spend by {abs(spend_change):.1f}% may improve overall profitability while maintaining acceptable performance.")
        
        elif scenario_type == 'product_launch':
            # Analyze breakeven periods
            breakeven_periods = [s.get('breakeven_period') for s in comparison_metrics if s.get('breakeven_period') is not None]
            if breakeven_periods:
                min_breakeven = min(breakeven_periods)
                max_breakeven = max(breakeven_periods)
                avg_breakeven = sum(breakeven_periods) / len(breakeven_periods)
                
                insights.append(f"Product launch scenarios show breakeven periods ranging from {min_breakeven} to {max_breakeven} months, with an average of {avg_breakeven:.1f} months.")
            
            # Analyze ROI distribution
            rois = [s.get('roi', 0) for s in comparison_metrics]
            if rois:
                max_roi = max(rois)
                min_roi = min(rois)
                
                if max_roi > 1.5:
                    max_roi_scenario = next(s for s in comparison_metrics if s.get('roi', 0) == max_roi)
                    insights.append(f"The '{max_roi_scenario.get('product_name', 'highest ROI product')}' scenario shows exceptional potential with an ROI of {max_roi:.2f}.")
                
                if min_roi < 0.5:
                    min_roi_scenario = next(s for s in comparison_metrics if s.get('roi', 0) == min_roi)
                    insights.append(f"The '{min_roi_scenario.get('product_name', 'lowest ROI product')}' scenario shows poor potential with an ROI of {min_roi:.2f} and should be reconsidered.")
            
            # Analyze cannibalization if available
            if any('total_cannibalization' in s for s in comparison_metrics):
                # Find scenario with highest net revenue impact (revenue - cannibalization)
                net_impacts = [(s.get('total_revenue', 0) - s.get('total_cannibalization', 0), s) for s in comparison_metrics]
                _, best_net_impact_scenario = max(net_impacts, key=lambda x: x[0])
                
                insights.append(f"When accounting for cannibalization effects, the '{best_net_impact_scenario.get('product_name', 'optimal product')}' scenario provides the best net revenue impact.")
        
        # Add general insight about the best scenario
        if best_scenario:
            insights.append(f"The '{best_scenario.get('scenario_name', 'optimal scenario')}' appears to be the most promising option based on key performance metrics.")
        
        return insights

    def generate_what_if_analysis(self, base_scenario: Dict[str, Any], 
                                 variables: List[Dict[str, Any]],
                                 simulation_type: str) -> Dict[str, Any]:
        """
        Perform a what-if analysis by varying specific variables
        
        Args:
            base_scenario: Base scenario parameters
            variables: List of variables to adjust and their ranges
                Each variable is a dict with keys:
                - 'name': Variable name
                - 'values': List of values to test
            simulation_type: 'price_change', 'marketing_spend_change', or 'product_launch'
            
        Returns:
            Dict containing what-if analysis results
        """
        if not self.company_data or not self.market_data:
            raise ValueError("Company and market data must be loaded before simulation")
        
        # Generate all combinations of variable values
        variable_combinations = self._generate_variable_combinations(base_scenario, variables)
        
        # Run simulations for each combination
        results = []
        
        for combination in variable_combinations:
            # Run appropriate simulation
            if simulation_type == 'price_change':
                price_change_pct = combination.get('price_change_pct', 0)
                periods = combination.get('periods', 12)
                
                result = self.simulate_price_change(price_change_pct, periods)
                
            elif simulation_type == 'marketing_spend_change':
                spend_change_pct = combination.get('spend_change_pct', 0)
                periods = combination.get('periods', 12)
                
                result = self.simulate_marketing_spend_change(spend_change_pct, periods)
                
            elif simulation_type == 'product_launch':
                periods = combination.get('periods', 24)
                
                result = self.simulate_product_launch(combination, periods)
                
            else:
                raise ValueError(f"Unknown simulation type: {simulation_type}")
            
            # Add variable values to result
            result['variable_values'] = {var['name']: combination.get(var['name']) for var in variables}
            results.append(result)
        
        # Analyze the results
        analysis = self._analyze_what_if_results(simulation_type, variables, results)
        
        return {
            'simulation_type': f'what_if_{simulation_type}',
            'timestamp': datetime.now().isoformat(),
            'base_scenario': base_scenario,
            'variables': variables,
            'results': results,
            'analysis': analysis
        }
    
    def _generate_variable_combinations(self, base_scenario: Dict[str, Any], 
                                       variables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate all combinations of variable values"""
        if not variables:
            return [base_scenario.copy()]
        
        # Start with the first variable
        current_var = variables[0]
        remaining_vars = variables[1:]
        
        combinations = []
        
        # For each value of the current variable
        for value in current_var['values']:
            # Create a new scenario with this value
            new_scenario = base_scenario.copy()
            new_scenario[current_var['name']] = value
            
            # If there are more variables, recurse
            if remaining_vars:
                sub_combinations = self._generate_variable_combinations(new_scenario, remaining_vars)
                combinations.extend(sub_combinations)
            else:
                combinations.append(new_scenario)
        
        return combinations
    
    def _analyze_what_if_results(self, simulation_type: str, 
                                variables: List[Dict[str, Any]],
                                results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze what-if analysis results"""
        # Extract key metrics based on simulation type
        if simulation_type == 'price_change':
            key_metrics = ['total_revenue', 'final_market_share', 'market_share_change']
        elif simulation_type == 'marketing_spend_change':
            key_metrics = ['total_revenue', 'total_profit', 'marketing_roi', 'market_share_change']
        elif simulation_type == 'product_launch':
            key_metrics = ['total_revenue', 'total_profit', 'roi', 'breakeven_period', 'market_penetration']
        
        # Analyze impact of each variable
        variable_impacts = {}
        
        for var in variables:
            var_name = var['name']
            var_values = var['values']
            
            # Group results by this variable's values
            value_groups = {}
            for value in var_values:
                value_groups[value] = [r for r in results if r['variable_values'].get(var_name) == value]
            
            # Calculate average metrics for each value
            metric_by_value = {}
            for value, group in value_groups.items():
                if not group:
                    continue
                
                metrics = {}
                for metric in key_metrics:
                    if metric == 'breakeven_period':
                        # Handle None values for breakeven period
                        valid_values = [r['summary'].get(metric) for r in group if r['summary'].get(metric) is not None]
                        metrics[metric] = sum(valid_values) / len(valid_values) if valid_values else None
                    else:
                        metrics[metric] = sum(r['summary'].get(metric, 0) for r in group) / len(group)
                
                metric_by_value[value] = metrics
            
            # Calculate sensitivity for each metric
            sensitivity = {}
            for metric in key_metrics:
                if metric == 'breakeven_period':
                    # Skip breakeven period for sensitivity calculation
                    continue
                
                values = [v for v in var_values if v in metric_by_value]
                if len(values) < 2:
                    continue
                
                metric_values = [metric_by_value[v][metric] for v in values if metric in metric_by_value[v]]
                if not metric_values or min(metric_values) == max(metric_values):
                    sensitivity[metric] = 0
                else:
                    # Normalize to percentage change relative to range of the variable
                    var_range = max(values) - min(values)
                    metric_range = max(metric_values) - min(metric_values)
                    base_metric = sum(metric_values) / len(metric_values)
                    
                    if var_range == 0 or base_metric == 0:
                        sensitivity[metric] = 0
                    else:
                        sensitivity[metric] = (metric_range / base_metric) / (var_range / ((max(values) + min(values)) / 2))
            
            variable_impacts[var_name] = {
                'metric_by_value': metric_by_value,
                'sensitivity': sensitivity
            }
        
        # Find optimal values for each variable
        optimal_values = {}
        
        for var in variables:
            var_name = var['name']
            
            if var_name not in variable_impacts:
                continue
            
            impact = variable_impacts[var_name]
            metric_by_value = impact['metric_by_value']
            
            # Determine the key metric to optimize based on simulation type
            if simulation_type == 'price_change':
                # Optimize for revenue, but penalize for market share loss
                optimal_value = None
                best_score = float('-inf')
                
                for value, metrics in metric_by_value.items():
                    revenue = metrics.get('total_revenue', 0)
                    market_share_change = metrics.get('market_share_change', 0)
                    
                    # Penalize market share loss
                    score = revenue * (1 + market_share_change / 100)
                    
                    if score > best_score:
                        best_score = score
                        optimal_value = value
                
            elif simulation_type == 'marketing_spend_change':
                # Optimize for marketing ROI
                optimal_value = max(metric_by_value.items(), 
                                   key=lambda x: x[1].get('marketing_roi', 0))[0]
                
            elif simulation_type == 'product_launch':
                # Optimize for ROI, but consider breakeven period
                optimal_value = None
                best_score = float('-inf')
                
                for value, metrics in metric_by_value.items():
                    roi = metrics.get('roi', 0)
                    breakeven = metrics.get('breakeven_period')
                    
                    # Penalize long breakeven periods
                    if breakeven is None:
                        score = roi * 0.5  # Severe penalty for no breakeven
                    else:
                        score = roi * (24 / (breakeven + 6))  # Mild penalty for longer breakeven
                    
                    if score > best_score:
                        best_score = score
                        optimal_value = value
            
            optimal_values[var_name] = optimal_value
        
        # Generate insights
        insights = self._generate_what_if_insights(simulation_type, variable_impacts, optimal_values)
        
        return {
            'variable_impacts': variable_impacts,
            'optimal_values': optimal_values,
            'insights': insights
        }
    
    def _generate_what_if_insights(self, simulation_type: str, 
                                  variable_impacts: Dict[str, Dict],
                                  optimal_values: Dict[str, Any]) -> List[str]:
        """Generate insights from what-if analysis"""
        insights = []
        
        # Add insights for each variable
        for var_name, impact in variable_impacts.items():
            sensitivity = impact['sensitivity']
            
            # Find the most sensitive metric
            if sensitivity:
                most_sensitive_metric = max(sensitivity.items(), key=lambda x: abs(x[1]))
                metric_name, sensitivity_value = most_sensitive_metric
                
                if abs(sensitivity_value) > 0.5:
                    direction = "positively" if sensitivity_value > 0 else "negatively"
                    insights.append(f"{var_name} strongly {direction} impacts {metric_name} (sensitivity: {abs(sensitivity_value):.2f}).")
                elif abs(sensitivity_value) > 0.1:
                    direction = "positively" if sensitivity_value > 0 else "negatively"
                    insights.append(f"{var_name} moderately {direction} impacts {metric_name} (sensitivity: {abs(sensitivity_value):.2f}).")
            
            # Add insight about optimal value
            if var_name in optimal_values:
                optimal = optimal_values[var_name]
                insights.append(f"The optimal value for {var_name} appears to be {optimal}.")
        
        # Add simulation-specific insights
        if simulation_type == 'price_change':
            price_change_var = 'price_change_pct'
            if price_change_var in optimal_values:
                optimal_price_change = optimal_values[price_change_var]
                if optimal_price_change > 0:
                    insights.append(f"A price increase of {optimal_price_change}% appears to be optimal, suggesting potential for premium positioning.")
                elif optimal_price_change < 0:
                    insights.append(f"A price decrease of {abs(optimal_price_change)}% appears to be optimal, suggesting potential for market share growth.")
                else:
                    insights.append("Maintaining current pricing appears to be optimal based on the analysis.")
        
        elif simulation_type == 'marketing_spend_change':
            spend_change_var = 'spend_change_pct'
            if spend_change_var in optimal_values:
                optimal_spend_change = optimal_values[spend_change_var]
                if optimal_spend_change > 20:
                    insights.append(f"A significant marketing investment increase of {optimal_spend_change}% appears to be optimal, suggesting strong growth potential.")
                elif optimal_spend_change > 0:
                    insights.append(f"A moderate marketing investment increase of {optimal_spend_change}% appears to be optimal.")
                elif optimal_spend_change < 0:
                    insights.append(f"Reducing marketing spend by {abs(optimal_spend_change)}% appears to be optimal, suggesting potential inefficiencies in current spending.")
        
        elif simulation_type == 'product_launch':
            price_var = 'price'
            if price_var in optimal_values:
                optimal_price = optimal_values[price_var]
                insights.append(f"The optimal price point for the new product appears to be {optimal_price}.")
            
            marketing_var = 'marketing_spend'
            if marketing_var in optimal_values:
                optimal_marketing = optimal_values[marketing_var]
                insights.append(f"The optimal marketing budget for the product launch appears to be {optimal_marketing}.")
        
        # Add a summary insight
        if optimal_values:
            insights.append("For best results, consider adjusting the variables to their optimal values as identified in this analysis.")
        
        return insights