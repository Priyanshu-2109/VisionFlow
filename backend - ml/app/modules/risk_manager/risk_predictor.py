import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
from datetime import datetime
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
import scipy.stats as stats

logger = logging.getLogger(__name__)

class RiskPredictor:
    """
    Predict business risks based on various factors
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.feature_importance = {}
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """Load risk prediction model from file"""
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Successfully loaded risk model from {model_path}")
            
            # Extract feature importance if available
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = dict(zip(self.model.feature_names_in_, self.model.feature_importances_))
                
        except Exception as e:
            logger.error(f"Error loading risk model: {str(e)}")
            self.model = None
    
    def predict_financial_risks(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict financial risks based on financial data
        
        Args:
            financial_data: Dictionary containing financial metrics
            
        Returns:
            Dictionary containing risk predictions
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'overall_risk_score': 0,
            'risk_categories': {},
            'risk_factors': [],
            'recommendations': []
        }
        
        # If model is available, use it for prediction
        if self.model:
            return self._predict_with_model(financial_data)
        
        # Otherwise use rule-based approach
        return self._predict_financial_risks_rule_based(financial_data)
    def _predict_with_model(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict risks using machine learning model"""
        try:
            # Prepare input features
            features = self._extract_features(financial_data)
            
            # Make prediction
            risk_score = self.model.predict([features])[0]
            
            # Get feature importance for this prediction
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = dict(zip(self.model.feature_names_in_, self.model.feature_importances_))
                top_factors = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            else:
                top_factors = []
            
            # Create result structure
            results = {
                'timestamp': datetime.now().isoformat(),
                'overall_risk_score': float(risk_score),
                'risk_categories': self._categorize_risk(risk_score),
                'risk_factors': [{"factor": factor, "importance": float(importance)} for factor, importance in top_factors],
                'recommendations': self._generate_recommendations(risk_score, top_factors)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error predicting with model: {str(e)}")
            # Fallback to rule-based approach
            return self._predict_financial_risks_rule_based(financial_data)
    
    def _extract_features(self, financial_data: Dict[str, Any]) -> List[float]:
        """Extract features from financial data for model input"""
        # Define expected features in the order the model expects
        expected_features = [
            'current_ratio', 'quick_ratio', 'debt_to_equity', 'interest_coverage',
            'gross_margin', 'operating_margin', 'net_profit_margin', 'return_on_assets',
            'return_on_equity', 'inventory_turnover', 'days_sales_outstanding',
            'asset_turnover', 'cash_conversion_cycle', 'revenue_growth', 'ebitda_growth'
        ]
        
        # Extract features, using 0 for missing values
        features = []
        for feature in expected_features:
            value = financial_data.get(feature, 0)
            features.append(float(value))
        
        return features
    
    def _categorize_risk(self, risk_score: float) -> Dict[str, Any]:
        """Categorize risk score into different risk categories"""
        categories = {
            'liquidity_risk': 0,
            'solvency_risk': 0,
            'profitability_risk': 0,
            'operational_risk': 0,
            'growth_risk': 0
        }
        
        # Map overall risk score to category scores
        # This is a simplified approach; a real model would have more sophisticated mapping
        if risk_score > 0.8:
            risk_level = "Critical"
            categories = {k: v + 0.8 + (np.random.rand() * 0.2) for k, v in categories.items()}
        elif risk_score > 0.6:
            risk_level = "High"
            categories = {k: v + 0.6 + (np.random.rand() * 0.2) for k, v in categories.items()}
        elif risk_score > 0.4:
            risk_level = "Medium"
            categories = {k: v + 0.4 + (np.random.rand() * 0.2) for k, v in categories.items()}
        elif risk_score > 0.2:
            risk_level = "Low"
            categories = {k: v + 0.2 + (np.random.rand() * 0.2) for k, v in categories.items()}
        else:
            risk_level = "Very Low"
            categories = {k: v + np.random.rand() * 0.2 for k, v in categories.items()}
        
        # Ensure values are between 0 and 1
        categories = {k: min(1.0, max(0.0, v)) for k, v in categories.items()}
        
        # Convert to float for JSON serialization
        categories = {k: float(v) for k, v in categories.items()}
        
        # Add risk level
        categories['overall_level'] = risk_level
        
        return categories
    
    def _generate_recommendations(self, risk_score: float, 
                                top_factors: List[Tuple[str, float]]) -> List[Dict[str, Any]]:
        """Generate recommendations based on risk score and top factors"""
        recommendations = []
        
        # Generate recommendations based on risk level
        if risk_score > 0.8:
            recommendations.append({
                "area": "Overall Risk",
                "recommendation": "Immediate action required to address critical financial risks",
                "priority": "Critical"
            })
        elif risk_score > 0.6:
            recommendations.append({
                "area": "Overall Risk",
                "recommendation": "Develop comprehensive risk mitigation plan for high financial risks",
                "priority": "High"
            })
        elif risk_score > 0.4:
            recommendations.append({
                "area": "Overall Risk",
                "recommendation": "Monitor key risk indicators and prepare contingency plans",
                "priority": "Medium"
            })
        else:
            recommendations.append({
                "area": "Overall Risk",
                "recommendation": "Continue monitoring financial metrics while focusing on growth",
                "priority": "Low"
            })
        
        # Generate specific recommendations based on top factors
        for factor, importance in top_factors:
            if 'ratio' in factor:
                recommendations.append({
                    "area": "Liquidity Management",
                    "recommendation": f"Improve {factor.replace('_', ' ')} to strengthen liquidity position",
                    "priority": "High" if importance > 0.2 else "Medium"
                })
            elif 'debt' in factor or 'interest' in factor:
                recommendations.append({
                    "area": "Debt Management",
                    "recommendation": f"Optimize {factor.replace('_', ' ')} to reduce financial leverage",
                    "priority": "High" if importance > 0.2 else "Medium"
                })
            elif 'margin' in factor or 'return' in factor:
                recommendations.append({
                    "area": "Profitability",
                    "recommendation": f"Focus on improving {factor.replace('_', ' ')} through operational efficiency",
                    "priority": "High" if importance > 0.2 else "Medium"
                })
            elif 'growth' in factor:
                recommendations.append({
                    "area": "Growth Strategy",
                    "recommendation": f"Develop strategies to enhance {factor.replace('_', ' ')}",
                    "priority": "High" if importance > 0.2 else "Medium"
                })
        
        return recommendations
    
    def _predict_financial_risks_rule_based(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict financial risks using rule-based approach (fallback method)"""
        risk_factors = []
        risk_score = 0
        risk_count = 0
        
        # Liquidity risk factors
        current_ratio = financial_data.get('current_ratio', 0)
        if current_ratio < 1.5:
            risk_severity = 'high' if current_ratio < 1.0 else 'medium'
            risk_factors.append({
                "factor": "Low Current Ratio",
                "value": current_ratio,
                "threshold": 1.5,
                "category": "Liquidity Risk",
                "severity": risk_severity
            })
            risk_score += 0.1 if risk_severity == 'medium' else 0.2
            risk_count += 1
        
        quick_ratio = financial_data.get('quick_ratio', 0)
        if quick_ratio < 1.0:
            risk_severity = 'high' if quick_ratio < 0.7 else 'medium'
            risk_factors.append({
                "factor": "Low Quick Ratio",
                "value": quick_ratio,
                "threshold": 1.0,
                "category": "Liquidity Risk",
                "severity": risk_severity
            })
            risk_score += 0.1 if risk_severity == 'medium' else 0.2
            risk_count += 1
        
        # Solvency risk factors
        debt_to_equity = financial_data.get('debt_to_equity', 0)
        if debt_to_equity > 2.0:
            risk_severity = 'high' if debt_to_equity > 3.0 else 'medium'
            risk_factors.append({
                "factor": "High Debt-to-Equity Ratio",
                "value": debt_to_equity,
                "threshold": 2.0,
                "category": "Solvency Risk",
                "severity": risk_severity
            })
            risk_score += 0.1 if risk_severity == 'medium' else 0.2
            risk_count += 1
        
        interest_coverage = financial_data.get('interest_coverage', 0)
        if interest_coverage < 3.0:
            risk_severity = 'high' if interest_coverage < 1.5 else 'medium'
            risk_factors.append({
                "factor": "Low Interest Coverage Ratio",
                "value": interest_coverage,
                "threshold": 3.0,
                "category": "Solvency Risk",
                "severity": risk_severity
            })
            risk_score += 0.1 if risk_severity == 'medium' else 0.2
            risk_count += 1
        
        # Profitability risk factors
        net_profit_margin = financial_data.get('net_profit_margin', 0)
        industry_avg_npm = financial_data.get('industry_avg_net_profit_margin', 10)
        if net_profit_margin < industry_avg_npm * 0.7:
            risk_severity = 'high' if net_profit_margin < industry_avg_npm * 0.4 else 'medium'
            risk_factors.append({
                "factor": "Below Average Net Profit Margin",
                "value": net_profit_margin,
                "threshold": industry_avg_npm,
                "category": "Profitability Risk",
                "severity": risk_severity
            })
            risk_score += 0.1 if risk_severity == 'medium' else 0.2
            risk_count += 1
        
        return_on_equity = financial_data.get('return_on_equity', 0)
        industry_avg_roe = financial_data.get('industry_avg_return_on_equity', 15)
        if return_on_equity < industry_avg_roe * 0.7:
            risk_severity = 'high' if return_on_equity < industry_avg_roe * 0.4 else 'medium'
            risk_factors.append({
                "factor": "Below Average Return on Equity",
                "value": return_on_equity,
                "threshold": industry_avg_roe,
                "category": "Profitability Risk",
                "severity": risk_severity
            })
            risk_score += 0.1 if risk_severity == 'medium' else 0.2
            risk_count += 1
        
        # Growth risk factors
        revenue_growth = financial_data.get('revenue_growth', 0)
        industry_avg_growth = financial_data.get('industry_avg_revenue_growth', 5)
        if revenue_growth < industry_avg_growth * 0.5:
            risk_severity = 'high' if revenue_growth < 0 else 'medium'
            risk_factors.append({
                "factor": "Below Average Revenue Growth",
                "value": revenue_growth,
                "threshold": industry_avg_growth,
                "category": "Growth Risk",
                "severity": risk_severity
            })
            risk_score += 0.1 if risk_severity == 'medium' else 0.2
            risk_count += 1
        
        # Normalize risk score
        if risk_count > 0:
            normalized_risk_score = min(1.0, risk_score)
        else:
            normalized_risk_score = 0.1  # Minimal risk if no risk factors identified
        
        # Generate recommendations
        recommendations = self._generate_rule_based_recommendations(risk_factors)
        
        # Categorize risks
        risk_categories = self._categorize_rule_based_risk(risk_factors, normalized_risk_score)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_risk_score': float(normalized_risk_score),
            'risk_categories': risk_categories,
            'risk_factors': risk_factors,
            'recommendations': recommendations
        }
    
    def _categorize_rule_based_risk(self, risk_factors: List[Dict[str, Any]], 
                                   overall_score: float) -> Dict[str, Any]:
        """Categorize risks based on identified risk factors"""
        categories = {
            'liquidity_risk': 0.0,
            'solvency_risk': 0.0,
            'profitability_risk': 0.0,
            'operational_risk': 0.0,
            'growth_risk': 0.0
        }
        
        # Count risk factors by category
        category_counts = {}
        for factor in risk_factors:
            category = factor['category'].lower().replace(' ', '_')
            if category in categories:
                if factor['severity'] == 'high':
                    categories[category] += 0.2
                else:
                    categories[category] += 0.1
                    
                category_counts[category] = category_counts.get(category, 0) + 1
        
        # Normalize category scores
        for category in categories:
            if category in category_counts and category_counts[category] > 0:
                # Cap at 1.0
                categories[category] = min(1.0, categories[category])
            else:
                # Set minimal risk for categories with no identified factors
                categories[category] = 0.1
        
        # Determine overall risk level
        if overall_score >= 0.8:
            risk_level = "Critical"
        elif overall_score >= 0.6:
            risk_level = "High"
        elif overall_score >= 0.4:
            risk_level = "Medium"
        elif overall_score >= 0.2:
            risk_level = "Low"
        else:
            risk_level = "Very Low"
            
        categories['overall_level'] = risk_level
        
        return categories
    
    def _generate_rule_based_recommendations(self, risk_factors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate recommendations based on identified risk factors"""
        recommendations = []
        
        # Group risk factors by category
        category_factors = {}
        for factor in risk_factors:
            category = factor['category']
            if category not in category_factors:
                category_factors[category] = []
            category_factors[category].append(factor)
        
        # Generate overall recommendation based on high severity factors
        high_severity_count = sum(1 for factor in risk_factors if factor['severity'] == 'high')
        if high_severity_count >= 3:
            recommendations.append({
                "area": "Overall Financial Health",
                "recommendation": "Develop comprehensive financial restructuring plan to address multiple high-severity risks",
                "priority": "Critical"
            })
        elif high_severity_count > 0:
            recommendations.append({
                "area": "Overall Financial Health",
                "recommendation": "Create targeted action plans for each high-severity risk area",
                "priority": "High"
            })
        elif risk_factors:
            recommendations.append({
                "area": "Overall Financial Health",
                "recommendation": "Monitor identified risk areas while maintaining current financial strategies",
                "priority": "Medium"
            })
        else:
            recommendations.append({
                "area": "Overall Financial Health",
                "recommendation": "Continue current financial management practices while seeking growth opportunities",
                "priority": "Low"
            })
        
        # Generate category-specific recommendations
        if 'Liquidity Risk' in category_factors:
            liquidity_factors = category_factors['Liquidity Risk']
            if any(factor['severity'] == 'high' for factor in liquidity_factors):
                recommendations.append({
                    "area": "Liquidity Management",
                    "recommendation": "Improve short-term liquidity position through working capital optimization",
                    "priority": "High"
                })
            else:
                recommendations.append({
                    "area": "Liquidity Management",
                    "recommendation": "Monitor liquidity metrics and maintain adequate cash reserves",
                    "priority": "Medium"
                })
        
        if 'Solvency Risk' in category_factors:
            solvency_factors = category_factors['Solvency Risk']
            if any(factor['severity'] == 'high' for factor in solvency_factors):
                recommendations.append({
                    "area": "Debt Management",
                    "recommendation": "Reduce debt levels and improve capital structure to enhance financial stability",
                    "priority": "High"
                })
            else:
                recommendations.append({
                    "area": "Debt Management",
                    "recommendation": "Review debt structure and interest expense to optimize cost of capital",
                    "priority": "Medium"
                })
        
        if 'Profitability Risk' in category_factors:
            profitability_factors = category_factors['Profitability Risk']
            if any(factor['severity'] == 'high' for factor in profitability_factors):
                recommendations.append({
                    "area": "Profitability Improvement",
                    "recommendation": "Implement comprehensive cost reduction and margin enhancement initiatives",
                    "priority": "High"
                })
            else:
                recommendations.append({
                    "area": "Profitability Improvement",
                    "recommendation": "Identify opportunities to improve operational efficiency and pricing strategies",
                    "priority": "Medium"
                })
        
        if 'Growth Risk' in category_factors:
            growth_factors = category_factors['Growth Risk']
            if any(factor['severity'] == 'high' for factor in growth_factors):
                recommendations.append({
                    "area": "Growth Strategy",
                    "recommendation": "Develop new growth initiatives and market expansion strategies",
                    "priority": "High"
                })
            else:
                recommendations.append({
                    "area": "Growth Strategy",
                    "recommendation": "Evaluate current growth strategy and identify enhancement opportunities",
                    "priority": "Medium"
                })
        
        return recommendations
    
    def predict_market_risks(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict market risks based on market data
        
        Args:
            market_data: Dictionary containing market metrics
            
        Returns:
            Dictionary containing risk predictions
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'overall_risk_score': 0,
            'risk_categories': {},
            'risk_factors': [],
            'recommendations': []
        }
        
        # Use rule-based approach for market risk prediction
        return self._predict_market_risks_rule_based(market_data)
    
    def _predict_market_risks_rule_based(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict market risks using rule-based approach"""
        risk_factors = []
        risk_score = 0
        risk_count = 0
        
        # Competitive intensity risk
        competitor_count = market_data.get('competitor_count', 0)
        market_concentration = market_data.get('market_concentration', 0)  # HHI index (0-10000)
        
        if competitor_count > 20 and market_concentration < 1500:
            risk_severity = 'high'
            risk_factors.append({
                "factor": "High Competitive Intensity",
                "value": competitor_count,
                "threshold": 20,
                "category": "Competitive Risk",
                "severity": risk_severity
            })
            risk_score += 0.2
            risk_count += 1
        elif competitor_count > 10:
            risk_severity = 'medium'
            risk_factors.append({
                "factor": "Moderate Competitive Intensity",
                "value": competitor_count,
                "threshold": 10,
                "category": "Competitive Risk",
                "severity": risk_severity
            })
            risk_score += 0.1
            risk_count += 1
        
        # Market share risk
        market_share = market_data.get('market_share', 0)
        market_leader_share = market_data.get('market_leader_share', 0)
        
        if market_share < 5 and market_leader_share > 30:
            risk_severity = 'high'
            risk_factors.append({
                "factor": "Low Market Share in Concentrated Market",
                "value": market_share,
                "threshold": 5,
                "category": "Market Position Risk",
                "severity": risk_severity
            })
            risk_score += 0.2
            risk_count += 1
        elif market_share < 10:
            risk_severity = 'medium'
            risk_factors.append({
                "factor": "Low Market Share",
                "value": market_share,
                "threshold": 10,
                "category": "Market Position Risk",
                "severity": risk_severity
            })
            risk_score += 0.1
            risk_count += 1
        
        # Market growth risk
        market_growth = market_data.get('market_growth', 0)
        
        if market_growth < 0:
            risk_severity = 'high'
            risk_factors.append({
                "factor": "Declining Market",
                "value": market_growth,
                "threshold": 0,
                "category": "Market Growth Risk",
                "severity": risk_severity
            })
            risk_score += 0.2
            risk_count += 1
        elif market_growth < 2:
            risk_severity = 'medium'
            risk_factors.append({
                "factor": "Slow Market Growth",
                "value": market_growth,
                "threshold": 2,
                "category": "Market Growth Risk",
                "severity": risk_severity
            })
            risk_score += 0.1
            risk_count += 1
        
        # Disruption risk
        disruption_score = market_data.get('disruption_score', 0)
        
        if disruption_score > 7:
            risk_severity = 'high'
            risk_factors.append({
                "factor": "High Disruption Potential",
                "value": disruption_score,
                "threshold": 7,
                "category": "Disruption Risk",
                "severity": risk_severity
            })
            risk_score += 0.2
            risk_count += 1
        elif disruption_score > 5:
            risk_severity = 'medium'
            risk_factors.append({
                "factor": "Moderate Disruption Potential",
                "value": disruption_score,
                "threshold": 5,
                "category": "Disruption Risk",
                "severity": risk_severity
            })
            risk_score += 0.1
            risk_count += 1
        
        # Customer concentration risk
        top_customer_percentage = market_data.get('top_customer_percentage', 0)
        top_5_customers_percentage = market_data.get('top_5_customers_percentage', 0)
        
        if top_customer_percentage > 20:
            risk_severity = 'high'
            risk_factors.append({
                "factor": "High Customer Concentration",
                "value": top_customer_percentage,
                "threshold": 20,
                "category": "Customer Concentration Risk",
                "severity": risk_severity
            })
            risk_score += 0.2
            risk_count += 1
        elif top_5_customers_percentage > 50:
            risk_severity = 'medium'
            risk_factors.append({
                "factor": "Moderate Customer Concentration",
                "value": top_5_customers_percentage,
                "threshold": 50,
                "category": "Customer Concentration Risk",
                "severity": risk_severity
            })
            risk_score += 0.1
            risk_count += 1
        
        # Normalize risk score
        if risk_count > 0:
            normalized_risk_score = min(1.0, risk_score)
        else:
            normalized_risk_score = 0.1  # Minimal risk if no risk factors identified
        
        # Generate recommendations
        recommendations = []
        
        # Overall recommendation
        high_severity_count = sum(1 for factor in risk_factors if factor['severity'] == 'high')
        if high_severity_count >= 2:
            recommendations.append({
                "area": "Market Strategy",
                "recommendation": "Develop comprehensive market repositioning strategy to address multiple high-risk areas",
                "priority": "Critical"
            })
        elif high_severity_count > 0:
            recommendations.append({
                "area": "Market Strategy",
                "recommendation": "Create targeted action plans for high-risk market factors",
                "priority": "High"
            })
        elif risk_factors:
            recommendations.append({
                "area": "Market Strategy",
                "recommendation": "Monitor identified market risks while maintaining current strategy",
                "priority": "Medium"
            })
        else:
            recommendations.append({
                "area": "Market Strategy",
                "recommendation": "Focus on growth opportunities in favorable market conditions",
                "priority": "Low"
            })
        
        # Specific recommendations
        for factor in risk_factors:
            if factor['category'] == 'Competitive Risk':
                if factor['severity'] == 'high':
                    recommendations.append({
                        "area": "Competitive Positioning",
                        "recommendation": "Develop clear differentiation strategy to stand out in highly competitive market",
                        "priority": "High"
                    })
                else:
                    recommendations.append({
                        "area": "Competitive Positioning",
                        "recommendation": "Monitor competitive landscape and strengthen value proposition",
                        "priority": "Medium"
                    })
            elif factor['category'] == 'Market Position Risk':
                if factor['severity'] == 'high':
                    recommendations.append({
                        "area": "Market Share",
                        "recommendation": "Identify niche segments or underserved markets to increase market share",
                        "priority": "High"
                    })
                else:
                    recommendations.append({
                        "area": "Market Share",
                        "recommendation": "Develop customer acquisition and retention strategies to gradually increase market share",
                        "priority": "Medium"
                    })
            elif factor['category'] == 'Market Growth Risk':
                if factor['severity'] == 'high':
                    recommendations.append({
                        "area": "Market Expansion",
                        "recommendation": "Explore adjacent markets or new geographies to compensate for declining core market",
                        "priority": "High"
                    })
                else:
                    recommendations.append({
                        "area": "Market Expansion",
                        "recommendation": "Diversify product/service offerings to tap into higher-growth segments",
                        "priority": "Medium"
                    })
            elif factor['category'] == 'Disruption Risk':
                if factor['severity'] == 'high':
                    recommendations.append({
                        "area": "Innovation Strategy",
                        "recommendation": "Develop disruptive innovation initiatives to stay ahead of market changes",
                        "priority": "High"
                    })
                else:
                    recommendations.append({
                        "area": "Innovation Strategy",
                        "recommendation": "Increase R&D investment and monitor emerging technologies",
                        "priority": "Medium"
                    })
            elif factor['category'] == 'Customer Concentration Risk':
                if factor['severity'] == 'high':
                    recommendations.append({
                        "area": "Customer Diversification",
                        "recommendation": "Implement customer acquisition strategy to reduce dependence on top customers",
                        "priority": "High"
                    })
                else:
                    recommendations.append({
                        "area": "Customer Diversification",
                        "recommendation": "Develop strategies to deepen relationships with existing customers while expanding customer base",
                        "priority": "Medium"
                    })
        
        # Categorize risks
        risk_categories = {
            'competitive_risk': 0.1,
            'market_position_risk': 0.1,
            'market_growth_risk': 0.1,
            'disruption_risk': 0.1,
            'customer_concentration_risk': 0.1
        }
        
        for factor in risk_factors:
            category = factor['category'].lower().replace(' ', '_')
            if category in risk_categories:
                if factor['severity'] == 'high':
                    risk_categories[category] = 0.8
                else:
                    risk_categories[category] = 0.5
        
        # Determine overall risk level
        if normalized_risk_score >= 0.8:
            risk_level = "Critical"
        elif normalized_risk_score >= 0.6:
            risk_level = "High"
        elif normalized_risk_score >= 0.4:
            risk_level = "Medium"
        elif normalized_risk_score >= 0.2:
            risk_level = "Low"
        else:
            risk_level = "Very Low"
            
        risk_categories['overall_level'] = risk_level
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_risk_score': float(normalized_risk_score),
            'risk_categories': risk_categories,
            'risk_factors': risk_factors,
            'recommendations': recommendations
        }
    def monte_carlo_simulation(self, data: Dict[str, Any], 
                              scenarios: int = 1000) -> Dict[str, Any]:
        """
        Perform Monte Carlo simulation to estimate risk distribution
        
        Args:
            data: Dictionary containing input data for simulation
            scenarios: Number of scenarios to simulate
            
        Returns:
            Dictionary containing simulation results
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'simulation_parameters': {
                'scenarios': scenarios
            },
            'simulation_results': {},
            'risk_metrics': {},
            'recommendations': []
        }
        
        try:
            # Extract relevant parameters
            revenue_mean = data.get('revenue', 1000000)
            revenue_std = data.get('revenue_std', revenue_mean * 0.2)
            
            cost_mean = data.get('cost', revenue_mean * 0.7)
            cost_std = data.get('cost_std', cost_mean * 0.15)
            
            interest_rate_mean = data.get('interest_rate', 0.05)
            interest_rate_std = data.get('interest_rate_std', 0.01)
            
            debt = data.get('debt', revenue_mean * 0.5)
            
            tax_rate = data.get('tax_rate', 0.25)
            
            # Run simulation
            np.random.seed(42)  # For reproducibility
            
            revenues = np.random.normal(revenue_mean, revenue_std, scenarios)
            costs = np.random.normal(cost_mean, cost_std, scenarios)
            interest_rates = np.random.normal(interest_rate_mean, interest_rate_std, scenarios)
            
            # Calculate key metrics
            ebitda = revenues - costs
            interest_expense = debt * interest_rates
            ebit = ebitda - interest_expense
            taxes = np.maximum(0, ebit * tax_rate)  # No tax benefit from losses
            net_income = ebit - taxes
            
            # Calculate risk metrics
            loss_probability = np.mean(net_income < 0) * 100
            expected_net_income = np.mean(net_income)
            net_income_std = np.std(net_income)
            
            # Value at Risk (95%)
            var_95 = np.percentile(net_income, 5)
            
            # Expected Shortfall (95%)
            es_95 = np.mean(net_income[net_income <= var_95])
            
            # Store simulation results
            results['simulation_results'] = {
                'revenue': {
                    'mean': float(np.mean(revenues)),
                    'std': float(np.std(revenues)),
                    'min': float(np.min(revenues)),
                    'max': float(np.max(revenues)),
                    'distribution': self._create_distribution_data(revenues)
                },
                'costs': {
                    'mean': float(np.mean(costs)),
                    'std': float(np.std(costs)),
                    'min': float(np.min(costs)),
                    'max': float(np.max(costs)),
                    'distribution': self._create_distribution_data(costs)
                },
                'ebitda': {
                    'mean': float(np.mean(ebitda)),
                    'std': float(np.std(ebitda)),
                    'min': float(np.min(ebitda)),
                    'max': float(np.max(ebitda)),
                    'distribution': self._create_distribution_data(ebitda)
                },
                'net_income': {
                    'mean': float(np.mean(net_income)),
                    'std': float(np.std(net_income)),
                    'min': float(np.min(net_income)),
                    'max': float(np.max(net_income)),
                    'distribution': self._create_distribution_data(net_income)
                }
            }
            
            # Store risk metrics
            results['risk_metrics'] = {
                'loss_probability': float(loss_probability),
                'expected_net_income': float(expected_net_income),
                'net_income_volatility': float(net_income_std),
                'value_at_risk_95': float(var_95),
                'expected_shortfall_95': float(es_95),
                'debt_coverage_ratio': float(np.mean(ebitda) / np.mean(interest_expense)) if np.mean(interest_expense) > 0 else float('inf'),
                'coefficient_of_variation': float(net_income_std / expected_net_income) if expected_net_income > 0 else float('inf')
            }
            
            # Generate recommendations based on risk metrics
            recommendations = []
            
            if loss_probability > 30:
                recommendations.append({
                    "area": "Profitability Risk",
                    "recommendation": "High probability of loss requires immediate action to reduce costs or increase revenue",
                    "priority": "Critical"
                })
            elif loss_probability > 15:
                recommendations.append({
                    "area": "Profitability Risk",
                    "recommendation": "Significant probability of loss requires attention to cost structure and pricing strategy",
                    "priority": "High"
                })
            elif loss_probability > 5:
                recommendations.append({
                    "area": "Profitability Risk",
                    "recommendation": "Moderate probability of loss suggests need for contingency planning",
                    "priority": "Medium"
                })
            
            if abs(var_95) > expected_net_income * 2:
                recommendations.append({
                    "area": "Downside Risk",
                    "recommendation": "Extreme downside risk indicates need for risk mitigation strategies",
                    "priority": "High"
                })
            elif abs(var_95) > expected_net_income:
                recommendations.append({
                    "area": "Downside Risk",
                    "recommendation": "Significant downside risk suggests need for more conservative financial planning",
                    "priority": "Medium"
                })
            
            debt_coverage = results['risk_metrics']['debt_coverage_ratio']
            if debt_coverage < 1.5:
                recommendations.append({
                    "area": "Debt Service Risk",
                    "recommendation": "Low debt coverage ratio indicates potential difficulty meeting debt obligations",
                    "priority": "High"
                })
            elif debt_coverage < 3:
                recommendations.append({
                    "area": "Debt Service Risk",
                    "recommendation": "Moderate debt coverage ratio suggests monitoring debt levels and interest expense",
                    "priority": "Medium"
                })
            
            results['recommendations'] = recommendations
            
            return results
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {str(e)}")
            results['error'] = str(e)
            return results
    
    def _create_distribution_data(self, data: np.ndarray) -> Dict[str, List[float]]:
        """Create distribution data for visualization"""
        hist, bin_edges = np.histogram(data, bins=20, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        return {
            'x': bin_centers.tolist(),
            'y': hist.tolist(),
            'percentiles': {
                '5': float(np.percentile(data, 5)),
                '25': float(np.percentile(data, 25)),
                '50': float(np.percentile(data, 50)),
                '75': float(np.percentile(data, 75)),
                '95': float(np.percentile(data, 95))
            }
        }
    
    def predict_operational_risks(self, operational_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict operational risks based on operational data
        
        Args:
            operational_data: Dictionary containing operational metrics
            
        Returns:
            Dictionary containing risk predictions
        """
        # Use rule-based approach for operational risk prediction
        return self._predict_operational_risks_rule_based(operational_data)
    
    def _predict_operational_risks_rule_based(self, operational_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict operational risks using rule-based approach"""
        risk_factors = []
        risk_score = 0
        risk_count = 0
        
        # Supply chain risk
        supplier_concentration = operational_data.get('supplier_concentration', 0)  # % from top supplier
        supplier_count = operational_data.get('supplier_count', 0)
        
        if supplier_concentration > 30:
            risk_severity = 'high' if supplier_concentration > 50 else 'medium'
            risk_factors.append({
                "factor": "High Supplier Concentration",
                "value": supplier_concentration,
                "threshold": 30,
                "category": "Supply Chain Risk",
                "severity": risk_severity
            })
            risk_score += 0.2 if risk_severity == 'high' else 0.1
            risk_count += 1
        
        if supplier_count < 5:
            risk_severity = 'high' if supplier_count < 3 else 'medium'
            risk_factors.append({
                "factor": "Low Supplier Diversity",
                "value": supplier_count,
                "threshold": 5,
                "category": "Supply Chain Risk",
                "severity": risk_severity
            })
            risk_score += 0.2 if risk_severity == 'high' else 0.1
            risk_count += 1
        
        # Production risk
        capacity_utilization = operational_data.get('capacity_utilization', 0)  # percentage
        equipment_age = operational_data.get('equipment_age', 0)  # years
        
        if capacity_utilization > 90:
            risk_severity = 'high' if capacity_utilization > 95 else 'medium'
            risk_factors.append({
                "factor": "High Capacity Utilization",
                "value": capacity_utilization,
                "threshold": 90,
                "category": "Production Risk",
                "severity": risk_severity
            })
            risk_score += 0.2 if risk_severity == 'high' else 0.1
            risk_count += 1
        
        if equipment_age > 7:
            risk_severity = 'high' if equipment_age > 10 else 'medium'
            risk_factors.append({
                "factor": "Aging Equipment",
                "value": equipment_age,
                "threshold": 7,
                "category": "Production Risk",
                "severity": risk_severity
            })
            risk_score += 0.2 if risk_severity == 'high' else 0.1
            risk_count += 1
        
        # Human resources risk
        employee_turnover = operational_data.get('employee_turnover', 0)  # percentage
        key_position_vacancy = operational_data.get('key_position_vacancy', 0)  # percentage
        
        if employee_turnover > 15:
            risk_severity = 'high' if employee_turnover > 25 else 'medium'
            risk_factors.append({
                "factor": "High Employee Turnover",
                "value": employee_turnover,
                "threshold": 15,
                "category": "Human Resources Risk",
                "severity": risk_severity
            })
            risk_score += 0.2 if risk_severity == 'high' else 0.1
            risk_count += 1
        
        if key_position_vacancy > 5:
            risk_severity = 'high' if key_position_vacancy > 10 else 'medium'
            risk_factors.append({
                "factor": "Key Position Vacancies",
                "value": key_position_vacancy,
                "threshold": 5,
                "category": "Human Resources Risk",
                "severity": risk_severity
            })
            risk_score += 0.2 if risk_severity == 'high' else 0.1
            risk_count += 1
        
        # IT risk
        system_downtime = operational_data.get('system_downtime', 0)  # hours per month
        security_incidents = operational_data.get('security_incidents', 0)  # past 12 months
        
        if system_downtime > 4:
            risk_severity = 'high' if system_downtime > 8 else 'medium'
            risk_factors.append({
                "factor": "High System Downtime",
                "value": system_downtime,
                "threshold": 4,
                "category": "IT Risk",
                "severity": risk_severity
            })
            risk_score += 0.2 if risk_severity == 'high' else 0.1
            risk_count += 1
        
        if security_incidents > 1:
            risk_severity = 'high' if security_incidents > 3 else 'medium'
            risk_factors.append({
                "factor": "Security Incidents",
                "value": security_incidents,
                "threshold": 1,
                "category": "IT Risk",
                "severity": risk_severity
            })
            risk_score += 0.2 if risk_severity == 'high' else 0.1
            risk_count += 1
        
        # Compliance risk
        compliance_issues = operational_data.get('compliance_issues', 0)  # past 12 months
        pending_litigation = operational_data.get('pending_litigation', 0)  # number of cases
        
        if compliance_issues > 0:
            risk_severity = 'high' if compliance_issues > 2 else 'medium'
            risk_factors.append({
                "factor": "Compliance Issues",
                "value": compliance_issues,
                "threshold": 0,
                "category": "Compliance Risk",
                "severity": risk_severity
            })
            risk_score += 0.2 if risk_severity == 'high' else 0.1
            risk_count += 1
        
        if pending_litigation > 0:
            risk_severity = 'high' if pending_litigation > 2 else 'medium'
            risk_factors.append({
                "factor": "Pending Litigation",
                "value": pending_litigation,
                "threshold": 0,
                "category": "Compliance Risk",
                "severity": risk_severity
            })
            risk_score += 0.2 if risk_severity == 'high' else 0.1
            risk_count += 1
        
        # Normalize risk score
        if risk_count > 0:
            normalized_risk_score = min(1.0, risk_score)
        else:
            normalized_risk_score = 0.1  # Minimal risk if no risk factors identified
        
        # Generate recommendations
        recommendations = []
        
        # Overall recommendation
        high_severity_count = sum(1 for factor in risk_factors if factor['severity'] == 'high')
        if high_severity_count >= 3:
            recommendations.append({
                "area": "Operational Risk Management",
                "recommendation": "Develop comprehensive operational risk mitigation plan to address multiple high-severity risks",
                "priority": "Critical"
            })
        elif high_severity_count > 0:
            recommendations.append({
                "area": "Operational Risk Management",
                "recommendation": "Create targeted action plans for each high-severity operational risk area",
                "priority": "High"
            })
        elif risk_factors:
            recommendations.append({
                "area": "Operational Risk Management",
                "recommendation": "Monitor identified operational risk areas while maintaining current controls",
                "priority": "Medium"
            })
        else:
            recommendations.append({
                "area": "Operational Risk Management",
                "recommendation": "Continue current operational risk management practices and focus on efficiency improvements",
                "priority": "Low"
            })
        
        # Specific recommendations
        for factor in risk_factors:
            if factor['category'] == 'Supply Chain Risk':
                if 'Supplier Concentration' in factor['factor']:
                    recommendations.append({
                        "area": "Supply Chain",
                        "recommendation": "Diversify supplier base to reduce concentration risk",
                        "priority": "High" if factor['severity'] == 'high' else "Medium"
                    })
                elif 'Supplier Diversity' in factor['factor']:
                    recommendations.append({
                        "area": "Supply Chain",
                        "recommendation": "Identify and onboard additional qualified suppliers",
                        "priority": "High" if factor['severity'] == 'high' else "Medium"
                    })
            elif factor['category'] == 'Production Risk':
                if 'Capacity Utilization' in factor['factor']:
                    recommendations.append({
                        "area": "Production",
                        "recommendation": "Evaluate capacity expansion or load balancing to reduce production constraints",
                        "priority": "High" if factor['severity'] == 'high' else "Medium"
                    })
                elif 'Aging Equipment' in factor['factor']:
                    recommendations.append({
                        "area": "Production",
                        "recommendation": "Develop equipment replacement and maintenance plan",
                        "priority": "High" if factor['severity'] == 'high' else "Medium"
                    })
            elif factor['category'] == 'Human Resources Risk':
                if 'Employee Turnover' in factor['factor']:
                    recommendations.append({
                        "area": "Human Resources",
                        "recommendation": "Implement employee retention initiatives and conduct exit interviews to identify turnover causes",
                        "priority": "High" if factor['severity'] == 'high' else "Medium"
                    })
                elif 'Key Position Vacancies' in factor['factor']:
                    recommendations.append({
                        "area": "Human Resources",
                        "recommendation": "Develop succession planning and accelerate recruitment for key positions",
                        "priority": "High" if factor['severity'] == 'high' else "Medium"
                    })
            elif factor['category'] == 'IT Risk':
                if 'System Downtime' in factor['factor']:
                    recommendations.append({
                        "area": "IT",
                        "recommendation": "Improve system reliability through infrastructure upgrades and redundancy",
                        "priority": "High" if factor['severity'] == 'high' else "Medium"
                    })
                elif 'Security Incidents' in factor['factor']:
                    recommendations.append({
                        "area": "IT",
                        "recommendation": "Enhance cybersecurity measures and conduct security awareness training",
                        "priority": "High" if factor['severity'] == 'high' else "Medium"
                    })
            elif factor['category'] == 'Compliance Risk':
                if 'Compliance Issues' in factor['factor'] or 'Pending Litigation' in factor['factor']:
                    recommendations.append({
                        "area": "Compliance",
                        "recommendation": "Strengthen compliance program and conduct thorough legal risk assessment",
                        "priority": "High" if factor['severity'] == 'high' else "Medium"
                    })
        
        # Categorize risks
        risk_categories = {
            'supply_chain_risk': 0.1,
            'production_risk': 0.1,
            'human_resources_risk': 0.1,
            'it_risk': 0.1,
            'compliance_risk': 0.1
        }
        
        for factor in risk_factors:
            category = factor['category'].lower().replace(' ', '_')
            if category in risk_categories:
                if factor['severity'] == 'high':
                    risk_categories[category] = 0.8
                else:
                    risk_categories[category] = 0.5
        
        # Determine overall risk level
        if normalized_risk_score >= 0.8:
            risk_level = "Critical"
        elif normalized_risk_score >= 0.6:
            risk_level = "High"
        elif normalized_risk_score >= 0.4:
            risk_level = "Medium"
        elif normalized_risk_score >= 0.2:
            risk_level = "Low"
        else:
            risk_level = "Very Low"
            
        risk_categories['overall_level'] = risk_level
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_risk_score': float(normalized_risk_score),
            'risk_categories': risk_categories,
            'risk_factors': risk_factors,
            'recommendations': recommendations
        }