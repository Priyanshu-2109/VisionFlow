# app/modules/recommender/growth_strategies.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class GrowthStrategies:
    """
    Generate tailored growth strategies for businesses
    """
    
    def __init__(self):
        self.strategy_database = self._load_strategy_database()
    
    def _load_strategy_database(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load growth strategy database"""
        # In a real implementation, this would load from a database or file
        # Using hardcoded values for demonstration
        
        return {
            'market_penetration': [
                {
                    'name': 'Enhanced Marketing',
                    'description': 'Increase market share through improved marketing effectiveness',
                    'actions': [
                        'Conduct customer segmentation analysis',
                        'Develop targeted messaging for key segments',
                        'Optimize marketing channel mix',
                        'Implement marketing performance metrics'
                    ],
                    'suitable_for': ['small', 'medium', 'large'],
                    'resource_requirements': 'Medium',
                    'timeframe': 'Short-term',
                    'expected_impact': 'Medium',
                    'risk_level': 'Low'
                },
                {
                    'name': 'Sales Effectiveness',
                    'description': 'Improve sales conversion and effectiveness',
                    'actions': [
                        'Enhance sales training and enablement',
                        'Implement sales methodology and process',
                        'Develop competitive battlecards',
                        'Create sales incentive structure'
                    ],
                    'suitable_for': ['small', 'medium', 'large'],
                    'resource_requirements': 'Medium',
                    'timeframe': 'Short-term',
                    'expected_impact': 'High',
                    'risk_level': 'Low'
                },
                {
                    'name': 'Pricing Optimization',
                    'description': 'Optimize pricing strategy to increase market share and revenue',
                    'actions': [
                        'Conduct pricing analysis and research',
                        'Develop value-based pricing strategy',
                        'Implement price testing methodology',
                        'Create pricing governance process'
                    ],
                    'suitable_for': ['medium', 'large'],
                    'resource_requirements': 'Medium',
                    'timeframe': 'Medium-term',
                    'expected_impact': 'High',
                    'risk_level': 'Medium'
                },
                {
                    'name': 'Customer Experience Enhancement',
                    'description': 'Improve customer experience to increase retention and referrals',
                    'actions': [
                        'Map customer journey and identify pain points',
                        'Implement experience improvements',
                        'Develop customer feedback mechanisms',
                        'Create customer success program'
                    ],
                    'suitable_for': ['small', 'medium', 'large'],
                    'resource_requirements': 'Medium',
                    'timeframe': 'Medium-term',
                    'expected_impact': 'High',
                    'risk_level': 'Low'
                }
            ],
            'market_development': [
                {
                    'name': 'Geographic Expansion',
                    'description': 'Enter new geographic markets with existing products/services',
                    'actions': [
                        'Conduct market assessment of target regions',
                        'Develop region-specific go-to-market strategy',
                        'Establish local partnerships or presence',
                        'Adapt offerings to regional requirements'
                    ],
                    'suitable_for': ['medium', 'large'],
                    'resource_requirements': 'High',
                    'timeframe': 'Long-term',
                    'expected_impact': 'High',
                    'risk_level': 'High'
                },
                {
                    'name': 'New Customer Segments',
                    'description': 'Target new customer segments with existing products/services',
                    'actions': [
                        'Identify and assess potential customer segments',
                        'Adapt value proposition for new segments',
                        'Develop segment-specific marketing',
                        'Create sales enablement for new segments'
                    ],
                    'suitable_for': ['small', 'medium', 'large'],
                    'resource_requirements': 'Medium',
                    'timeframe': 'Medium-term',
                    'expected_impact': 'Medium',
                    'risk_level': 'Medium'
                },
                {
                    'name': 'Channel Expansion',
                    'description': 'Develop new sales and distribution channels',
                    'actions': [
                        'Evaluate potential new channels',
                        'Develop channel strategy and program',
                        'Create channel enablement resources',
                        'Implement channel management process'
                    ],
                    'suitable_for': ['small', 'medium', 'large'],
                    'resource_requirements': 'Medium',
                    'timeframe': 'Medium-term',
                    'expected_impact': 'Medium',
                    'risk_level': 'Medium'
                }
            ],
            'product_development': [
                {
                    'name': 'New Product Introduction',
                    'description': 'Develop and launch new products for existing markets',
                    'actions': [
                        'Conduct customer needs assessment',
                        'Develop product concept and roadmap',
                        'Implement agile development process',
                        'Create go-to-market launch plan'
                    ],
                    'suitable_for': ['small', 'medium', 'large'],
                    'resource_requirements': 'High',
                    'timeframe': 'Long-term',
                    'expected_impact': 'High',
                    'risk_level': 'High'
                },
                {
                    'name': 'Product Line Extension',
                    'description': 'Extend existing product lines with variants or additions',
                    'actions': [
                        'Identify product line extension opportunities',
                        'Develop extension strategy and roadmap',
                        'Create extension development plan',
                        'Implement marketing strategy for extensions'
                    ],
                    'suitable_for': ['small', 'medium', 'large'],
                    'resource_requirements': 'Medium',
                    'timeframe': 'Medium-term',
                    'expected_impact': 'Medium',
                    'risk_level': 'Medium'
                },
                {
                    'name': 'Product Enhancement',
                    'description': 'Improve existing products with new features or capabilities',
                    'actions': [
                        'Gather customer feedback on improvement opportunities',
                        'Prioritize enhancements based on impact',
                        'Develop enhancement roadmap',
                        'Implement enhancement development process'
                    ],
                    'suitable_for': ['small', 'medium', 'large'],
                    'resource_requirements': 'Medium',
                    'timeframe': 'Short-term',
                    'expected_impact': 'Medium',
                    'risk_level': 'Low'
                }
            ],
            'diversification': [
                {
                    'name': 'Related Diversification',
                    'description': 'Enter new markets with new products related to current business',
                    'actions': [
                        'Identify related market opportunities',
                        'Assess capability fit and gaps',
                        'Develop market entry strategy',
                        'Create new business unit or structure'
                    ],
                    'suitable_for': ['medium', 'large'],
                    'resource_requirements': 'High',
                    'timeframe': 'Long-term',
                    'expected_impact': 'High',
                    'risk_level': 'High'
                },
                {
                    'name': 'Unrelated Diversification',
                    'description': 'Enter entirely new markets unrelated to current business',
                    'actions': [
                        'Identify high-potential market opportunities',
                        'Conduct thorough market and capability assessment',
                        'Develop acquisition or new venture strategy',
                        'Create separate business entity'
                    ],
                    'suitable_for': ['large'],
                    'resource_requirements': 'Very High',
                    'timeframe': 'Long-term',
                    'expected_impact': 'High',
                    'risk_level': 'Very High'
                },
                {
                    'name': 'Strategic Acquisition',
                    'description': 'Acquire companies to enter new markets or add capabilities',
                    'actions': [
                        'Develop acquisition strategy and criteria',
                        'Identify and evaluate acquisition targets',
                        'Conduct due diligence process',
                        'Create post-acquisition integration plan'
                    ],
                    'suitable_for': ['medium', 'large'],
                    'resource_requirements': 'Very High',
                    'timeframe': 'Medium-term',
                    'expected_impact': 'High',
                    'risk_level': 'High'
                }
            ],
            'digital_transformation': [
                {
                    'name': 'Digital Business Model',
                    'description': 'Transform business model leveraging digital technologies',
                    'actions': [
                        'Assess digital transformation opportunities',
                        'Develop digital business model strategy',
                        'Create implementation roadmap',
                        'Build necessary capabilities and infrastructure'
                    ],
                    'suitable_for': ['medium', 'large'],
                    'resource_requirements': 'High',
                    'timeframe': 'Long-term',
                    'expected_impact': 'High',
                    'risk_level': 'High'
                },
                {
                    'name': 'E-commerce Expansion',
                    'description': 'Develop or enhance e-commerce capabilities',
                    'actions': [
                        'Assess e-commerce opportunity and requirements',
                        'Develop e-commerce strategy and roadmap',
                        'Implement e-commerce platform and capabilities',
                        'Create digital marketing and operations'
                    ],
                    'suitable_for': ['small', 'medium', 'large'],
                    'resource_requirements': 'Medium',
                    'timeframe': 'Medium-term',
                    'expected_impact': 'High',
                    'risk_level': 'Medium'
                },
                {
                    'name': 'Digital Customer Experience',
                    'description': 'Enhance customer experience through digital technologies',
                    'actions': [
                        'Map digital customer journey',
                        'Identify digital experience enhancement opportunities',
                        'Implement digital experience improvements',
                        'Develop digital engagement strategy'
                    ],
                    'suitable_for': ['small', 'medium', 'large'],
                    'resource_requirements': 'Medium',
                    'timeframe': 'Medium-term',
                    'expected_impact': 'Medium',
                    'risk_level': 'Medium'
                }
            ]
        }
    
    def generate_tailored_strategy(self, business_data: Dict[str, Any], 
                                  market_data: Dict[str, Any],
                                  growth_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate tailored growth strategy based on business data and preferences
        
        Args:
            business_data: Dictionary containing business information
            market_data: Dictionary containing market information
            growth_preferences: Dictionary containing growth preferences and constraints
            
        Returns:
            Dictionary containing tailored growth strategy
        """
        result = {
            'timestamp': datetime.now().isoformat(),
            'company_name': business_data.get('name', 'Unknown'),
            'recommended_strategies': [],
            'implementation_roadmap': {},
            'resource_requirements': {},
            'expected_outcomes': {},
            'risk_assessment': {}
        }
        
        # Determine suitable strategy types based on business and market data
        suitable_strategy_types = self._determine_suitable_strategy_types(business_data, market_data)
        
        # Filter strategies based on preferences and constraints
        filtered_strategies = self._filter_strategies_by_preferences(
            suitable_strategy_types, business_data, growth_preferences
        )
        
        # If no strategies match, use broader criteria
        if not filtered_strategies:
            filtered_strategies = self._get_default_strategies(business_data)
        
        # Prioritize and select top strategies
        recommended_strategies = self._prioritize_strategies(filtered_strategies, business_data, market_data)
        result['recommended_strategies'] = recommended_strategies
        
        # Create implementation roadmap
        implementation_roadmap = self._create_implementation_roadmap(recommended_strategies)
        result['implementation_roadmap'] = implementation_roadmap
        
        # Calculate resource requirements
        resource_requirements = self._calculate_resource_requirements(recommended_strategies)
        result['resource_requirements'] = resource_requirements
        
        # Estimate expected outcomes
        expected_outcomes = self._estimate_expected_outcomes(recommended_strategies, business_data, market_data)
        result['expected_outcomes'] = expected_outcomes
        
        # Assess risks
        risk_assessment = self._assess_risks(recommended_strategies, business_data)
        result['risk_assessment'] = risk_assessment
        
        return result
    
    def _determine_suitable_strategy_types(self, business_data: Dict[str, Any], 
                                         market_data: Dict[str, Any]) -> List[str]:
        """Determine which strategy types are suitable based on business and market data"""
        suitable_types = []
        
        # Business characteristics
        business_size = business_data.get('size', 'small')
        market_share = business_data.get('market_share', 0)
        financial_strength = business_data.get('financial_strength', 5)  # 1-10 scale
        
        # Market characteristics
        market_growth = market_data.get('market_growth', 0)
        market_saturation = market_data.get('market_saturation', 50)  # percentage
        competitive_intensity = market_data.get('competitive_intensity', 5)  # 1-10 scale
        
        # Market Penetration suitability
        if market_share < 20 and market_saturation < 80:
            suitable_types.append('market_penetration')
        
        # Market Development suitability
        if financial_strength >= 6 and business_size != 'small':
            suitable_types.append('market_development')
        elif business_size == 'small' and financial_strength >= 7:
            suitable_types.append('market_development')
        
        # Product Development suitability
        if 'r_and_d_capability' in business_data and business_data['r_and_d_capability'] >= 6:
            suitable_types.append('product_development')
        elif business_data.get('innovation_score', 0) >= 7:
            suitable_types.append('product_development')
        
        # Diversification suitability
        if business_size == 'large' and financial_strength >= 8:
            suitable_types.append('diversification')
        elif business_size == 'medium' and financial_strength >= 9:
            suitable_types.append('diversification')
        
        # Digital Transformation suitability
        if business_data.get('digital_maturity', 5) < 7:
            suitable_types.append('digital_transformation')
        
        # Ensure at least one strategy type is included
        if not suitable_types:
            # Default to market penetration as most accessible strategy
            suitable_types.append('market_penetration')
            
            # Add product development if market is saturated
            if market_saturation > 80:
                suitable_types.append('product_development')
        
        return suitable_types
    
    def _filter_strategies_by_preferences(self, strategy_types: List[str],
                                        business_data: Dict[str, Any],
                                        preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter strategies based on preferences and constraints"""
        filtered_strategies = []
        
        # Extract preferences and constraints
        max_timeframe = preferences.get('max_timeframe', 'long-term')
        max_risk = preferences.get('max_risk', 'high')
        max_resources = preferences.get('max_resources', 'high')
        priority_areas = preferences.get('priority_areas', [])
        
        # Map text values to numeric for comparison
        timeframe_map = {'short-term': 1, 'medium-term': 2, 'long-term': 3}
        risk_map = {'low': 1, 'medium': 2, 'high': 3, 'very high': 4}
        resource_map = {'low': 1, 'medium': 2, 'high': 3, 'very high': 4}
        
        max_timeframe_value = timeframe_map.get(max_timeframe.lower(), 3)
        max_risk_value = risk_map.get(max_risk.lower(), 3)
        max_resource_value = resource_map.get(max_resources.lower(), 3)
        
        # Get business size
        business_size = business_data.get('size', 'small')
        
        # Collect all strategies that match criteria
        for strategy_type in strategy_types:
            strategies = self.strategy_database.get(strategy_type, [])
            
            for strategy in strategies:
                # Check if suitable for business size
                if business_size not in strategy.get('suitable_for', []):
                    continue
                
                # Check timeframe constraint
                strategy_timeframe = strategy.get('timeframe', 'Medium-term')
                if timeframe_map.get(strategy_timeframe.lower(), 2) > max_timeframe_value:
                    continue
                
                # Check risk constraint
                strategy_risk = strategy.get('risk_level', 'Medium')
                if risk_map.get(strategy_risk.lower(), 2) > max_risk_value:
                    continue
                
                # Check resource constraint
                strategy_resources = strategy.get('resource_requirements', 'Medium')
                if resource_map.get(strategy_resources.lower(), 2) > max_resource_value:
                    continue
                
                # Add strategy type to the strategy
                strategy_copy = strategy.copy()
                strategy_copy['strategy_type'] = strategy_type
                
                # Add priority score based on priority areas
                priority_score = 0
                if priority_areas:
                    # Check if strategy matches any priority areas
                    strategy_text = json.dumps(strategy).lower()
                    for area in priority_areas:
                        if area.lower() in strategy_text:
                            priority_score += 1
                
                strategy_copy['priority_score'] = priority_score
                
                filtered_strategies.append(strategy_copy)
        
        return filtered_strategies
    
    def _get_default_strategies(self, business_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get default strategies if no strategies match filters"""
        default_strategies = []
        business_size = business_data.get('size', 'small')
        
        # Add market penetration strategies (generally accessible to all)
        for strategy in self.strategy_database.get('market_penetration', []):
            if business_size in strategy.get('suitable_for', []):
                strategy_copy = strategy.copy()
                strategy_copy['strategy_type'] = 'market_penetration'
                strategy_copy['priority_score'] = 0
                default_strategies.append(strategy_copy)
        
        # Add one product enhancement strategy if available
        for strategy in self.strategy_database.get('product_development', []):
            if business_size in strategy.get('suitable_for', []) and strategy.get('name') == 'Product Enhancement':
                strategy_copy = strategy.copy()
                strategy_copy['strategy_type'] = 'product_development'
                strategy_copy['priority_score'] = 0
                default_strategies.append(strategy_copy)
                break
        
        return default_strategies
    
    def _prioritize_strategies(self, strategies: List[Dict[str, Any]],
                             business_data: Dict[str, Any],
                             market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prioritize and select top strategies"""
        # Calculate priority score for each strategy
        for strategy in strategies:
            base_score = strategy.get('priority_score', 0)
            
            # Add points based on expected impact
            impact_score = 0
            if strategy.get('expected_impact') == 'High':
                impact_score = 3
            elif strategy.get('expected_impact') == 'Medium':
                impact_score = 2
            else:
                impact_score = 1
                
            # Add points based on timeframe (shorter is better)
            timeframe_score = 0
            if strategy.get('timeframe') == 'Short-term':
                timeframe_score = 3
            elif strategy.get('timeframe') == 'Medium-term':
                timeframe_score = 2
            else:
                timeframe_score = 1
                
            # Add points based on risk level (lower is better)
            risk_score = 0
            if strategy.get('risk_level') == 'Low':
                risk_score = 3
            elif strategy.get('risk_level') == 'Medium':
                risk_score = 2
            else:
                risk_score = 1
                
            # Calculate final score
            final_score = base_score + impact_score + timeframe_score + risk_score
            strategy['final_score'] = final_score
        
        # Sort strategies by final score (descending)
        sorted_strategies = sorted(strategies, key=lambda x: x.get('final_score', 0), reverse=True)
        
        # Select top strategies (max 5)
        top_strategies = sorted_strategies[:5]
        
        # Remove scoring fields from output
        for strategy in top_strategies:
            if 'priority_score' in strategy:
                del strategy['priority_score']
            if 'final_score' in strategy:
                del strategy['final_score']
        
        return top_strategies
    
    def _create_implementation_roadmap(self, strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create phased implementation roadmap"""
        roadmap = {
            'phase_1': {
                'timeframe': '0-3 months',
                'activities': []
            },
            'phase_2': {
                'timeframe': '3-6 months',
                'activities': []
            },
            'phase_3': {
                'timeframe': '6-12 months',
                'activities': []
            },
            'phase_4': {
                'timeframe': '12+ months',
                'activities': []
            }
        }
        
        # Distribute strategy actions across phases
        for strategy in strategies:
            strategy_name = strategy.get('name', 'Strategy')
            actions = strategy.get('actions', [])
            timeframe = strategy.get('timeframe', 'Medium-term')
            
            # Determine starting phase based on timeframe
            if timeframe == 'Short-term':
                start_phase = 'phase_1'
            elif timeframe == 'Medium-term':
                start_phase = 'phase_2'
            else:
                start_phase = 'phase_3'
            
            # Add strategy initialization to starting phase
            roadmap[start_phase]['activities'].append(f"Initialize {strategy_name} strategy")
            
            # Distribute actions across phases
            if len(actions) == 4:  # Most strategies have 4 actions
                if start_phase == 'phase_1':
                    roadmap['phase_1']['activities'].append(f"{strategy_name}: {actions[0]}")
                    roadmap['phase_2']['activities'].append(f"{strategy_name}: {actions[1]}")
                    roadmap['phase_3']['activities'].append(f"{strategy_name}: {actions[2]}")
                    roadmap['phase_4']['activities'].append(f"{strategy_name}: {actions[3]}")
                elif start_phase == 'phase_2':
                    roadmap['phase_2']['activities'].append(f"{strategy_name}: {actions[0]}")
                    roadmap['phase_2']['activities'].append(f"{strategy_name}: {actions[1]}")
                    roadmap['phase_3']['activities'].append(f"{strategy_name}: {actions[2]}")
                    roadmap['phase_4']['activities'].append(f"{strategy_name}: {actions[3]}")
                else:
                    roadmap['phase_3']['activities'].append(f"{strategy_name}: {actions[0]}")
                    roadmap['phase_3']['activities'].append(f"{strategy_name}: {actions[1]}")
                    roadmap['phase_4']['activities'].append(f"{strategy_name}: {actions[2]}")
                    roadmap['phase_4']['activities'].append(f"{strategy_name}: {actions[3]}")
            else:
                # Distribute evenly for any number of actions
                phase_index = ['phase_1', 'phase_2', 'phase_3', 'phase_4'].index(start_phase)
                for i, action in enumerate(actions):
                    phase = min(phase_index + (i // 2), 3)  # Limit to phase_4
                    phase_name = f"phase_{phase + 1}"
                    roadmap[phase_name]['activities'].append(f"{strategy_name}: {action}")
        
        return roadmap
    def _calculate_resource_requirements(self, strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate resource requirements for implementing strategies"""
        resource_requirements = {
            'financial': {
                'initial_investment': 0,
                'ongoing_monthly': 0,
                'breakdown': {}
            },
            'human': {
                'headcount': 0,
                'skills_required': [],
                'breakdown': {}
            },
            'time': {
                'implementation_timeline': '',
                'key_milestones': []
            }
        }
        
        # Define resource mapping (very simplified)
        financial_map = {
            'Low': {'initial': 10000, 'monthly': 2000},
            'Medium': {'initial': 50000, 'monthly': 8000},
            'High': {'initial': 150000, 'monthly': 20000},
            'Very High': {'initial': 500000, 'monthly': 50000}
        }
        
        headcount_map = {
            'Low': 0.5,  # Part-time resource
            'Medium': 1,  # Full-time resource
            'High': 3,    # Small team
            'Very High': 5  # Dedicated team
        }
        
        skills_map = {
            'market_penetration': ['Marketing', 'Sales', 'Customer Success'],
            'market_development': ['Market Research', 'Business Development', 'International Operations'],
            'product_development': ['Product Management', 'R&D', 'UX Design', 'Engineering'],
            'diversification': ['M&A', 'Strategic Planning', 'New Business Development'],
            'digital_transformation': ['Digital Strategy', 'Technology', 'Change Management']
        }
        
        # Calculate total resources
        for strategy in strategies:
            strategy_name = strategy.get('name', 'Strategy')
            resource_level = strategy.get('resource_requirements', 'Medium')
            strategy_type = strategy.get('strategy_type', 'market_penetration')
            
            # Financial resources
            financial_need = financial_map.get(resource_level, financial_map['Medium'])
            resource_requirements['financial']['initial_investment'] += financial_need['initial']
            resource_requirements['financial']['ongoing_monthly'] += financial_need['monthly']
            
            # Add to breakdown
            resource_requirements['financial']['breakdown'][strategy_name] = {
                'initial': financial_need['initial'],
                'monthly': financial_need['monthly']
            }
            
            # Human resources
            headcount = headcount_map.get(resource_level, headcount_map['Medium'])
            resource_requirements['human']['headcount'] += headcount
            
            # Add to breakdown
            resource_requirements['human']['breakdown'][strategy_name] = {
                'headcount': headcount
            }
            
            # Skills required
            if strategy_type in skills_map:
                for skill in skills_map[strategy_type]:
                    if skill not in resource_requirements['human']['skills_required']:
                        resource_requirements['human']['skills_required'].append(skill)
        
        # Calculate implementation timeline
        has_short_term = any(s.get('timeframe') == 'Short-term' for s in strategies)
        has_medium_term = any(s.get('timeframe') == 'Medium-term' for s in strategies)
        has_long_term = any(s.get('timeframe') == 'Long-term' for s in strategies)
        
        if has_long_term:
            resource_requirements['time']['implementation_timeline'] = '12-18 months'
        elif has_medium_term:
            resource_requirements['time']['implementation_timeline'] = '6-12 months'
        else:
            resource_requirements['time']['implementation_timeline'] = '3-6 months'
        
        # Define key milestones
        resource_requirements['time']['key_milestones'] = [
            {'timeframe': '30 days', 'milestone': 'Strategy kickoff and initial planning complete'},
            {'timeframe': '90 days', 'milestone': 'First implementations launched'},
            {'timeframe': '6 months', 'milestone': 'Initial results assessment and strategy refinement'},
            {'timeframe': '12 months', 'milestone': 'Full implementation and results evaluation'}
        ]
        
        return resource_requirements
    
    def _estimate_expected_outcomes(self, strategies: List[Dict[str, Any]],
                                  business_data: Dict[str, Any],
                                  market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate expected outcomes from implementing strategies"""
        outcomes = {
            'revenue_impact': {
                'year_1': 0,
                'year_2': 0,
                'year_3': 0
            },
            'market_share_impact': {
                'year_1': 0,
                'year_2': 0,
                'year_3': 0
            },
            'roi_estimate': {
                'payback_period': '',
                'expected_roi': 0
            },
            'qualitative_outcomes': []
        }
        
        # Get baseline metrics
        current_revenue = business_data.get('revenue', 1000000)  # Default $1M if not provided
        current_market_share = business_data.get('market_share', 5)  # Default 5% if not provided
        market_growth = market_data.get('market_growth', 3)  # Default 3% if not provided
        
        # Define impact mapping (very simplified)
        revenue_impact_map = {
            'Low': {'year_1': 0.03, 'year_2': 0.05, 'year_3': 0.07},
            'Medium': {'year_1': 0.05, 'year_2': 0.10, 'year_3': 0.15},
            'High': {'year_1': 0.10, 'year_2': 0.20, 'year_3': 0.30}
        }
        
        market_share_impact_map = {
            'Low': {'year_1': 0.2, 'year_2': 0.5, 'year_3': 0.7},
            'Medium': {'year_1': 0.5, 'year_2': 1.0, 'year_3': 1.5},
            'High': {'year_1': 1.0, 'year_2': 2.0, 'year_3': 3.0}
        }
        
        # Calculate cumulative impact
        total_revenue_impact = {'year_1': 0, 'year_2': 0, 'year_3': 0}
        total_market_share_impact = {'year_1': 0, 'year_2': 0, 'year_3': 0}
        
        for strategy in strategies:
            impact_level = strategy.get('expected_impact', 'Medium')
            
            # Revenue impact
            revenue_impact = revenue_impact_map.get(impact_level, revenue_impact_map['Medium'])
            
            # Apply diminishing returns for multiple strategies
            diminishing_factor = 0.8  # Each additional strategy has 80% of the impact
            factor = diminishing_factor ** (len(total_revenue_impact) - 1)
            
            for year in ['year_1', 'year_2', 'year_3']:
                total_revenue_impact[year] += revenue_impact[year] * factor
            
            # Market share impact
            market_impact = market_share_impact_map.get(impact_level, market_share_impact_map['Medium'])
            
            for year in ['year_1', 'year_2', 'year_3']:
                total_market_share_impact[year] += market_impact[year] * factor
        
        # Cap maximum impacts at reasonable levels
        total_revenue_impact = {
            'year_1': min(0.25, total_revenue_impact['year_1']),
            'year_2': min(0.40, total_revenue_impact['year_2']),
            'year_3': min(0.60, total_revenue_impact['year_3'])
        }
        
        total_market_share_impact = {
            'year_1': min(2.0, total_market_share_impact['year_1']),
            'year_2': min(4.0, total_market_share_impact['year_2']),
            'year_3': min(6.0, total_market_share_impact['year_3'])
        }
        
        # Calculate absolute impacts
        for year in ['year_1', 'year_2', 'year_3']:
            # Compound growth for revenue (market growth + strategy impact)
            compound_growth = (1 + market_growth/100) * (1 + total_revenue_impact[year])
            if year == 'year_1':
                outcomes['revenue_impact']['year_1'] = current_revenue * total_revenue_impact['year_1']
            elif year == 'year_2':
                base_revenue = current_revenue * (1 + total_revenue_impact['year_1'])
                outcomes['revenue_impact']['year_2'] = base_revenue * total_revenue_impact['year_2']
            else:
                base_revenue = current_revenue * (1 + total_revenue_impact['year_1']) * (1 + total_revenue_impact['year_2'])
                outcomes['revenue_impact']['year_3'] = base_revenue * total_revenue_impact['year_3']
        
        # Calculate market share impact
        outcomes['market_share_impact'] = {
            'year_1': total_market_share_impact['year_1'],
            'year_2': total_market_share_impact['year_2'],
            'year_3': total_market_share_impact['year_3']
        }
        
        # Calculate ROI
        total_investment = 0
        for strategy in strategies:
            resource_level = strategy.get('resource_requirements', 'Medium')
            if resource_level == 'Low':
                total_investment += 50000  # Simplified total cost over 3 years
            elif resource_level == 'Medium':
                total_investment += 200000
            elif resource_level == 'High':
                total_investment += 500000
            else:  # Very High
                total_investment += 1000000
        
        total_revenue_gain = (
            outcomes['revenue_impact']['year_1'] +
            outcomes['revenue_impact']['year_2'] +
            outcomes['revenue_impact']['year_3']
        )
        
        # Assume 20% of revenue gain is profit
        profit_gain = total_revenue_gain * 0.2
        
        # Calculate ROI
        if total_investment > 0:
            roi = (profit_gain / total_investment) * 100
            outcomes['roi_estimate']['expected_roi'] = roi
            
            # Estimate payback period
            if profit_gain <= 0:
                outcomes['roi_estimate']['payback_period'] = 'Not within 3 years'
            else:
                annual_profit = profit_gain / 3
                payback_years = total_investment / annual_profit
                
                if payback_years <= 1:
                    outcomes['roi_estimate']['payback_period'] = 'Less than 1 year'
                elif payback_years <= 2:
                    outcomes['roi_estimate']['payback_period'] = '1-2 years'
                elif payback_years <= 3:
                    outcomes['roi_estimate']['payback_period'] = '2-3 years'
                else:
                    outcomes['roi_estimate']['payback_period'] = 'More than 3 years'
        else:
            outcomes['roi_estimate']['expected_roi'] = 0
            outcomes['roi_estimate']['payback_period'] = 'N/A'
        
        # Qualitative outcomes
        for strategy in strategies:
            strategy_type = strategy.get('strategy_type', '')
            
            if strategy_type == 'market_penetration':
                outcomes['qualitative_outcomes'].append('Increased market share in existing markets')
                outcomes['qualitative_outcomes'].append('Strengthened brand recognition and loyalty')
                outcomes['qualitative_outcomes'].append('Improved competitive position')
            elif strategy_type == 'market_development':
                outcomes['qualitative_outcomes'].append('Diversified revenue streams across markets')
                outcomes['qualitative_outcomes'].append('Reduced dependency on single market')
                outcomes['qualitative_outcomes'].append('Expanded customer base')
            elif strategy_type == 'product_development':
                outcomes['qualitative_outcomes'].append('Enhanced product portfolio')
                outcomes['qualitative_outcomes'].append('Increased customer value and retention')
                outcomes['qualitative_outcomes'].append('Strengthened innovation capabilities')
            elif strategy_type == 'diversification':
                outcomes['qualitative_outcomes'].append('Reduced business risk through diversification')
                outcomes['qualitative_outcomes'].append('Access to new growth opportunities')
                outcomes['qualitative_outcomes'].append('Enhanced organizational capabilities')
            elif strategy_type == 'digital_transformation':
                outcomes['qualitative_outcomes'].append('Improved operational efficiency')
                outcomes['qualitative_outcomes'].append('Enhanced customer experience and engagement')
                outcomes['qualitative_outcomes'].append('Increased business agility and resilience')
        
        # Remove duplicates while preserving order
        outcomes['qualitative_outcomes'] = list(dict.fromkeys(outcomes['qualitative_outcomes']))
        
        return outcomes
    
    def _assess_risks(self, strategies: List[Dict[str, Any]], 
                    business_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risks associated with recommended strategies"""
        risk_assessment = {
            'overall_risk_level': '',
            'key_risks': [],
            'risk_mitigation': [],
            'risk_factors': {}
        }
        
        # Define risk mapping
        strategy_risks = {
            'market_penetration': [
                {'risk': 'Competitive response', 'likelihood': 'High', 'impact': 'Medium'},
                {'risk': 'Market saturation', 'likelihood': 'Medium', 'impact': 'Medium'},
                {'risk': 'Price pressure', 'likelihood': 'High', 'impact': 'Medium'}
            ],
            'market_development': [
                {'risk': 'Market entry barriers', 'likelihood': 'Medium', 'impact': 'High'},
                {'risk': 'Cultural/regional differences', 'likelihood': 'High', 'impact': 'Medium'},
                {'risk': 'Resource strain', 'likelihood': 'Medium', 'impact': 'Medium'}
            ],
            'product_development': [
                {'risk': 'Development delays', 'likelihood': 'High', 'impact': 'Medium'},
                {'risk': 'Product-market fit failure', 'likelihood': 'Medium', 'impact': 'High'},
                {'risk': 'Higher than expected costs', 'likelihood': 'Medium', 'impact': 'Medium'}
            ],
            'diversification': [
                {'risk': 'Lack of expertise in new area', 'likelihood': 'High', 'impact': 'High'},
                {'risk': 'Overextension of resources', 'likelihood': 'High', 'impact': 'High'},
                {'risk': 'Integration challenges', 'likelihood': 'High', 'impact': 'Medium'}
            ],
            'digital_transformation': [
                {'risk': 'Technology implementation issues', 'likelihood': 'Medium', 'impact': 'High'},
                {'risk': 'Change resistance', 'likelihood': 'High', 'impact': 'Medium'},
                {'risk': 'Skill gaps', 'likelihood': 'Medium', 'impact': 'Medium'}
            ]
        }
        
        # Collect risks from all strategies
        all_risks = []
        risk_scores = []
        
        for strategy in strategies:
            strategy_type = strategy.get('strategy_type', 'market_penetration')
            strategy_risk_level = strategy.get('risk_level', 'Medium')
            
            # Get risks for this strategy type
            type_risks = strategy_risks.get(strategy_type, [])
            
            # Add to risk factors
            risk_assessment['risk_factors'][strategy.get('name', 'Strategy')] = {
                'risk_level': strategy_risk_level,
                'specific_risks': []
            }
            
            # Add specific risks
            for risk in type_risks:
                # Adjust likelihood based on strategy risk level
                if strategy_risk_level == 'High':
                    likelihood = risk['likelihood']
                elif strategy_risk_level == 'Medium':
                    # Downgrade high to medium
                    likelihood = 'Medium' if risk['likelihood'] == 'High' else risk['likelihood']
                else:  # Low
                    # Downgrade all likelihoods
                    likelihood = 'Low' if risk['likelihood'] == 'High' else ('Low' if risk['likelihood'] == 'Medium' else 'Low')
                
                risk_item = {
                    'risk': risk['risk'],
                    'likelihood': likelihood,
                    'impact': risk['impact']
                }
                
                risk_assessment['risk_factors'][strategy.get('name', 'Strategy')]['specific_risks'].append(risk_item)
                all_risks.append(risk_item)
                
                # Calculate risk score
                likelihood_score = 3 if likelihood == 'High' else (2 if likelihood == 'Medium' else 1)
                impact_score = 3 if risk['impact'] == 'High' else (2 if risk['impact'] == 'Medium' else 1)
                risk_scores.append(likelihood_score * impact_score)
        
        # Determine overall risk level
        if risk_scores:
            avg_risk_score = sum(risk_scores) / len(risk_scores)
            
            if avg_risk_score > 6:
                risk_assessment['overall_risk_level'] = 'High'
            elif avg_risk_score > 3:
                risk_assessment['overall_risk_level'] = 'Medium'
            else:
                risk_assessment['overall_risk_level'] = 'Low'
        else:
            risk_assessment['overall_risk_level'] = 'Low'
        
        # Identify key risks (high impact risks)
        high_impact_risks = [risk for risk in all_risks if risk['impact'] == 'High']
        
        # If no high impact risks, take highest likelihood risks
        if not high_impact_risks:
            high_likelihood_risks = [risk for risk in all_risks if risk['likelihood'] == 'High']
            key_risks = high_likelihood_risks[:3] if high_likelihood_risks else all_risks[:3]
        else:
            key_risks = high_impact_risks[:3]
        
        risk_assessment['key_risks'] = key_risks
        
        # Generate risk mitigation strategies
        for risk in key_risks:
            risk_type = risk['risk']
            
            if 'competitive response' in risk_type.lower():
                risk_assessment['risk_mitigation'].append({
                    'risk': risk_type,
                    'mitigation': 'Develop competitive response plan and monitor competitor actions closely'
                })
            elif 'market entry' in risk_type.lower():
                risk_assessment['risk_mitigation'].append({
                    'risk': risk_type,
                    'mitigation': 'Conduct thorough market research and consider partnerships to overcome entry barriers'
                })
            elif 'product-market fit' in risk_type.lower():
                risk_assessment['risk_mitigation'].append({
                    'risk': risk_type,
                    'mitigation': 'Implement iterative development with customer feedback throughout process'
                })
            elif 'expertise' in risk_type.lower():
                risk_assessment['risk_mitigation'].append({
                    'risk': risk_type,
                    'mitigation': 'Acquire necessary expertise through hiring, partnerships, or acquisitions'
                })
            elif 'technology' in risk_type.lower():
                risk_assessment['risk_mitigation'].append({
                    'risk': risk_type,
                    'mitigation': 'Implement phased approach with thorough testing and contingency planning'
                })
            elif 'change resistance' in risk_type.lower():
                risk_assessment['risk_mitigation'].append({
                    'risk': risk_type,
                    'mitigation': 'Develop comprehensive change management and communication plan'
                })
            elif 'resource' in risk_type.lower() or 'overextension' in risk_type.lower():
                risk_assessment['risk_mitigation'].append({
                    'risk': risk_type,
                    'mitigation': 'Implement phased approach and establish clear resource allocation priorities'
                })
            elif 'cost' in risk_type.lower():
                risk_assessment['risk_mitigation'].append({
                    'risk': risk_type,
                    'mitigation': 'Establish budget contingencies and implement strict cost monitoring'
                })
            else:
                risk_assessment['risk_mitigation'].append({
                    'risk': risk_type,
                    'mitigation': 'Develop contingency plans and establish ongoing monitoring'
                })
        
        return risk_assessment