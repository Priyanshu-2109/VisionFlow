# app/modules/risk_manager/mitigation_strategies.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class MitigationStrategies:
    """
    Generate risk mitigation strategies for various business risks
    """
    
    def __init__(self):
        self.strategy_database = self._load_strategy_database()
    
    def _load_strategy_database(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load risk mitigation strategy database"""
        # In a real implementation, this would load from a database or file
        # Using hardcoded values for demonstration
        
        return {
            'financial_risk': [
                {
                    'risk_factor': 'Low Current Ratio',
                    'strategies': [
                        {
                            'strategy': 'Improve working capital management',
                            'actions': [
                                'Optimize inventory levels to reduce excess stock',
                                'Improve accounts receivable collection processes',
                                'Negotiate extended payment terms with suppliers',
                                'Implement cash forecasting system'
                            ],
                            'timeframe': 'Short-term',
                            'complexity': 'Medium',
                            'resource_requirements': 'Low'
                        },
                        {
                            'strategy': 'Restructure short-term debt',
                            'actions': [
                                'Convert short-term debt to long-term financing',
                                'Negotiate with lenders to modify repayment schedules',
                                'Explore alternative financing options'
                            ],
                            'timeframe': 'Medium-term',
                            'complexity': 'High',
                            'resource_requirements': 'Medium'
                        }
                    ]
                },
                {
                    'risk_factor': 'High Debt-to-Equity Ratio',
                    'strategies': [
                        {
                            'strategy': 'Reduce debt levels',
                            'actions': [
                                'Develop debt reduction plan with specific targets',
                                'Allocate free cash flow to debt repayment',
                                'Consider asset sales to pay down debt',
                                'Limit new capital expenditures'
                            ],
                            'timeframe': 'Medium-term',
                            'complexity': 'Medium',
                            'resource_requirements': 'Low'
                        },
                        {
                            'strategy': 'Increase equity funding',
                            'actions': [
                                'Explore equity financing options',
                                'Retain earnings rather than distributing dividends',
                                'Consider strategic investors or partnerships'
                            ],
                            'timeframe': 'Medium-term',
                            'complexity': 'High',
                            'resource_requirements': 'Medium'
                        }
                    ]
                },
                {
                    'risk_factor': 'Low Profit Margins',
                    'strategies': [
                        {
                            'strategy': 'Improve operational efficiency',
                            'actions': [
                                'Conduct comprehensive cost structure analysis',
                                'Implement cost reduction initiatives',
                                'Optimize production processes',
                                'Automate manual processes where feasible'
                            ],
                            'timeframe': 'Medium-term',
                            'complexity': 'Medium',
                            'resource_requirements': 'Medium'
                        },
                        {
                            'strategy': 'Enhance pricing strategy',
                            'actions': [
                                'Analyze pricing relative to value delivered',
                                'Implement value-based pricing',
                                'Evaluate product mix and focus on higher-margin offerings',
                                'Reduce discounting practices'
                            ],
                            'timeframe': 'Short-term',
                            'complexity': 'Medium',
                            'resource_requirements': 'Low'
                        }
                    ]
                }
            ],
            'market_risk': [
                {
                    'risk_factor': 'High Competitive Intensity',
                    'strategies': [
                        {
                            'strategy': 'Enhance differentiation',
                            'actions': [
                                'Identify and strengthen unique value proposition',
                                'Invest in product/service innovation',
                                'Develop proprietary technologies or processes',
                                'Create switching costs for customers'
                            ],
                            'timeframe': 'Medium-term',
                            'complexity': 'High',
                            'resource_requirements': 'Medium'
                        },
                        {
                            'strategy': 'Focus on underserved segments',
                            'actions': [
                                'Identify niche market segments with less competition',
                                'Develop targeted offerings for specific customer segments',
                                'Create specialized expertise in selected areas'
                            ],
                            'timeframe': 'Medium-term',
                            'complexity': 'Medium',
                            'resource_requirements': 'Medium'
                        }
                    ]
                },
                {
                    'risk_factor': 'Low Market Share',
                    'strategies': [
                        {
                            'strategy': 'Increase customer acquisition',
                            'actions': [
                                'Develop targeted marketing campaigns',
                                'Enhance sales capabilities and processes',
                                'Create compelling introductory offers',
                                'Explore partnership channels'
                            ],
                            'timeframe': 'Short-term',
                            'complexity': 'Medium',
                            'resource_requirements': 'Medium'
                        },
                        {
                            'strategy': 'Improve customer retention',
                            'actions': [
                                'Implement customer success program',
                                'Develop loyalty initiatives',
                                'Enhance customer service capabilities',
                                'Create ongoing engagement touchpoints'
                            ],
                            'timeframe': 'Short-term',
                            'complexity': 'Medium',
                            'resource_requirements': 'Medium'
                        }
                    ]
                },
                {
                    'risk_factor': 'Declining Market',
                    'strategies': [
                        {
                            'strategy': 'Diversify into adjacent markets',
                            'actions': [
                                'Identify related markets with growth potential',
                                'Leverage existing capabilities in new markets',
                                'Develop market entry strategies',
                                'Consider acquisitions to accelerate entry'
                            ],
                            'timeframe': 'Long-term',
                            'complexity': 'High',
                            'resource_requirements': 'High'
                        },
                        {
                            'strategy': 'Optimize for profitability',
                            'actions': [
                                'Focus on most profitable customer segments',
                                'Streamline operations and reduce costs',
                                'Eliminate underperforming products/services',
                                'Consider consolidation opportunities'
                            ],
                            'timeframe': 'Medium-term',
                            'complexity': 'Medium',
                            'resource_requirements': 'Medium'
                        }
                    ]
                }
            ],
            'operational_risk': [
                {
                    'risk_factor': 'High Supplier Concentration',
                    'strategies': [
                        {
                            'strategy': 'Diversify supplier base',
                            'actions': [
                                'Identify and qualify alternative suppliers',
                                'Develop relationships with multiple suppliers',
                                'Implement dual-sourcing for critical components',
                                'Create supplier onboarding process'
                            ],
                            'timeframe': 'Medium-term',
                            'complexity': 'Medium',
                            'resource_requirements': 'Medium'
                        },
                        {
                            'strategy': 'Strengthen supplier relationships',
                            'actions': [
                                'Develop strategic partnerships with key suppliers',
                                'Implement supplier performance management system',
                                'Create joint improvement initiatives',
                                'Establish clear communication channels'
                            ],
                            'timeframe': 'Medium-term',
                            'complexity': 'Medium',
                            'resource_requirements': 'Low'
                        }
                    ]
                },
                {
                    'risk_factor': 'High Employee Turnover',
                    'strategies': [
                        {
                            'strategy': 'Enhance employee engagement',
                            'actions': [
                                'Conduct employee satisfaction surveys',
                                'Implement engagement initiatives based on feedback',
                                'Improve internal communication',
                                'Create career development opportunities'
                            ],
                            'timeframe': 'Medium-term',
                            'complexity': 'Medium',
                            'resource_requirements': 'Medium'
                        },
                        {
                            'strategy': 'Improve compensation and benefits',
                            'actions': [
                                'Benchmark compensation against industry standards',
                                'Develop competitive compensation packages',
                                'Implement performance-based incentives',
                                'Enhance non-monetary benefits'
                            ],
                            'timeframe': 'Short-term',
                            'complexity': 'Medium',
                            'resource_requirements': 'High'
                        }
                    ]
                },
                {
                    'risk_factor': 'IT Security Incidents',
                    'strategies': [
                        {
                            'strategy': 'Strengthen cybersecurity measures',
                            'actions': [
                                'Conduct comprehensive security assessment',
                                'Implement security best practices and controls',
                                'Develop incident response plan',
                                'Conduct regular security testing'
                            ],
                            'timeframe': 'Short-term',
                            'complexity': 'High',
                            'resource_requirements': 'Medium'
                        },
                        {
                            'strategy': 'Enhance security awareness',
                            'actions': [
                                'Implement security awareness training program',
                                'Conduct phishing simulations',
                                'Develop security policies and procedures',
                                'Create security-focused culture'
                            ],
                            'timeframe': 'Short-term',
                            'complexity': 'Medium',
                            'resource_requirements': 'Low'
                        }
                    ]
                }
            ]
        }
    def get_mitigation_strategies(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate mitigation strategies based on identified risks
        
        Args:
            risk_data: Dictionary containing risk analysis results
            
        Returns:
            Dictionary containing mitigation strategies
        """
        result = {
            'timestamp': datetime.now().isoformat(),
            'risk_type': risk_data.get('risk_type', 'unknown'),
            'strategies': [],
            'implementation_plan': {},
            'resource_requirements': {}
        }
        
        # Extract risk factors from risk data
        risk_factors = risk_data.get('risk_factors', [])
        
        if not risk_factors:
            result['strategies'] = []
            result['implementation_plan'] = {
                'immediate_actions': [],
                'short_term_actions': [],
                'long_term_actions': []
            }
            return result
        
        # Determine risk type
        if 'financial' in risk_data.get('risk_type', '').lower():
            risk_type = 'financial_risk'
        elif 'market' in risk_data.get('risk_type', '').lower():
            risk_type = 'market_risk'
        elif 'operational' in risk_data.get('risk_type', '').lower():
            risk_type = 'operational_risk'
        else:
            # Default to financial risk
            risk_type = 'financial_risk'
        
        # Get strategies for each risk factor
        all_strategies = []
        
        for risk_factor in risk_factors:
            factor_name = risk_factor.get('factor', '')
            severity = risk_factor.get('severity', 'medium')
            
            strategies = self._find_strategies(risk_type, factor_name)
            
            if strategies:
                for strategy in strategies:
                    strategy['risk_factor'] = factor_name
                    strategy['risk_severity'] = severity
                    all_strategies.append(strategy)
            else:
                # If no specific strategy found, provide generic strategy
                generic_strategy = self._create_generic_strategy(risk_type, factor_name, severity)
                all_strategies.append(generic_strategy)
        
        # Prioritize strategies
        prioritized_strategies = self._prioritize_strategies(all_strategies)
        result['strategies'] = prioritized_strategies
        
        # Create implementation plan
        implementation_plan = self._create_implementation_plan(prioritized_strategies)
        result['implementation_plan'] = implementation_plan
        
        # Calculate resource requirements
        resource_requirements = self._calculate_resource_requirements(prioritized_strategies)
        result['resource_requirements'] = resource_requirements
        
        return result
    
    def _find_strategies(self, risk_type: str, factor_name: str) -> List[Dict[str, Any]]:
        """Find strategies for a specific risk factor"""
        strategies = []
        
        # Get strategies for the risk type
        risk_type_strategies = self.strategy_database.get(risk_type, [])
        
        # Find strategies that match the risk factor (partial match)
        for strategy_entry in risk_type_strategies:
            if self._is_similar_risk_factor(strategy_entry.get('risk_factor', ''), factor_name):
                for strategy in strategy_entry.get('strategies', []):
                    strategies.append(strategy.copy())
        
        return strategies
    
    def _is_similar_risk_factor(self, db_factor: str, input_factor: str) -> bool:
        """Check if risk factors are similar (partial match)"""
        db_factor_lower = db_factor.lower()
        input_factor_lower = input_factor.lower()
        
        # Check for exact match
        if db_factor_lower == input_factor_lower:
            return True
        
        # Check for partial matches
        key_terms_db = set(db_factor_lower.split())
        key_terms_input = set(input_factor_lower.split())
        
        # If any key terms match, consider it similar
        if key_terms_db.intersection(key_terms_input):
            return True
        
        # Check for specific patterns
        if 'ratio' in db_factor_lower and 'ratio' in input_factor_lower:
            return True
        if 'margin' in db_factor_lower and 'margin' in input_factor_lower:
            return True
        if 'debt' in db_factor_lower and 'debt' in input_factor_lower:
            return True
        if 'liquidity' in db_factor_lower and ('current' in input_factor_lower or 'quick' in input_factor_lower):
            return True
        if 'market share' in db_factor_lower and 'market share' in input_factor_lower:
            return True
        if 'supplier' in db_factor_lower and 'supplier' in input_factor_lower:
            return True
        if 'employee' in db_factor_lower and 'employee' in input_factor_lower:
            return True
        if 'turnover' in db_factor_lower and 'turnover' in input_factor_lower:
            return True
        
        return False
    
    def _create_generic_strategy(self, risk_type: str, factor_name: str, 
                               severity: str) -> Dict[str, Any]:
        """Create generic strategy for risk factors without specific strategies"""
        generic_strategy = {
            'risk_factor': factor_name,
            'risk_severity': severity,
            'strategy': f"Address {factor_name} risk",
            'actions': [
                f"Conduct detailed assessment of {factor_name}",
                "Identify root causes and contributing factors",
                "Develop targeted mitigation actions",
                "Implement regular monitoring and reporting"
            ],
            'timeframe': 'Medium-term',
            'complexity': 'Medium',
            'resource_requirements': 'Medium'
        }
        
        # Adjust based on risk type
        if risk_type == 'financial_risk':
            generic_strategy['actions'].append("Review financial policies and controls")
            generic_strategy['actions'].append("Consider financial restructuring options")
        elif risk_type == 'market_risk':
            generic_strategy['actions'].append("Analyze market trends and competitive landscape")
            generic_strategy['actions'].append("Evaluate strategic positioning and differentiation")
        elif risk_type == 'operational_risk':
            generic_strategy['actions'].append("Review operational processes and controls")
            generic_strategy['actions'].append("Implement operational improvements and redundancies")
        
        # Adjust based on severity
        if severity == 'high':
            generic_strategy['timeframe'] = 'Short-term'
            generic_strategy['actions'].insert(0, "Develop immediate response plan")
        
        return generic_strategy
    
    def _prioritize_strategies(self, strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize strategies based on risk severity and implementation complexity"""
        # Add priority score to each strategy
        for strategy in strategies:
            severity_score = 3 if strategy.get('risk_severity') == 'high' else (2 if strategy.get('risk_severity') == 'medium' else 1)
            complexity_score = 1 if strategy.get('complexity') == 'High' else (2 if strategy.get('complexity') == 'Medium' else 3)
            timeframe_score = 3 if strategy.get('timeframe') == 'Short-term' else (2 if strategy.get('timeframe') == 'Medium-term' else 1)
            
            # Calculate priority score (higher is higher priority)
            priority_score = severity_score * 0.5 + complexity_score * 0.3 + timeframe_score * 0.2
            strategy['priority_score'] = priority_score
            
            # Assign priority level
            if priority_score >= 2.5:
                strategy['priority'] = 'High'
            elif priority_score >= 1.5:
                strategy['priority'] = 'Medium'
            else:
                strategy['priority'] = 'Low'
        
        # Sort strategies by priority score (descending)
        sorted_strategies = sorted(strategies, key=lambda x: x.get('priority_score', 0), reverse=True)
        
        # Remove priority score from output
        for strategy in sorted_strategies:
            if 'priority_score' in strategy:
                del strategy['priority_score']
        
        return sorted_strategies
    
    def _create_implementation_plan(self, strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create phased implementation plan based on strategies"""
        implementation_plan = {
            'immediate_actions': [],
            'short_term_actions': [],
            'long_term_actions': []
        }
        
        # Distribute actions based on timeframe and priority
        for strategy in strategies:
            actions = strategy.get('actions', [])
            timeframe = strategy.get('timeframe', 'Medium-term')
            priority = strategy.get('priority', 'Medium')
            
            if not actions:
                continue
            
            if timeframe == 'Short-term' and priority == 'High':
                # Immediate actions (high priority, short-term)
                implementation_plan['immediate_actions'].extend([
                    f"{action} [{strategy.get('risk_factor', '')}]" for action in actions[:2]
                ])
                if len(actions) > 2:
                    implementation_plan['short_term_actions'].extend([
                        f"{action} [{strategy.get('risk_factor', '')}]" for action in actions[2:]
                    ])
            elif timeframe == 'Short-term' or priority == 'High':
                # Short-term actions (either short-term or high priority)
                implementation_plan['short_term_actions'].extend([
                    f"{action} [{strategy.get('risk_factor', '')}]" for action in actions
                ])
            else:
                # Long-term actions (medium/long-term and medium/low priority)
                implementation_plan['long_term_actions'].extend([
                    f"{action} [{strategy.get('risk_factor', '')}]" for action in actions
                ])
        
        # Remove duplicates while preserving order
        implementation_plan['immediate_actions'] = list(dict.fromkeys(implementation_plan['immediate_actions']))
        implementation_plan['short_term_actions'] = list(dict.fromkeys(implementation_plan['short_term_actions']))
        implementation_plan['long_term_actions'] = list(dict.fromkeys(implementation_plan['long_term_actions']))
        
        # Limit number of actions in each category
        implementation_plan['immediate_actions'] = implementation_plan['immediate_actions'][:5]
        implementation_plan['short_term_actions'] = implementation_plan['short_term_actions'][:10]
        implementation_plan['long_term_actions'] = implementation_plan['long_term_actions'][:10]
        
        return implementation_plan
    
    def _calculate_resource_requirements(self, strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate resource requirements for implementing strategies"""
        resource_requirements = {
            'financial': {
                'low': 0,
                'medium': 0,
                'high': 0,
                'total_estimate': 0
            },
            'human': {
                'low': 0,
                'medium': 0,
                'high': 0,
                'total_fte': 0
            },
            'time': {
                'short_term': 0,
                'medium_term': 0,
                'long_term': 0,
                'total_months': 0
            }
        }
        
        # Count strategies by resource requirement level
        for strategy in strategies:
            resource_level = strategy.get('resource_requirements', 'Medium')
            timeframe = strategy.get('timeframe', 'Medium-term')
            
            # Financial resources
            if resource_level == 'High':
                resource_requirements['financial']['high'] += 1
            elif resource_level == 'Medium':
                resource_requirements['financial']['medium'] += 1
            else:
                resource_requirements['financial']['low'] += 1
            
            # Human resources (FTE)
            if resource_level == 'High':
                resource_requirements['human']['high'] += 1
            elif resource_level == 'Medium':
                resource_requirements['human']['medium'] += 1
            else:
                resource_requirements['human']['low'] += 1
            
            # Time resources
            if timeframe == 'Short-term':
                resource_requirements['time']['short_term'] += 1
            elif timeframe == 'Medium-term':
                resource_requirements['time']['medium_term'] += 1
            else:
                resource_requirements['time']['long_term'] += 1
        
        # Estimate total financial resources (arbitrary units)
        resource_requirements['financial']['total_estimate'] = (
            resource_requirements['financial']['low'] * 10000 +
            resource_requirements['financial']['medium'] * 50000 +
            resource_requirements['financial']['high'] * 200000
        )
        
        # Estimate total human resources (FTE)
        resource_requirements['human']['total_fte'] = (
            resource_requirements['human']['low'] * 0.2 +
            resource_requirements['human']['medium'] * 0.5 +
            resource_requirements['human']['high'] * 1.5
        )
        
        # Estimate total time (months)
        resource_requirements['time']['total_months'] = (
            resource_requirements['time']['short_term'] * 3 +
            resource_requirements['time']['medium_term'] * 9 +
            resource_requirements['time']['long_term'] * 18
        ) / len(strategies) if strategies else 0
        
        return resource_requirements
    
    def generate_custom_strategy(self, risk_type: str, risk_factors: List[Dict[str, Any]],
                               business_constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate custom mitigation strategy based on specific business constraints
        
        Args:
            risk_type: Type of risk (financial, market, operational)
            risk_factors: List of specific risk factors
            business_constraints: Dictionary containing business constraints
            
        Returns:
            Dictionary containing customized mitigation strategy
        """
        # Get base strategies
        base_strategies = []
        for risk_factor in risk_factors:
            factor_name = risk_factor.get('factor', '')
            severity = risk_factor.get('severity', 'medium')
            
            strategies = self._find_strategies(risk_type, factor_name)
            
            if strategies:
                for strategy in strategies:
                    strategy['risk_factor'] = factor_name
                    strategy['risk_severity'] = severity
                    base_strategies.append(strategy)
            else:
                # If no specific strategy found, provide generic strategy
                generic_strategy = self._create_generic_strategy(risk_type, factor_name, severity)
                base_strategies.append(generic_strategy)
        
        # Apply business constraints
        customized_strategies = self._apply_business_constraints(base_strategies, business_constraints)
        
        # Create implementation plan
        implementation_plan = self._create_implementation_plan(customized_strategies)
        
        # Calculate resource requirements
        resource_requirements = self._calculate_resource_requirements(customized_strategies)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'risk_type': risk_type,
            'strategies': customized_strategies,
            'implementation_plan': implementation_plan,
            'resource_requirements': resource_requirements,
            'applied_constraints': business_constraints
        }
    
    def _apply_business_constraints(self, strategies: List[Dict[str, Any]],
                                  business_constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply business constraints to strategies"""
        customized_strategies = []
        
        # Extract constraints
        resource_constraint = business_constraints.get('resource_constraint', 'none')  # none, low, high
        time_constraint = business_constraints.get('time_constraint', 'none')  # none, urgent, flexible
        focus_areas = business_constraints.get('focus_areas', [])  # list of areas to focus on
        
        for strategy in strategies:
            # Create a copy of the strategy to customize
            custom_strategy = strategy.copy()
            
            # Apply resource constraints
            if resource_constraint == 'low':
                # Reduce resource requirements
                if custom_strategy.get('resource_requirements') == 'High':
                    custom_strategy['resource_requirements'] = 'Medium'
                    # Adjust actions to reduce resource needs
                    custom_strategy['actions'] = [action for action in custom_strategy.get('actions', [])
                                               if not any(term in action.lower() for term in ['comprehensive', 'extensive', 'significant'])]
            elif resource_constraint == 'high':
                # Increase resource allocation for critical strategies
                if custom_strategy.get('risk_severity') == 'high':
                    custom_strategy['resource_requirements'] = 'High'
                    # Add more thorough actions
                    custom_strategy['actions'] = custom_strategy.get('actions', []) + [
                        "Allocate dedicated resources to implementation",
                        "Consider external expertise if needed"
                    ]
            
            # Apply time constraints
            if time_constraint == 'urgent':
                # Accelerate implementation timeframe
                if custom_strategy.get('timeframe') == 'Long-term':
                    custom_strategy['timeframe'] = 'Medium-term'
                elif custom_strategy.get('timeframe') == 'Medium-term':
                    custom_strategy['timeframe'] = 'Short-term'
                    
                # Adjust actions for urgency
                custom_strategy['actions'] = ["Accelerate: " + action for action in custom_strategy.get('actions', [])]
            elif time_constraint == 'flexible':
                # Extend timeframe for less critical strategies
                if custom_strategy.get('risk_severity') != 'high' and custom_strategy.get('timeframe') == 'Short-term':
                    custom_strategy['timeframe'] = 'Medium-term'
            
            # Apply focus areas
            if focus_areas:
                # Prioritize strategies that match focus areas
                strategy_text = json.dumps(custom_strategy).lower()
                if any(area.lower() in strategy_text for area in focus_areas):
                    custom_strategy['priority'] = 'High'
                    custom_strategy['note'] = f"Aligned with focus area: {', '.join(focus_areas)}"
            
            customized_strategies.append(custom_strategy)
        
        # Reprioritize strategies after customization
        customized_strategies = self._prioritize_strategies(customized_strategies)
        
        return customized_strategies