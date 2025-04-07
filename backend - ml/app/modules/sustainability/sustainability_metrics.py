# app/modules/sustainability/sustainability_metrics.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class SustainabilityMetrics:
    """
    Calculate and analyze sustainability metrics for business
    """
    
    def __init__(self):
        self.metrics = {}
        self.benchmarks = self._load_industry_benchmarks()
    
    def _load_industry_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """Load industry benchmark data"""
        # In a real implementation, this would load from a database or API
        # Using hardcoded values for demonstration
        return {
            'technology': {
                'carbon_intensity': 15.2,  # tCO2e per $1M revenue
                'energy_intensity': 120.5,  # MWh per $1M revenue
                'water_intensity': 250.3,  # m3 per $1M revenue
                'waste_recycling_rate': 72.5,  # percentage
                'renewable_energy': 45.8,  # percentage
                'gender_diversity': 38.2,  # percentage female
                'training_hours': 42.5  # hours per employee per year
            },
            'manufacturing': {
                'carbon_intensity': 85.3,
                'energy_intensity': 420.8,
                'water_intensity': 850.6,
                'waste_recycling_rate': 65.3,
                'renewable_energy': 28.4,
                'gender_diversity': 32.5,
                'training_hours': 35.2
            },
            'retail': {
                'carbon_intensity': 32.6,
                'energy_intensity': 210.3,
                'water_intensity': 180.4,
                'waste_recycling_rate': 68.7,
                'renewable_energy': 35.2,
                'gender_diversity': 52.3,
                'training_hours': 28.6
            },
            'financial_services': {
                'carbon_intensity': 8.5,
                'energy_intensity': 95.2,
                'water_intensity': 120.8,
                'waste_recycling_rate': 75.2,
                'renewable_energy': 42.5,
                'gender_diversity': 45.6,
                'training_hours': 45.8
            },
            'healthcare': {
                'carbon_intensity': 25.4,
                'energy_intensity': 185.6,
                'water_intensity': 450.2,
                'waste_recycling_rate': 58.5,
                'renewable_energy': 32.1,
                'gender_diversity': 65.3,
                'training_hours': 52.4
            },
            'default': {
                'carbon_intensity': 35.0,
                'energy_intensity': 200.0,
                'water_intensity': 350.0,
                'waste_recycling_rate': 65.0,
                'renewable_energy': 35.0,
                'gender_diversity': 40.0,
                'training_hours': 40.0
            }
        }
    
    def calculate_environmental_metrics(self, data: Dict[str, Any], 
                                       revenue: float,
                                       industry: str = 'default') -> Dict[str, Any]:
        """
        Calculate environmental sustainability metrics
        
        Args:
            data: Dictionary containing environmental data
            revenue: Annual revenue in dollars
            industry: Industry for benchmarking
            
        Returns:
            Dictionary containing calculated metrics
        """
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'carbon_emissions': {},
            'energy_usage': {},
            'water_usage': {},
            'waste_management': {},
            'benchmarks': {},
            'overall_rating': ""
        }
        
        # Get relevant benchmarks
        benchmarks = self.benchmarks.get(industry.lower(), self.benchmarks['default'])
        
        # Calculate carbon metrics
        if 'carbon_emissions' in data:
            carbon = data['carbon_emissions']
            total_emissions = carbon.get('total', 0)  # in tCO2e
            
            # Calculate intensity
            revenue_in_millions = revenue / 1000000
            carbon_intensity = total_emissions / revenue_in_millions if revenue_in_millions > 0 else 0
            
            metrics['carbon_emissions'] = {
                'total': total_emissions,
                'intensity': carbon_intensity,
                'intensity_unit': 'tCO2e per $1M revenue',
                'previous_year': carbon.get('previous_year', None),
                'by_scope': {
                    'scope1': carbon.get('scope1', 0),
                    'scope2': carbon.get('scope2', 0),
                    'scope3': carbon.get('scope3', 0)
                }
            }
            
            # Add benchmark comparison
            benchmark_intensity = benchmarks.get('carbon_intensity', 35.0)
            metrics['benchmarks']['carbon_intensity'] = {
                'value': benchmark_intensity,
                'comparison': 'better' if carbon_intensity < benchmark_intensity else 'worse',
                'percentage_difference': abs((carbon_intensity - benchmark_intensity) / benchmark_intensity * 100)
            }
        
        # Calculate energy metrics
        if 'energy_usage' in data:
            energy = data['energy_usage']
            total_energy = energy.get('total', 0)  # in MWh
            renewable = energy.get('renewable', 0)  # in MWh
            
            # Calculate intensity and renewable percentage
            energy_intensity = total_energy / revenue_in_millions if revenue_in_millions > 0 else 0
            renewable_percentage = (renewable / total_energy * 100) if total_energy > 0 else 0
            
            metrics['energy_usage'] = {
                'total': total_energy,
                'intensity': energy_intensity,
                'intensity_unit': 'MWh per $1M revenue',
                'renewable': renewable,
                'renewable_percentage': renewable_percentage
            }
            
            # Add benchmark comparisons
            benchmark_intensity = benchmarks.get('energy_intensity', 200.0)
            benchmark_renewable = benchmarks.get('renewable_energy', 35.0)
            
            metrics['benchmarks']['energy_intensity'] = {
                'value': benchmark_intensity,
                'comparison': 'better' if energy_intensity < benchmark_intensity else 'worse',
                'percentage_difference': abs((energy_intensity - benchmark_intensity) / benchmark_intensity * 100)
            }
            
            metrics['benchmarks']['renewable_energy'] = {
                'value': benchmark_renewable,
                'comparison': 'better' if renewable_percentage > benchmark_renewable else 'worse',
                'percentage_difference': abs((renewable_percentage - benchmark_renewable) / benchmark_renewable * 100)
            }
        
        # Calculate water metrics
        if 'water_usage' in data:
            water = data['water_usage']
            total_water = water.get('total', 0)  # in cubic meters
            
            # Calculate intensity
            water_intensity = total_water / revenue_in_millions if revenue_in_millions > 0 else 0
            
            metrics['water_usage'] = {
                'total': total_water,
                'intensity': water_intensity,
                'intensity_unit': 'm³ per $1M revenue',
                'recycled': water.get('recycled', 0),
                'sources': water.get('sources', {})
            }
            
            # Add benchmark comparison
            benchmark_intensity = benchmarks.get('water_intensity', 350.0)
            metrics['benchmarks']['water_intensity'] = {
                'value': benchmark_intensity,
                'comparison': 'better' if water_intensity < benchmark_intensity else 'worse',
                'percentage_difference': abs((water_intensity - benchmark_intensity) / benchmark_intensity * 100)
            }
        
        # Calculate waste metrics
        if 'waste' in data:
            waste = data['waste']
            total_waste = waste.get('total', 0)  # in metric tons
            recycled = waste.get('recycled', 0)  # in metric tons
            
            # Calculate intensity and recycling rate
            waste_intensity = total_waste / revenue_in_millions if revenue_in_millions > 0 else 0
            recycling_rate = (recycled / total_waste * 100) if total_waste > 0 else 0
            
            metrics['waste_management'] = {
                'total': total_waste,
                'intensity': waste_intensity,
                'intensity_unit': 'metric tons per $1M revenue',
                'recycled': recycled,
                'recycling_rate': recycling_rate,
                'landfill': waste.get('landfill', 0),
                'hazardous': waste.get('hazardous', 0)
            }
            
            # Add benchmark comparison
            benchmark_recycling = benchmarks.get('waste_recycling_rate', 65.0)
            metrics['benchmarks']['waste_recycling_rate'] = {
                'value': benchmark_recycling,
                'comparison': 'better' if recycling_rate > benchmark_recycling else 'worse',
                'percentage_difference': abs((recycling_rate - benchmark_recycling) / benchmark_recycling * 100)
            }
        
        # Calculate overall environmental rating
        better_count = sum(1 for metric in metrics['benchmarks'].values() if metric.get('comparison') == 'better')
        total_benchmarks = len(metrics['benchmarks'])
        
        if total_benchmarks > 0:
            better_percentage = better_count / total_benchmarks * 100
            
            if better_percentage >= 80:
                metrics['overall_rating'] = "Excellent - Industry leading environmental performance"
            elif better_percentage >= 60:
                metrics['overall_rating'] = "Good - Above average environmental performance"
            elif better_percentage >= 40:
                metrics['overall_rating'] = "Average - Comparable to industry benchmarks"
            elif better_percentage >= 20:
                metrics['overall_rating'] = "Below Average - Room for improvement in multiple areas"
            else:
                metrics['overall_rating'] = "Poor - Significant environmental performance gaps"
        else:
            metrics['overall_rating'] = "Insufficient data for rating"
        
        return metrics
    def calculate_social_metrics(self, data: Dict[str, Any],
                               industry: str = 'default') -> Dict[str, Any]:
        """
        Calculate social sustainability metrics
        
        Args:
            data: Dictionary containing social data
            industry: Industry for benchmarking
            
        Returns:
            Dictionary containing calculated metrics
        """
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'workforce': {},
            'community': {},
            'customer': {},
            'benchmarks': {},
            'overall_rating': ""
        }
        
        # Get relevant benchmarks
        benchmarks = self.benchmarks.get(industry.lower(), self.benchmarks['default'])
        
        # Calculate workforce metrics
        if 'workforce' in data:
            workforce = data['workforce']
            
            # Process diversity data
            diversity = workforce.get('diversity', {})
            gender_ratio = diversity.get('gender_ratio', {})
            
            female_percentage = gender_ratio.get('female', 0)
            
            # Process health and safety data
            health_safety = workforce.get('health_safety', {})
            incident_rate = health_safety.get('incident_rate', 0)
            
            # Process training data
            training = workforce.get('training', {})
            hours_per_employee = training.get('hours_per_employee', 0)
            
            # Process turnover data
            turnover = workforce.get('turnover', {})
            turnover_rate = turnover.get('rate', 0)
            
            metrics['workforce'] = {
                'headcount': workforce.get('headcount', 0),
                'diversity': {
                    'gender_ratio': gender_ratio,
                    'female_percentage': female_percentage,
                    'ethnic_diversity': diversity.get('ethnic_diversity', {}),
                    'age_distribution': diversity.get('age_distribution', {})
                },
                'health_safety': {
                    'incident_rate': incident_rate,
                    'lost_time_injuries': health_safety.get('lost_time_injuries', 0),
                    'fatalities': health_safety.get('fatalities', 0)
                },
                'training': {
                    'hours_per_employee': hours_per_employee,
                    'investment_per_employee': training.get('investment_per_employee', 0)
                },
                'turnover': {
                    'rate': turnover_rate,
                    'voluntary_rate': turnover.get('voluntary_rate', 0)
                }
            }
            
            # Add benchmark comparisons
            benchmark_gender = benchmarks.get('gender_diversity', 40.0)
            benchmark_training = benchmarks.get('training_hours', 40.0)
            
            metrics['benchmarks']['gender_diversity'] = {
                'value': benchmark_gender,
                'comparison': 'better' if female_percentage > benchmark_gender else 'worse',
                'percentage_difference': abs((female_percentage - benchmark_gender) / benchmark_gender * 100)
            }
            
            metrics['benchmarks']['training_hours'] = {
                'value': benchmark_training,
                'comparison': 'better' if hours_per_employee > benchmark_training else 'worse',
                'percentage_difference': abs((hours_per_employee - benchmark_training) / benchmark_training * 100)
            }
        
        # Calculate community metrics
        if 'community' in data:
            community = data['community']
            
            metrics['community'] = {
                'investment': community.get('investment', 0),
                'volunteer_hours': community.get('volunteer_hours', 0),
                'programs': community.get('programs', []),
                'beneficiaries': community.get('beneficiaries', 0)
            }
        
        # Calculate customer metrics
        if 'customer' in data:
            customer = data['customer']
            
            metrics['customer'] = {
                'satisfaction_score': customer.get('satisfaction_score', 0),
                'net_promoter_score': customer.get('net_promoter_score', 0),
                'privacy_incidents': customer.get('privacy_incidents', 0),
                'product_recalls': customer.get('product_recalls', 0)
            }
        
        # Calculate overall social rating
        better_count = sum(1 for metric in metrics['benchmarks'].values() if metric.get('comparison') == 'better')
        total_benchmarks = len(metrics['benchmarks'])
        
        if total_benchmarks > 0:
            better_percentage = better_count / total_benchmarks * 100
            
            if better_percentage >= 80:
                metrics['overall_rating'] = "Excellent - Industry leading social performance"
            elif better_percentage >= 60:
                metrics['overall_rating'] = "Good - Above average social performance"
            elif better_percentage >= 40:
                metrics['overall_rating'] = "Average - Comparable to industry benchmarks"
            elif better_percentage >= 20:
                metrics['overall_rating'] = "Below Average - Room for improvement in multiple areas"
            else:
                metrics['overall_rating'] = "Poor - Significant social performance gaps"
        else:
            metrics['overall_rating'] = "Insufficient data for rating"
        
        return metrics
    
    def calculate_governance_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate governance sustainability metrics
        
        Args:
            data: Dictionary containing governance data
            
        Returns:
            Dictionary containing calculated metrics
        """
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'board': {},
            'ethics': {},
            'risk_management': {},
            'transparency': {},
            'overall_rating': ""
        }
        
        # Calculate board metrics
        if 'board' in data:
            board = data['board']
            
            # Calculate board diversity
            diversity = board.get('diversity', {})
            female_percentage = diversity.get('female_percentage', 0)
            independent_percentage = board.get('independent_percentage', 0)
            
            metrics['board'] = {
                'size': board.get('size', 0),
                'independence': {
                    'independent_directors': board.get('independent_directors', 0),
                    'independent_percentage': independent_percentage
                },
                'diversity': {
                    'female_directors': diversity.get('female_directors', 0),
                    'female_percentage': female_percentage,
                    'ethnic_diversity': diversity.get('ethnic_diversity', {})
                },
                'committees': board.get('committees', []),
                'expertise': board.get('expertise', {})
            }
        
        # Calculate ethics metrics
        if 'ethics' in data:
            ethics = data['ethics']
            
            metrics['ethics'] = {
                'code_of_conduct': ethics.get('code_of_conduct', False),
                'ethics_training': ethics.get('ethics_training', False),
                'reporting_mechanism': ethics.get('reporting_mechanism', False),
                'incidents': {
                    'total': ethics.get('incidents', {}).get('total', 0),
                    'resolved': ethics.get('incidents', {}).get('resolved', 0)
                },
                'anti_corruption': ethics.get('anti_corruption', False)
            }
        
        # Calculate risk management metrics
        if 'risk_management' in data:
            risk = data['risk_management']
            
            metrics['risk_management'] = {
                'framework': risk.get('framework', False),
                'assessment_frequency': risk.get('assessment_frequency', 'annual'),
                'board_oversight': risk.get('board_oversight', False),
                'esg_risks_integrated': risk.get('esg_risks_integrated', False)
            }
        
        # Calculate transparency metrics
        if 'transparency' in data:
            transparency = data['transparency']
            
            metrics['transparency'] = {
                'sustainability_reporting': transparency.get('sustainability_reporting', False),
                'financial_reporting': transparency.get('financial_reporting', False),
                'tax_transparency': transparency.get('tax_transparency', False),
                'political_contributions': transparency.get('political_contributions', False)
            }
        
        # Calculate overall governance rating
        rating_points = 0
        max_points = 0
        
        # Board rating factors
        if 'board' in metrics:
            max_points += 3
            if metrics['board'].get('independence', {}).get('independent_percentage', 0) >= 75:
                rating_points += 1
            if metrics['board'].get('diversity', {}).get('female_percentage', 0) >= 40:
                rating_points += 1
            if len(metrics['board'].get('committees', [])) >= 3:
                rating_points += 1
        
        # Ethics rating factors
        if 'ethics' in metrics:
            max_points += 3
            if metrics['ethics'].get('code_of_conduct', False):
                rating_points += 1
            if metrics['ethics'].get('ethics_training', False):
                rating_points += 1
            if metrics['ethics'].get('reporting_mechanism', False):
                rating_points += 1
        
        # Risk management rating factors
        if 'risk_management' in metrics:
            max_points += 2
            if metrics['risk_management'].get('framework', False):
                rating_points += 1
            if metrics['risk_management'].get('esg_risks_integrated', False):
                rating_points += 1
        
        # Transparency rating factors
        if 'transparency' in metrics:
            max_points += 2
            if metrics['transparency'].get('sustainability_reporting', False):
                rating_points += 1
            if metrics['transparency'].get('tax_transparency', False):
                rating_points += 1
        
        # Calculate overall rating
        if max_points > 0:
            rating_percentage = rating_points / max_points * 100
            
            if rating_percentage >= 80:
                metrics['overall_rating'] = "Excellent - Strong governance practices"
            elif rating_percentage >= 60:
                metrics['overall_rating'] = "Good - Solid governance framework with some areas for improvement"
            elif rating_percentage >= 40:
                metrics['overall_rating'] = "Average - Basic governance practices in place"
            elif rating_percentage >= 20:
                metrics['overall_rating'] = "Below Average - Significant governance gaps"
            else:
                metrics['overall_rating'] = "Poor - Inadequate governance framework"
        else:
            metrics['overall_rating'] = "Insufficient data for rating"
        
        return metrics
    
    def calculate_esg_score(self, environmental_metrics: Dict[str, Any],
                          social_metrics: Dict[str, Any],
                          governance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate overall ESG score based on environmental, social, and governance metrics
        
        Args:
            environmental_metrics: Dictionary containing environmental metrics
            social_metrics: Dictionary containing social metrics
            governance_metrics: Dictionary containing governance metrics
            
        Returns:
            Dictionary containing ESG score and breakdown
        """
        esg_score = {
            'timestamp': datetime.now().isoformat(),
            'overall_score': 0,
            'environmental_score': 0,
            'social_score': 0,
            'governance_score': 0,
            'component_breakdown': {},
            'rating': "",
            'peer_comparison': "",
            'improvement_areas': []
        }
        
        # Calculate environmental score (0-100)
        env_score = 0
        env_components = {}
        
        # Carbon emissions component (0-25 points)
        if 'carbon_emissions' in environmental_metrics and 'benchmarks' in environmental_metrics:
            carbon_benchmark = environmental_metrics['benchmarks'].get('carbon_intensity', {})
            if carbon_benchmark.get('comparison') == 'better':
                # Better than benchmark
                diff_pct = carbon_benchmark.get('percentage_difference', 0)
                if diff_pct > 50:
                    env_components['carbon'] = 25  # Significantly better
                elif diff_pct > 20:
                    env_components['carbon'] = 20  # Moderately better
                else:
                    env_components['carbon'] = 15  # Slightly better
            else:
                # Worse than benchmark
                diff_pct = carbon_benchmark.get('percentage_difference', 0)
                if diff_pct > 50:
                    env_components['carbon'] = 5  # Significantly worse
                elif diff_pct > 20:
                    env_components['carbon'] = 10  # Moderately worse
                else:
                    env_components['carbon'] = 12  # Slightly worse
        else:
            env_components['carbon'] = 0  # No data
        
        # Energy component (0-25 points)
        if 'energy_usage' in environmental_metrics and 'benchmarks' in environmental_metrics:
            energy_benchmark = environmental_metrics['benchmarks'].get('energy_intensity', {})
            renewable_benchmark = environmental_metrics['benchmarks'].get('renewable_energy', {})
            
            energy_score = 0
            if energy_benchmark.get('comparison') == 'better':
                energy_score += 10
            else:
                energy_score += 5
                
            if renewable_benchmark.get('comparison') == 'better':
                energy_score += 15
            else:
                energy_score += 7
                
            env_components['energy'] = energy_score
        else:
            env_components['energy'] = 0  # No data
        
        # Waste component (0-25 points)
        if 'waste_management' in environmental_metrics and 'benchmarks' in environmental_metrics:
            waste_benchmark = environmental_metrics['benchmarks'].get('waste_recycling_rate', {})
            
            if waste_benchmark.get('comparison') == 'better':
                # Better than benchmark
                diff_pct = waste_benchmark.get('percentage_difference', 0)
                if diff_pct > 30:
                    env_components['waste'] = 25  # Significantly better
                elif diff_pct > 10:
                    env_components['waste'] = 20  # Moderately better
                else:
                    env_components['waste'] = 15  # Slightly better
            else:
                # Worse than benchmark
                diff_pct = waste_benchmark.get('percentage_difference', 0)
                if diff_pct > 30:
                    env_components['waste'] = 5  # Significantly worse
                elif diff_pct > 10:
                    env_components['waste'] = 10  # Moderately worse
                else:
                    env_components['waste'] = 12  # Slightly worse
        else:
            env_components['waste'] = 0  # No data
        
        # Water component (0-25 points)
        if 'water_usage' in environmental_metrics and 'benchmarks' in environmental_metrics:
            water_benchmark = environmental_metrics['benchmarks'].get('water_intensity', {})
            
            if water_benchmark.get('comparison') == 'better':
                # Better than benchmark
                diff_pct = water_benchmark.get('percentage_difference', 0)
                if diff_pct > 30:
                    env_components['water'] = 25  # Significantly better
                elif diff_pct > 10:
                    env_components['water'] = 20  # Moderately better
                else:
                    env_components['water'] = 15  # Slightly better
            else:
                # Worse than benchmark
                diff_pct = water_benchmark.get('percentage_difference', 0)
                if diff_pct > 30:
                    env_components['water'] = 5  # Significantly worse
                elif diff_pct > 10:
                    env_components['water'] = 10  # Moderately worse
                else:
                    env_components['water'] = 12  # Slightly worse
        else:
            env_components['water'] = 0  # No data
        
        # Calculate total environmental score
        component_count = sum(1 for v in env_components.values() if v > 0)
        if component_count > 0:
            env_score = sum(env_components.values()) / component_count * (100/25)  # Scale to 0-100
        
        esg_score['environmental_score'] = env_score
        esg_score['component_breakdown']['environmental'] = env_components
        
        # Calculate social score (0-100)
        social_score = 0
        social_components = {}
        
        # Diversity component (0-25 points)
        if 'workforce' in social_metrics and 'benchmarks' in social_metrics:
            gender_benchmark = social_metrics['benchmarks'].get('gender_diversity', {})
            
            if gender_benchmark.get('comparison') == 'better':
                # Better than benchmark
                diff_pct = gender_benchmark.get('percentage_difference', 0)
                if diff_pct > 30:
                    social_components['diversity'] = 25  # Significantly better
                elif diff_pct > 10:
                    social_components['diversity'] = 20  # Moderately better
                else:
                    social_components['diversity'] = 15  # Slightly better
            else:
                # Worse than benchmark
                diff_pct = gender_benchmark.get('percentage_difference', 0)
                if diff_pct > 30:
                    social_components['diversity'] = 5  # Significantly worse
                elif diff_pct > 10:
                    social_components['diversity'] = 10  # Moderately worse
                else:
                    social_components['diversity'] = 12  # Slightly worse
        else:
            social_components['diversity'] = 0  # No data
        
        # Training component (0-25 points)
        if 'workforce' in social_metrics and 'benchmarks' in social_metrics:
            training_benchmark = social_metrics['benchmarks'].get('training_hours', {})
            
            if training_benchmark.get('comparison') == 'better':
                # Better than benchmark
                diff_pct = training_benchmark.get('percentage_difference', 0)
                if diff_pct > 30:
                    social_components['training'] = 25  # Significantly better
                elif diff_pct > 10:
                    social_components['training'] = 20  # Moderately better
                else:
                    social_components['training'] = 15  # Slightly better
            else:
                # Worse than benchmark
                diff_pct = training_benchmark.get('percentage_difference', 0)
                if diff_pct > 30:
                    social_components['training'] = 5  # Significantly worse
                elif diff_pct > 10:
                    social_components['training'] = 10  # Moderately worse
                else:
                    social_components['training'] = 12  # Slightly worse
        else:
            social_components['training'] = 0  # No data
        
        # Health & Safety component (0-25 points)
        if 'workforce' in social_metrics and 'workforce' in social_metrics.get('health_safety', {}):
            incident_rate = social_metrics['workforce']['health_safety'].get('incident_rate', 0)
            
            if incident_rate == 0:
                social_components['health_safety'] = 25  # Perfect record
            elif incident_rate < 1:
                social_components['health_safety'] = 20  # Very good
            elif incident_rate < 3:
                social_components['health_safety'] = 15  # Good
            elif incident_rate < 5:
                social_components['health_safety'] = 10  # Average
            else:
                social_components['health_safety'] = 5  # Poor
        else:
            social_components['health_safety'] = 0  # No data
        
        # Community component (0-25 points)
        if 'community' in social_metrics:
            community = social_metrics['community']
            community_score = 0
            
            # Investment
            if community.get('investment', 0) > 0:
                community_score += 10
            
            # Volunteer hours
            if community.get('volunteer_hours', 0) > 0:
                community_score += 10
            
            # Programs
            if len(community.get('programs', [])) > 0:
                community_score += 5
                
            social_components['community'] = community_score
        else:
            social_components['community'] = 0  # No data
        
        # Calculate total social score
        component_count = sum(1 for v in social_components.values() if v > 0)
        if component_count > 0:
            social_score = sum(social_components.values()) / component_count * (100/25)  # Scale to 0-100
        
        esg_score['social_score'] = social_score
        esg_score['component_breakdown']['social'] = social_components
        
        # Calculate governance score (0-100)
        governance_score = 0
        governance_components = {}
        
        # Board component (0-25 points)
        if 'board' in governance_metrics:
            board = governance_metrics['board']
            board_score = 0
            
            # Independence
            independent_pct = board.get('independence', {}).get('independent_percentage', 0)
            if independent_pct >= 75:
                board_score += 10
            elif independent_pct >= 50:
                board_score += 5
            
            # Diversity
            female_pct = board.get('diversity', {}).get('female_percentage', 0)
            if female_pct >= 40:
                board_score += 10
            elif female_pct >= 30:
                board_score += 5
            
            # Committees
            if len(board.get('committees', [])) >= 3:
                board_score += 5
                
            governance_components['board'] = board_score
        else:
            governance_components['board'] = 0  # No data
        
        # Ethics component (0-25 points)
        if 'ethics' in governance_metrics:
            ethics = governance_metrics['ethics']
            ethics_score = 0
            
            # Code of conduct
            if ethics.get('code_of_conduct', False):
                ethics_score += 8
            
            # Ethics training
            if ethics.get('ethics_training', False):
                ethics_score += 8
            
            # Reporting mechanism
            if ethics.get('reporting_mechanism', False):
                ethics_score += 9
                
            governance_components['ethics'] = ethics_score
        else:
            governance_components['ethics'] = 0  # No data
        
        # Risk management component (0-25 points)
        if 'risk_management' in governance_metrics:
            risk = governance_metrics['risk_management']
            risk_score = 0
            
            # Framework
            if risk.get('framework', False):
                risk_score += 10
            
            # Board oversight
            if risk.get('board_oversight', False):
                risk_score += 8
            
            # ESG risks integrated
            if risk.get('esg_risks_integrated', False):
                risk_score += 7
                
            governance_components['risk_management'] = risk_score
        else:
            governance_components['risk_management'] = 0  # No data
        
        # Transparency component (0-25 points)
        if 'transparency' in governance_metrics:
            transparency = governance_metrics['transparency']
            transparency_score = 0
            
            # Sustainability reporting
            if transparency.get('sustainability_reporting', False):
                transparency_score += 8
            
            # Financial reporting
            if transparency.get('financial_reporting', False):
                transparency_score += 6
            
            # Tax transparency
            if transparency.get('tax_transparency', False):
                transparency_score += 6
            
            # Political contributions
            if transparency.get('political_contributions', False):
                transparency_score += 5
                
            governance_components['transparency'] = transparency_score
        else:
            governance_components['transparency'] = 0  # No data
        
        # Calculate total governance score
        component_count = sum(1 for v in governance_components.values() if v > 0)
        if component_count > 0:
            governance_score = sum(governance_components.values()) / component_count * (100/25)  # Scale to 0-100
        
        esg_score['governance_score'] = governance_score
        esg_score['component_breakdown']['governance'] = governance_components
        
        # Calculate overall ESG score (weighted average)
        # Environmental: 30%, Social: 30%, Governance: 40%
        has_env = esg_score['environmental_score'] > 0
        has_social = esg_score['social_score'] > 0
        has_gov = esg_score['governance_score'] > 0
        
        if has_env and has_social and has_gov:
            esg_score['overall_score'] = (
                esg_score['environmental_score'] * 0.3 +
                esg_score['social_score'] * 0.3 +
                esg_score['governance_score'] * 0.4
            )
        elif has_env and has_social:
            esg_score['overall_score'] = (
                esg_score['environmental_score'] * 0.5 +
                esg_score['social_score'] * 0.5
            )
        elif has_env and has_gov:
            esg_score['overall_score'] = (
                esg_score['environmental_score'] * 0.4 +
                esg_score['governance_score'] * 0.6
            )
        elif has_social and has_gov:
            esg_score['overall_score'] = (
                esg_score['social_score'] * 0.4 +
                esg_score['governance_score'] * 0.6
            )
        elif has_env:
            esg_score['overall_score'] = esg_score['environmental_score']
        elif has_social:
            esg_score['overall_score'] = esg_score['social_score']
        elif has_gov:
            esg_score['overall_score'] = esg_score['governance_score']
        
        # Determine ESG rating
        overall_score = esg_score['overall_score']
        if overall_score >= 85:
            esg_score['rating'] = "AAA - Industry Leader"
            esg_score['peer_comparison'] = "Top 10% of industry peers"
        elif overall_score >= 70:
            esg_score['rating'] = "AA - Strong Performer"
            esg_score['peer_comparison'] = "Top 25% of industry peers"
        elif overall_score >= 60:
            esg_score['rating'] = "A - Good Performer"
            esg_score['peer_comparison'] = "Above average compared to industry peers"
        elif overall_score >= 50:
            esg_score['rating'] = "BBB - Average Performer"
            esg_score['peer_comparison'] = "Average compared to industry peers"
        elif overall_score >= 40:
            esg_score['rating'] = "BB - Below Average"
            esg_score['peer_comparison'] = "Below average compared to industry peers"
        elif overall_score >= 30:
            esg_score['rating'] = "B - Poor Performer"
            esg_score['peer_comparison'] = "Bottom 25% of industry peers"
        else:
            esg_score['rating'] = "CCC - Very Poor Performer"
            esg_score['peer_comparison'] = "Bottom 10% of industry peers"
        
        # Identify improvement areas
        improvement_areas = []
        
        # Environmental improvement areas
        if env_components.get('carbon', 0) < 15:
            improvement_areas.append("Carbon emissions management")
        if env_components.get('energy', 0) < 15:
            improvement_areas.append("Energy efficiency and renewable energy")
        if env_components.get('waste', 0) < 15:
            improvement_areas.append("Waste management and recycling")
        if env_components.get('water', 0) < 15:
            improvement_areas.append("Water conservation")
        
        # Social improvement areas
        if social_components.get('diversity', 0) < 15:
            improvement_areas.append("Workforce diversity and inclusion")
        if social_components.get('training', 0) < 15:
            improvement_areas.append("Employee training and development")
        if social_components.get('health_safety', 0) < 15:
            improvement_areas.append("Workplace health and safety")
        if social_components.get('community', 0) < 15:
            improvement_areas.append("Community engagement and investment")
        
        # Governance improvement areas
        if governance_components.get('board', 0) < 15:
            improvement_areas.append("Board structure and diversity")
        if governance_components.get('ethics', 0) < 15:
            improvement_areas.append("Ethics policies and programs")
        if governance_components.get('risk_management', 0) < 15:
            improvement_areas.append("Risk management framework")
        if governance_components.get('transparency', 0) < 15:
            improvement_areas.append("Transparency and disclosure")
        
        esg_score['improvement_areas'] = improvement_areas
        
        return esg_score