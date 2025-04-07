# app/modules/sustainability/ethical_analyzer.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
import os
import openai
from datetime import datetime

logger = logging.getLogger(__name__)

class EthicalAnalyzer:
    """
    Analyze business practices for ethical considerations
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            logger.warning("No OpenAI API key provided. Ethical analysis will have limited functionality.")
    
    def analyze_business_practices(self, business_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze business practices for ethical considerations
        
        Args:
            business_data: Dictionary containing business practices data
            
        Returns:
            Dictionary containing ethical analysis results
        """
        # Prepare the analysis
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'ethical_considerations': [],
            'risk_areas': [],
            'recommendations': [],
            'overall_assessment': "",
            'esg_factors': {
                'environmental': [],
                'social': [],
                'governance': []
            }
        }
        
        # Extract relevant data
        industry = business_data.get('industry', 'Unknown')
        practices = business_data.get('practices', {})
        policies = business_data.get('policies', {})
        supply_chain = business_data.get('supply_chain', {})
        
        # If API key available, use AI for analysis
        if self.api_key:
            return self._analyze_with_ai(business_data)
        
        # Fallback to rule-based analysis
        return self._rule_based_analysis(business_data)
    
    def _analyze_with_ai(self, business_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform ethical analysis using OpenAI API"""
        try:
            # Prepare the context from business data
            context = json.dumps(business_data, indent=2)
            
            # Construct the prompt
            prompt = f"""
            Analyze the following business data for ethical considerations, risks, and ESG (Environmental, Social, Governance) factors:
            
            {context}
            
            Format your response as a JSON object with the following structure:
            {{
                "ethical_considerations": [
                    {{"area": "Area name", "description": "Description of ethical consideration", "severity": "High/Medium/Low"}}
                ],
                "risk_areas": [
                    {{"area": "Risk area", "description": "Description of risk", "likelihood": "High/Medium/Low", "impact": "High/Medium/Low"}}
                ],
                "recommendations": [
                    {{"recommendation": "Recommendation text", "priority": "High/Medium/Low", "implementation_complexity": "High/Medium/Low"}}
                ],
                "overall_assessment": "Overall ethical assessment of the business practices",
                "esg_factors": {{
                    "environmental": [
                        {{"factor": "Environmental factor", "status": "Strength/Weakness/Neutral", "description": "Description"}}
                    ],
                    "social": [
                        {{"factor": "Social factor", "status": "Strength/Weakness/Neutral", "description": "Description"}}
                    ],
                    "governance": [
                        {{"factor": "Governance factor", "status": "Strength/Weakness/Neutral", "description": "Description"}}
                    ]
                }}
            }}
            
            Provide a comprehensive analysis focusing on practical ethical considerations and actionable recommendations.
            """
            
            # Call OpenAI API
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert in business ethics and ESG analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2500
            )
            
            # Extract and parse the response
            response_text = response.choices[0].message.content
            
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'({[\s\S]*})', response_text)
            if json_match:
                json_str = json_match.group(1)
                analysis = json.loads(json_str)
            else:
                raise ValueError("Could not extract valid JSON from API response")
            
            # Add timestamp
            analysis['timestamp'] = datetime.now().isoformat()
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in AI ethical analysis: {str(e)}")
            # Fallback to rule-based analysis
            return self._rule_based_analysis(business_data)
    
    def _rule_based_analysis(self, business_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform rule-based ethical analysis as fallback"""
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'ethical_considerations': [],
            'risk_areas': [],
            'recommendations': [],
            'overall_assessment': "",
            'esg_factors': {
                'environmental': [],
                'social': [],
                'governance': []
            }
        }
        
        # Extract relevant data
        industry = business_data.get('industry', 'Unknown')
        practices = business_data.get('practices', {})
        policies = business_data.get('policies', {})
        supply_chain = business_data.get('supply_chain', {})
        
        # Environmental considerations
        environmental_policies = policies.get('environmental', {})
        
        # Check for environmental policy
        if not environmental_policies:
            analysis['ethical_considerations'].append({
                'area': 'Environmental Policy',
                'description': 'No formal environmental policy identified',
                'severity': 'Medium'
            })
            
            analysis['recommendations'].append({
                'recommendation': 'Develop and implement a formal environmental policy',
                'priority': 'Medium',
                'implementation_complexity': 'Medium'
            })
            
            analysis['esg_factors']['environmental'].append({
                'factor': 'Environmental Policy',
                'status': 'Weakness',
                'description': 'No formal environmental policy in place'
            })
        else:
            analysis['esg_factors']['environmental'].append({
                'factor': 'Environmental Policy',
                'status': 'Strength',
                'description': 'Formal environmental policy in place'
            })
        
        # Check for carbon footprint tracking
        carbon_tracking = practices.get('carbon_tracking', False)
        if not carbon_tracking:
            analysis['ethical_considerations'].append({
                'area': 'Carbon Emissions',
                'description': 'No carbon footprint tracking identified',
                'severity': 'Medium'
            })
            
            analysis['recommendations'].append({
                'recommendation': 'Implement carbon footprint tracking and reduction goals',
                'priority': 'Medium',
                'implementation_complexity': 'Medium'
            })
            
            analysis['esg_factors']['environmental'].append({
                'factor': 'Carbon Management',
                'status': 'Weakness',
                'description': 'No carbon tracking or management system'
            })
        else:
            analysis['esg_factors']['environmental'].append({
                'factor': 'Carbon Management',
                'status': 'Strength',
                'description': 'Active carbon tracking and management'
            })
        
        # Social considerations
        labor_practices = practices.get('labor', {})
        diversity_inclusion = practices.get('diversity_inclusion', {})
        
        # Check for labor policies
        if not labor_practices:
            analysis['ethical_considerations'].append({
                'area': 'Labor Practices',
                'description': 'Limited information on labor practices',
                'severity': 'High'
            })
            
            analysis['risk_areas'].append({
                'area': 'Labor Compliance',
                'description': 'Potential labor compliance risks due to limited documentation',
                'likelihood': 'Medium',
                'impact': 'High'
            })
            
            analysis['recommendations'].append({
                'recommendation': 'Develop comprehensive labor policies and compliance monitoring',
                'priority': 'High',
                'implementation_complexity': 'Medium'
            })
            
            analysis['esg_factors']['social'].append({
                'factor': 'Labor Practices',
                'status': 'Weakness',
                'description': 'Limited labor practice documentation'
            })
        else:
            analysis['esg_factors']['social'].append({
                'factor': 'Labor Practices',
                'status': 'Neutral',
                'description': 'Basic labor practices documented'
            })
        
        # Check for diversity and inclusion
        if not diversity_inclusion:
            analysis['ethical_considerations'].append({
                'area': 'Diversity and Inclusion',
                'description': 'Limited focus on diversity and inclusion',
                'severity': 'Medium'
            })
            
            analysis['recommendations'].append({
                'recommendation': 'Develop diversity and inclusion initiatives',
                'priority': 'Medium',
                'implementation_complexity': 'Medium'
            })
            
            analysis['esg_factors']['social'].append({
                'factor': 'Diversity & Inclusion',
                'status': 'Weakness',
                'description': 'Limited diversity and inclusion focus'
            })
        else:
            has_targets = diversity_inclusion.get('targets', False)
            has_programs = diversity_inclusion.get('programs', False)
            
            if has_targets and has_programs:
                status = 'Strength'
                description = 'Comprehensive diversity and inclusion program with targets'
            elif has_targets or has_programs:
                status = 'Neutral'
                description = 'Some diversity and inclusion initiatives in place'
            else:
                status = 'Weakness'
                description = 'Basic diversity and inclusion acknowledgment without specific programs'
            
            analysis['esg_factors']['social'].append({
                'factor': 'Diversity & Inclusion',
                'status': status,
                'description': description
            })
        
        # Governance considerations
        governance = practices.get('governance', {})
        
        # Check for ethics policy
        ethics_policy = policies.get('ethics', False)
        if not ethics_policy:
            analysis['ethical_considerations'].append({
                'area': 'Ethics Policy',
                'description': 'No formal ethics policy identified',
                'severity': 'High'
            })
            
            analysis['risk_areas'].append({
                'area': 'Ethical Conduct',
                'description': 'Increased risk of ethical issues without formal guidance',
                'likelihood': 'Medium',
                'impact': 'High'
            })
            
            analysis['recommendations'].append({
                'recommendation': 'Develop and implement a comprehensive ethics policy',
                'priority': 'High',
                'implementation_complexity': 'Medium'
            })
            
            analysis['esg_factors']['governance'].append({
                'factor': 'Ethics Policy',
                'status': 'Weakness',
                'description': 'No formal ethics policy'
            })
        else:
            analysis['esg_factors']['governance'].append({
                'factor': 'Ethics Policy',
                'status': 'Strength',
                'description': 'Formal ethics policy in place'
            })
        
        # Supply chain considerations
        if supply_chain:
            supplier_code = supply_chain.get('supplier_code_of_conduct', False)
            supplier_audits = supply_chain.get('supplier_audits', False)
            
            if not supplier_code:
                analysis['ethical_considerations'].append({
                    'area': 'Supply Chain Ethics',
                    'description': 'No supplier code of conduct identified',
                    'severity': 'High'
                })
                
                analysis['risk_areas'].append({
                    'area': 'Supply Chain',
                    'description': 'Potential ethical risks in supply chain without formal standards',
                    'likelihood': 'High',
                    'impact': 'High'
                })
                
                analysis['recommendations'].append({
                    'recommendation': 'Develop and implement a supplier code of conduct',
                    'priority': 'High',
                    'implementation_complexity': 'Medium'
                })
                
                analysis['esg_factors']['governance'].append({
                    'factor': 'Supply Chain Management',
                    'status': 'Weakness',
                    'description': 'Limited supply chain ethical oversight'
                })
            else:
                if supplier_audits:
                    analysis['esg_factors']['governance'].append({
                        'factor': 'Supply Chain Management',
                        'status': 'Strength',
                        'description': 'Comprehensive supply chain management with code of conduct and audits'
                    })
                else:
                    analysis['esg_factors']['governance'].append({
                        'factor': 'Supply Chain Management',
                        'status': 'Neutral',
                        'description': 'Supply chain code of conduct without regular audits'
                    })
                    
                    analysis['recommendations'].append({
                        'recommendation': 'Implement regular supplier audits to verify compliance',
                        'priority': 'Medium',
                        'implementation_complexity': 'High'
                    })
        
        # Generate overall assessment
        strengths = sum(1 for category in analysis['esg_factors'].values() 
                      for factor in category if factor.get('status') == 'Strength')
        weaknesses = sum(1 for category in analysis['esg_factors'].values() 
                       for factor in category if factor.get('status') == 'Weakness')
        
        if weaknesses == 0:
            analysis['overall_assessment'] = "Strong ethical practices with comprehensive ESG considerations"
        elif strengths > weaknesses:
            analysis['overall_assessment'] = "Generally sound ethical practices with some areas for improvement"
        elif strengths == weaknesses:
            analysis['overall_assessment'] = "Mixed ethical practices with significant room for improvement"
        else:
            analysis['overall_assessment'] = "Substantial ethical considerations requiring immediate attention"
        
        return analysis
    
    def analyze_supply_chain(self, supply_chain_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze supply chain for ethical considerations
        
        Args:
            supply_chain_data: Dictionary containing supply chain data
            
        Returns:
            Dictionary containing ethical analysis results
        """
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'risk_areas': [],
            'recommendations': [],
            'supplier_assessment': [],
            'overall_risk_level': "",
            'priority_actions': []
        }
        
        # If API key available, use AI for analysis
        if self.api_key:
            return self._analyze_supply_chain_with_ai(supply_chain_data)
        
        # Fallback to rule-based analysis
        return self._analyze_supply_chain_rule_based(supply_chain_data)
    
    def _analyze_supply_chain_with_ai(self, supply_chain_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze supply chain using OpenAI API"""
        try:
            # Prepare the context from supply chain data
            context = json.dumps(supply_chain_data, indent=2)
            
            # Construct the prompt
            prompt = f"""
            Analyze the following supply chain data for ethical considerations, risks, and recommendations:
            
            {context}
            
            Format your response as a JSON object with the following structure:
            {{
                "risk_areas": [
                    {{"area": "Risk area", "description": "Description of risk", "likelihood": "High/Medium/Low", "impact": "High/Medium/Low"}}
                ],
                "recommendations": [
                    {{"recommendation": "Recommendation text", "priority": "High/Medium/Low", "implementation_complexity": "High/Medium/Low"}}
                ],
                "supplier_assessment": [
                    {{"supplier_type": "Type of supplier", "risk_level": "High/Medium/Low", "key_concerns": ["Concern 1", "Concern 2"]}}
                ],
                "overall_risk_level": "Overall risk assessment of the supply chain",
                "priority_actions": ["Action 1", "Action 2", "Action 3"]
            }}
            
            Focus on practical ethical considerations in the supply chain including labor practices, environmental impact, transparency, and compliance with regulations.
            """
            
            # Call OpenAI API
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert in supply chain ethics and sustainability."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            # Extract and parse the response
            response_text = response.choices[0].message.content
            
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'({[\s\S]*})', response_text)
            if json_match:
                json_str = json_match.group(1)
                analysis = json.loads(json_str)
            else:
                raise ValueError("Could not extract valid JSON from API response")
            
            # Add timestamp
            analysis['timestamp'] = datetime.now().isoformat()
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in AI supply chain analysis: {str(e)}")
            # Fallback to rule-based analysis
            return self._analyze_supply_chain_rule_based(supply_chain_data)
    
    def _analyze_supply_chain_rule_based(self, supply_chain_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform rule-based supply chain ethical analysis as fallback"""
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'risk_areas': [],
            'recommendations': [],
            'supplier_assessment': [],
            'overall_risk_level': "",
            'priority_actions': []
        }
        
        # Extract relevant data
        suppliers = supply_chain_data.get('suppliers', [])
        regions = supply_chain_data.get('regions', [])
        practices = supply_chain_data.get('practices', {})
        
        # Check for supplier code of conduct
        supplier_code = practices.get('supplier_code_of_conduct', False)
        if not supplier_code:
            analysis['risk_areas'].append({
                'area': 'Supplier Standards',
                'description': 'No formal supplier code of conduct',
                'likelihood': 'High',
                'impact': 'High'
            })
            
            analysis['recommendations'].append({
                'recommendation': 'Develop and implement a comprehensive supplier code of conduct',
                'priority': 'High',
                'implementation_complexity': 'Medium'
            })
            
            analysis['priority_actions'].append(
                "Establish supplier code of conduct with clear ethical requirements"
            )
        
        # Check for supplier audits
        supplier_audits = practices.get('supplier_audits', False)
        if not supplier_audits:
            analysis['risk_areas'].append({
                'area': 'Supplier Verification',
                'description': 'No regular supplier audits or verification process',
                'likelihood': 'High',
                'impact': 'High'
            })
            
            analysis['recommendations'].append({
                'recommendation': 'Implement regular supplier audits and verification process',
                'priority': 'High',
                'implementation_complexity': 'High'
            })
            
            analysis['priority_actions'].append(
                "Develop supplier audit program to verify compliance with standards"
            )
        
        # Check for transparency
        transparency = practices.get('transparency', {})
        if not transparency or not transparency.get('public_reporting', False):
            analysis['risk_areas'].append({
                'area': 'Supply Chain Transparency',
                'description': 'Limited public transparency about supply chain practices',
                'likelihood': 'Medium',
                'impact': 'Medium'
            })
            
            analysis['recommendations'].append({
                'recommendation': 'Increase supply chain transparency through public reporting',
                'priority': 'Medium',
                'implementation_complexity': 'Medium'
            })
        
        # Assess suppliers by region
        high_risk_regions = ['Southeast Asia', 'South Asia', 'Africa', 'Central America']
        medium_risk_regions = ['Eastern Europe', 'South America', 'Middle East']
        
        for region in regions:
            region_name = region.get('name', '')
            
            risk_level = 'Low'
            key_concerns = []
            
            if region_name in high_risk_regions:
                risk_level = 'High'
                key_concerns = ["Labor rights concerns", "Limited regulatory oversight", "Potential for child labor"]
            elif region_name in medium_risk_regions:
                risk_level = 'Medium'
                key_concerns = ["Variable labor standards", "Environmental compliance challenges"]
            
            if risk_level != 'Low':
                analysis['supplier_assessment'].append({
                    'supplier_type': f"Suppliers in {region_name}",
                    'risk_level': risk_level,
                    'key_concerns': key_concerns
                })
                
                if risk_level == 'High':
                    analysis['risk_areas'].append({
                        'area': f"High-Risk Region: {region_name}",
                        'description': f"Operations in region with elevated ethical risks",
                        'likelihood': 'High',
                        'impact': 'High'
                    })
                    
                    analysis['recommendations'].append({
                        'recommendation': f"Implement enhanced due diligence for suppliers in {region_name}",
                        'priority': 'High',
                        'implementation_complexity': 'High'
                    })
                    
                    analysis['priority_actions'].append(
                        f"Conduct comprehensive risk assessment of suppliers in {region_name}"
                    )
        
        # Assess supplier types
        supplier_types = set(supplier.get('type', '') for supplier in suppliers)
        
        high_risk_types = ['raw materials', 'manufacturing', 'agriculture']
        for supplier_type in supplier_types:
            if supplier_type.lower() in high_risk_types:
                analysis['supplier_assessment'].append({
                    'supplier_type': supplier_type,
                    'risk_level': 'High',
                    'key_concerns': ["Labor intensive processes", "Environmental impact concerns", "Complex subcontracting"]
                })
                
                analysis['risk_areas'].append({
                    'area': f"High-Risk Supplier Type: {supplier_type}",
                    'description': f"Suppliers in categories with elevated ethical risks",
                    'likelihood': 'High',
                    'impact': 'High'
                })
                
                analysis['recommendations'].append({
                    'recommendation': f"Develop specific standards and monitoring for {supplier_type} suppliers",
                    'priority': 'High',
                    'implementation_complexity': 'Medium'
                })
        
        # Determine overall risk level
        high_risks = sum(1 for risk in analysis['risk_areas'] if risk.get('likelihood') == 'High' and risk.get('impact') == 'High')
        
        if high_risks >= 3:
            analysis['overall_risk_level'] = "High - Significant ethical risks requiring immediate attention"
        elif high_risks >= 1:
            analysis['overall_risk_level'] = "Medium-High - Notable ethical risks requiring systematic mitigation"
        elif analysis['risk_areas']:
            analysis['overall_risk_level'] = "Medium - Some ethical risks requiring attention"
        else:
            analysis['overall_risk_level'] = "Low - Limited identified ethical risks"
        
        # Ensure we have priority actions
        if not analysis['priority_actions']:
            analysis['priority_actions'] = [
                "Conduct comprehensive supply chain ethical risk assessment",
                "Develop supplier code of conduct if not already in place",
                "Implement supplier monitoring and verification process"
            ]
        
        return analysis
    
    def generate_sustainability_report(self, business_data: Dict[str, Any],
                                     environmental_data: Dict[str, Any],
                                     social_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a sustainability report based on business, environmental, and social data
        
        Args:
            business_data: Dictionary containing business information
            environmental_data: Dictionary containing environmental metrics
            social_data: Dictionary containing social metrics
            
        Returns:
            Dictionary containing sustainability report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'company_name': business_data.get('name', 'Unknown'),
            'executive_summary': "",
            'environmental_performance': {},
            'social_performance': {},
            'governance_highlights': {},
            'key_metrics': {},
            'goals_and_targets': {},
            'recommendations': []
        }
        
        # If API key available, use AI for report generation
        if self.api_key:
            return self._generate_sustainability_report_with_ai(business_data, environmental_data, social_data)
        
        # Fallback to rule-based report generation
        return self._generate_sustainability_report_rule_based(business_data, environmental_data, social_data)
    def _generate_sustainability_report_with_ai(self, business_data: Dict[str, Any],
                                              environmental_data: Dict[str, Any],
                                              social_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate sustainability report using OpenAI API"""
        try:
            # Prepare the context from data
            context = {
                'business': business_data,
                'environmental': environmental_data,
                'social': social_data
            }
            
            context_str = json.dumps(context, indent=2)
            
            # Construct the prompt
            prompt = f"""
            Generate a comprehensive sustainability report based on the following business, environmental, and social data:
            
            {context_str}
            
            Format your response as a JSON object with the following structure:
            {{
                "executive_summary": "Brief overview of sustainability performance and key findings",
                "environmental_performance": {{
                    "highlights": ["Key highlight 1", "Key highlight 2"],
                    "areas_of_concern": ["Concern 1", "Concern 2"],
                    "trends": "Description of environmental performance trends"
                }},
                "social_performance": {{
                    "highlights": ["Key highlight 1", "Key highlight 2"],
                    "areas_of_concern": ["Concern 1", "Concern 2"],
                    "trends": "Description of social performance trends"
                }},
                "governance_highlights": {{
                    "strengths": ["Strength 1", "Strength 2"],
                    "improvement_areas": ["Area 1", "Area 2"]
                }},
                "key_metrics": {{
                    "environmental": [
                        {{"metric": "Metric name", "value": "Metric value", "trend": "Improving/Declining/Stable", "benchmark": "Industry comparison"}}
                    ],
                    "social": [
                        {{"metric": "Metric name", "value": "Metric value", "trend": "Improving/Declining/Stable", "benchmark": "Industry comparison"}}
                    ],
                    "governance": [
                        {{"metric": "Metric name", "value": "Metric value", "trend": "Improving/Declining/Stable", "benchmark": "Industry comparison"}}
                    ]
                }},
                "goals_and_targets": {{
                    "short_term": ["Goal 1", "Goal 2"],
                    "long_term": ["Goal 1", "Goal 2"]
                }},
                "recommendations": [
                    {{"area": "Area for improvement", "recommendation": "Specific recommendation", "priority": "High/Medium/Low"}}
                ]
            }}
            
            Focus on practical sustainability insights and actionable recommendations.
            """
            
            # Call OpenAI API
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert in corporate sustainability and ESG reporting."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2500
            )
            
            # Extract and parse the response
            response_text = response.choices[0].message.content
            
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'({[\s\S]*})', response_text)
            if json_match:
                json_str = json_match.group(1)
                report = json.loads(json_str)
            else:
                raise ValueError("Could not extract valid JSON from API response")
            
            # Add metadata
            report['timestamp'] = datetime.now().isoformat()
            report['company_name'] = business_data.get('name', 'Unknown')
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating sustainability report with AI: {str(e)}")
            # Fallback to rule-based report generation
            return self._generate_sustainability_report_rule_based(business_data, environmental_data, social_data)
    
    def _generate_sustainability_report_rule_based(self, business_data: Dict[str, Any],
                                                 environmental_data: Dict[str, Any],
                                                 social_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate sustainability report using rule-based approach"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'company_name': business_data.get('name', 'Unknown'),
            'executive_summary': "",
            'environmental_performance': {
                'highlights': [],
                'areas_of_concern': [],
                'trends': ""
            },
            'social_performance': {
                'highlights': [],
                'areas_of_concern': [],
                'trends': ""
            },
            'governance_highlights': {
                'strengths': [],
                'improvement_areas': []
            },
            'key_metrics': {
                'environmental': [],
                'social': [],
                'governance': []
            },
            'goals_and_targets': {
                'short_term': [],
                'long_term': []
            },
            'recommendations': []
        }
        
        # Process environmental data
        carbon_emissions = environmental_data.get('carbon_emissions', {})
        energy_usage = environmental_data.get('energy_usage', {})
        waste_management = environmental_data.get('waste_management', {})
        water_usage = environmental_data.get('water_usage', {})
        
        # Carbon emissions analysis
        if carbon_emissions:
            total_emissions = carbon_emissions.get('total', 0)
            previous_emissions = carbon_emissions.get('previous_year', 0)
            industry_avg = carbon_emissions.get('industry_average', 0)
            
            # Add metric
            report['key_metrics']['environmental'].append({
                'metric': 'Carbon Emissions (tCO2e)',
                'value': str(total_emissions),
                'trend': 'Improving' if total_emissions < previous_emissions else 'Declining' if total_emissions > previous_emissions else 'Stable',
                'benchmark': 'Better than industry average' if industry_avg and total_emissions < industry_avg else 'Worse than industry average' if industry_avg and total_emissions > industry_avg else 'No benchmark data'
            })
            
            # Determine if highlight or concern
            if previous_emissions and total_emissions < previous_emissions * 0.95:  # 5% or more reduction
                report['environmental_performance']['highlights'].append(
                    f"Reduced carbon emissions by {((previous_emissions - total_emissions) / previous_emissions * 100):.1f}% compared to previous year"
                )
            elif previous_emissions and total_emissions > previous_emissions * 1.05:  # 5% or more increase
                report['environmental_performance']['areas_of_concern'].append(
                    f"Increased carbon emissions by {((total_emissions - previous_emissions) / previous_emissions * 100):.1f}% compared to previous year"
                )
                
                report['recommendations'].append({
                    'area': 'Carbon Emissions',
                    'recommendation': 'Develop and implement a carbon reduction strategy',
                    'priority': 'High'
                })
        else:
            report['environmental_performance']['areas_of_concern'].append(
                "No carbon emissions tracking data available"
            )
            
            report['recommendations'].append({
                'area': 'Carbon Emissions',
                'recommendation': 'Implement carbon emissions tracking and reporting',
                'priority': 'High'
            })
        
        # Energy usage analysis
        if energy_usage:
            total_energy = energy_usage.get('total', 0)
            renewable_percentage = energy_usage.get('renewable_percentage', 0)
            
            # Add metric
            report['key_metrics']['environmental'].append({
                'metric': 'Renewable Energy (%)',
                'value': f"{renewable_percentage}%",
                'trend': 'N/A',
                'benchmark': 'N/A'
            })
            
            # Determine if highlight or concern
            if renewable_percentage >= 50:
                report['environmental_performance']['highlights'].append(
                    f"{renewable_percentage}% of energy from renewable sources"
                )
            elif renewable_percentage < 20:
                report['environmental_performance']['areas_of_concern'].append(
                    f"Low renewable energy usage ({renewable_percentage}%)"
                )
                
                report['recommendations'].append({
                    'area': 'Energy Usage',
                    'recommendation': 'Increase renewable energy procurement',
                    'priority': 'Medium'
                })
        
        # Waste management analysis
        if waste_management:
            recycling_rate = waste_management.get('recycling_rate', 0)
            
            # Add metric
            report['key_metrics']['environmental'].append({
                'metric': 'Recycling Rate (%)',
                'value': f"{recycling_rate}%",
                'trend': 'N/A',
                'benchmark': 'N/A'
            })
            
            # Determine if highlight or concern
            if recycling_rate >= 70:
                report['environmental_performance']['highlights'].append(
                    f"High waste recycling rate of {recycling_rate}%"
                )
            elif recycling_rate < 30:
                report['environmental_performance']['areas_of_concern'].append(
                    f"Low waste recycling rate ({recycling_rate}%)"
                )
                
                report['recommendations'].append({
                    'area': 'Waste Management',
                    'recommendation': 'Implement comprehensive waste reduction and recycling program',
                    'priority': 'Medium'
                })
        
        # Process social data
        workforce = social_data.get('workforce', {})
        community = social_data.get('community', {})
        
        # Workforce analysis
        if workforce:
            diversity = workforce.get('diversity', {})
            training = workforce.get('training', {})
            
            # Diversity analysis
            if diversity:
                gender_ratio = diversity.get('gender_ratio', {})
                female_percentage = gender_ratio.get('female', 0)
                
                # Add metric
                report['key_metrics']['social'].append({
                    'metric': 'Female Employees (%)',
                    'value': f"{female_percentage}%",
                    'trend': 'N/A',
                    'benchmark': 'N/A'
                })
                
                # Determine if highlight or concern
                if female_percentage >= 45:
                    report['social_performance']['highlights'].append(
                        f"Gender-balanced workforce with {female_percentage}% female employees"
                    )
                elif female_percentage < 30:
                    report['social_performance']['areas_of_concern'].append(
                        f"Low female representation in workforce ({female_percentage}%)"
                    )
                    
                    report['recommendations'].append({
                        'area': 'Workforce Diversity',
                        'recommendation': 'Develop diversity and inclusion initiatives to improve gender balance',
                        'priority': 'Medium'
                    })
            
            # Training analysis
            if training:
                hours_per_employee = training.get('hours_per_employee', 0)
                
                # Add metric
                report['key_metrics']['social'].append({
                    'metric': 'Training Hours per Employee',
                    'value': str(hours_per_employee),
                    'trend': 'N/A',
                    'benchmark': 'N/A'
                })
                
                # Determine if highlight or concern
                if hours_per_employee >= 40:
                    report['social_performance']['highlights'].append(
                        f"Strong employee development with {hours_per_employee} training hours per employee"
                    )
                elif hours_per_employee < 10:
                    report['social_performance']['areas_of_concern'].append(
                        f"Limited employee training ({hours_per_employee} hours per employee)"
                    )
                    
                    report['recommendations'].append({
                        'area': 'Employee Development',
                        'recommendation': 'Increase investment in employee training and development',
                        'priority': 'Medium'
                    })
        
        # Community analysis
        if community:
            investment = community.get('investment', 0)
            volunteer_hours = community.get('volunteer_hours', 0)
            
            # Add metrics
            if investment:
                report['key_metrics']['social'].append({
                    'metric': 'Community Investment',
                    'value': f"${investment:,}",
                    'trend': 'N/A',
                    'benchmark': 'N/A'
                })
            
            if volunteer_hours:
                report['key_metrics']['social'].append({
                    'metric': 'Employee Volunteer Hours',
                    'value': str(volunteer_hours),
                    'trend': 'N/A',
                    'benchmark': 'N/A'
                })
                
                report['social_performance']['highlights'].append(
                    f"Community engagement through {volunteer_hours} employee volunteer hours"
                )
        
        # Process governance data
        governance = business_data.get('governance', {})
        
        if governance:
            board_diversity = governance.get('board_diversity', {})
            ethics_program = governance.get('ethics_program', {})
            
            # Board diversity analysis
            if board_diversity:
                female_percentage = board_diversity.get('female_percentage', 0)
                
                # Add metric
                report['key_metrics']['governance'].append({
                    'metric': 'Board Gender Diversity (%)',
                    'value': f"{female_percentage}%",
                    'trend': 'N/A',
                    'benchmark': 'N/A'
                })
                
                # Determine if strength or improvement area
                if female_percentage >= 40:
                    report['governance_highlights']['strengths'].append(
                        f"Diverse board composition with {female_percentage}% female representation"
                    )
                elif female_percentage < 20:
                    report['governance_highlights']['improvement_areas'].append(
                        f"Limited board diversity with {female_percentage}% female representation"
                    )
                    
                    report['recommendations'].append({
                        'area': 'Board Diversity',
                        'recommendation': 'Increase diversity in board composition',
                        'priority': 'Medium'
                    })
            
            # Ethics program analysis
            if ethics_program:
                has_code = ethics_program.get('code_of_conduct', False)
                has_training = ethics_program.get('ethics_training', False)
                has_reporting = ethics_program.get('reporting_mechanism', False)
                
                if has_code and has_training and has_reporting:
                    report['governance_highlights']['strengths'].append(
                        "Comprehensive ethics program with code of conduct, training, and reporting mechanisms"
                    )
                elif not has_code:
                    report['governance_highlights']['improvement_areas'].append(
                        "No formal code of conduct"
                    )
                    
                    report['recommendations'].append({
                        'area': 'Ethics & Compliance',
                        'recommendation': 'Develop and implement a formal code of conduct',
                        'priority': 'High'
                    })
                elif not has_training:
                    report['governance_highlights']['improvement_areas'].append(
                        "No formal ethics training program"
                    )
                    
                    report['recommendations'].append({
                        'area': 'Ethics & Compliance',
                        'recommendation': 'Implement ethics training for all employees',
                        'priority': 'Medium'
                    })
        
        # Generate goals and targets
        # Short-term goals
        if report['environmental_performance']['areas_of_concern']:
            report['goals_and_targets']['short_term'].append(
                "Address identified environmental concerns within 12 months"
            )
        
        if report['social_performance']['areas_of_concern']:
            report['goals_and_targets']['short_term'].append(
                "Develop action plans for social performance improvement areas"
            )
        
        if not carbon_emissions:
            report['goals_and_targets']['short_term'].append(
                "Implement carbon emissions tracking and reporting"
            )
        
        # Long-term goals
        report['goals_and_targets']['long_term'].append(
            "Achieve carbon neutrality by 2030"
        )
        
        report['goals_and_targets']['long_term'].append(
            "Increase renewable energy usage to 75% by 2030"
        )
        
        report['goals_and_targets']['long_term'].append(
            "Achieve gender parity across all levels of the organization"
        )
        
        # Generate environmental trends
        if carbon_emissions and 'previous_year' in carbon_emissions:
            current = carbon_emissions.get('total', 0)
            previous = carbon_emissions.get('previous_year', 0)
            
            if current < previous:
                report['environmental_performance']['trends'] = "Overall positive trend in environmental performance with decreasing carbon emissions."
            elif current > previous:
                report['environmental_performance']['trends'] = "Concerning trend in environmental performance with increasing carbon emissions."
            else:
                report['environmental_performance']['trends'] = "Stable environmental performance with minimal change in key metrics."
        else:
            report['environmental_performance']['trends'] = "Insufficient historical data to determine clear environmental performance trends."
        
        # Generate social trends
        report['social_performance']['trends'] = "Social performance trends cannot be determined without historical data. Establishing baseline metrics for future comparison is recommended."
        
        # Generate executive summary
        company_name = business_data.get('name', 'The company')
        
        highlight_count = len(report['environmental_performance']['highlights']) + len(report['social_performance']['highlights'])
        concern_count = len(report['environmental_performance']['areas_of_concern']) + len(report['social_performance']['areas_of_concern'])
        
        if highlight_count > concern_count:
            report['executive_summary'] = (
                f"{company_name} demonstrates generally positive sustainability performance with notable strengths in "
                f"{len(report['environmental_performance']['highlights'])} environmental and {len(report['social_performance']['highlights'])} social areas. "
                f"Key improvement opportunities exist in {concern_count} identified areas. The company should focus on addressing these gaps "
                f"while maintaining momentum in areas of strong performance."
            )
        elif concern_count > highlight_count:
            report['executive_summary'] = (
                f"{company_name}'s sustainability performance reveals significant room for improvement with {concern_count} identified areas of concern. "
                f"While the company demonstrates strengths in {highlight_count} areas, a comprehensive sustainability strategy is needed to address "
                f"key gaps and establish stronger environmental and social performance."
            )
        else:
            report['executive_summary'] = (
                f"{company_name} shows a balanced sustainability profile with equal numbers of strengths and areas for improvement. "
                f"A strategic approach to sustainability can help the company build on existing strengths while systematically addressing "
                f"identified gaps to improve overall performance."
            )
        
        return report