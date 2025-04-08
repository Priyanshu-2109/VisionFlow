import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
import re
import os
import openai
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import time

logger = logging.getLogger(__name__)

class StrategyAdvisor:
    """
    AI-powered business strategy advisor
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            logger.warning("No OpenAI API key provided. Strategy advisor will have limited functionality.")
        
        self.company_data = {}
        self.industry_data = {}
        self.market_data = {}
        self.competitor_data = {}
        self.financial_data = {}
        self.conversation_history = []
        
    def set_company_data(self, company_data: Dict[str, Any]) -> None:
        """Set company data for strategy analysis"""
        self.company_data = company_data
        logger.info(f"Company data set for {company_data.get('name', 'Unknown')}")
        
    def set_industry_data(self, industry_data: Dict[str, Any]) -> None:
        """Set industry data for strategy analysis"""
        self.industry_data = industry_data
        logger.info(f"Industry data set for {industry_data.get('name', 'Unknown')}")
        
    def set_market_data(self, market_data: Dict[str, Any]) -> None:
        """Set market data for strategy analysis"""
        self.market_data = market_data
        logger.info(f"Market data set")
        
    def set_competitor_data(self, competitor_data: Dict[str, Any]) -> None:
        """Set competitor data for strategy analysis"""
        self.competitor_data = competitor_data
        logger.info(f"Competitor data set for {len(competitor_data.get('competitors', []))} competitors")
        
    def set_financial_data(self, financial_data: Dict[str, Any]) -> None:
        """Set financial data for strategy analysis"""
        self.financial_data = financial_data
        logger.info(f"Financial data set")
    
    def generate_swot_analysis(self) -> Dict[str, Any]:
        """
        Generate SWOT (Strengths, Weaknesses, Opportunities, Threats) analysis
        
        Returns:
            Dictionary containing SWOT analysis results
        """
        # Check if we have necessary data
        if not self.company_data:
            raise ValueError("Company data must be set before generating SWOT analysis")
            
        # Prepare context for analysis
        context = self._prepare_context_for_swot()
        
        if self.api_key:
            # Use OpenAI API for advanced analysis
            swot = self._generate_swot_with_api(context)
        else:
            # Fallback to rule-based analysis
            swot = self._generate_swot_rule_based()
        
        # Add metadata
        swot['timestamp'] = datetime.now().isoformat()
        swot['company_name'] = self.company_data.get('name', 'Unknown')
        swot['industry'] = self.company_data.get('industry', 'Unknown')
        
        return swot
    
    def _prepare_context_for_swot(self) -> str:
        """Prepare context information for SWOT analysis"""
        context = []
        
        # Add company information
        company_info = f"Company: {self.company_data.get('name', 'Unknown')}\n"
        company_info += f"Industry: {self.company_data.get('industry', 'Unknown')}\n"
        company_info += f"Size: {self.company_data.get('size', 'Unknown')}\n"
        company_info += f"Business Model: {self.company_data.get('business_model', 'Unknown')}\n"
        company_info += f"Target Market: {self.company_data.get('target_market', 'Unknown')}\n"
        
        if 'products' in self.company_data:
            company_info += "Products/Services:\n"
            for product in self.company_data['products']:
                company_info += f"- {product.get('name')}: {product.get('description')}\n"
                
        if 'key_metrics' in self.company_data:
            company_info += "Key Metrics:\n"
            for metric, value in self.company_data['key_metrics'].items():
                company_info += f"- {metric}: {value}\n"
                
        context.append(company_info)
        
        # Add industry information if available
        if self.industry_data:
            industry_info = "Industry Information:\n"
            industry_info += f"Growth Rate: {self.industry_data.get('growth_rate', 'Unknown')}%\n"
            industry_info += f"Market Size: ${self.industry_data.get('market_size', 'Unknown')}\n"
            
            if 'trends' in self.industry_data:
                industry_info += "Key Trends:\n"
                for trend in self.industry_data['trends']:
                    industry_info += f"- {trend}\n"
                    
            if 'challenges' in self.industry_data:
                industry_info += "Industry Challenges:\n"
                for challenge in self.industry_data['challenges']:
                    industry_info += f"- {challenge}\n"
                    
            context.append(industry_info)
        
        # Add competitor information if available
        if self.competitor_data and 'competitors' in self.competitor_data:
            competitor_info = "Key Competitors:\n"
            for competitor in self.competitor_data['competitors']:
                competitor_info += f"- {competitor.get('name')}: {competitor.get('description', '')}\n"
                if 'strengths' in competitor:
                    competitor_info += f"  Strengths: {', '.join(competitor['strengths'])}\n"
                if 'weaknesses' in competitor:
                    competitor_info += f"  Weaknesses: {', '.join(competitor['weaknesses'])}\n"
                    
            context.append(competitor_info)
        
        # Add financial information if available
        if self.financial_data:
            financial_info = "Financial Information:\n"
            financial_info += f"Revenue: ${self.financial_data.get('revenue', 'Unknown')}\n"
            financial_info += f"Profit Margin: {self.financial_data.get('profit_margin', 'Unknown')}%\n"
            financial_info += f"Growth Rate: {self.financial_data.get('growth_rate', 'Unknown')}%\n"
            
            context.append(financial_info)
            
        return "\n".join(context)
    
    def _generate_swot_with_api(self, context: str) -> Dict[str, Any]:
        """Generate SWOT analysis using OpenAI API"""
        try:
            # Construct the prompt
            prompt = f"""
            Based on the following information about a company and its industry, provide a comprehensive SWOT (Strengths, Weaknesses, Opportunities, Threats) analysis.
            
            {context}
            
            Format your response as a JSON object with the following structure:
            {{
                "strengths": [
                    {{"point": "Strength 1", "description": "Detailed explanation", "impact": "High/Medium/Low"}},
                    ...
                ],
                "weaknesses": [
                    {{"point": "Weakness 1", "description": "Detailed explanation", "impact": "High/Medium/Low"}},
                    ...
                ],
                "opportunities": [
                    {{"point": "Opportunity 1", "description": "Detailed explanation", "impact": "High/Medium/Low"}},
                    ...
                ],
                "threats": [
                    {{"point": "Threat 1", "description": "Detailed explanation", "impact": "High/Medium/Low"}},
                    ...
                ],
                "summary": "Overall strategic assessment and key recommendations"
            }}
            
            For each category, provide 3-5 points. Ensure each point is specific, actionable, and directly relevant to the company's situation.
            """
            
            # Call OpenAI API
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a strategic business consultant with expertise in SWOT analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            # Parse the response
            response_text = response.choices[0].message.content
            
            # Extract JSON from response (handling potential text before/after the JSON)
            json_match = re.search(r'({[\s\S]*})', response_text)
            if json_match:
                json_str = json_match.group(1)
                swot = json.loads(json_str)
            else:
                raise ValueError("Could not extract valid JSON from API response")
            
            return swot
            
        except Exception as e:
            logger.error(f"Error generating SWOT analysis with API: {str(e)}")
            # Fallback to rule-based analysis
            return self._generate_swot_rule_based()
    
    def _generate_swot_rule_based(self) -> Dict[str, Any]:
        """Generate basic SWOT analysis using rule-based approach (fallback)"""
        swot = {
            "strengths": [],
            "weaknesses": [],
            "opportunities": [],
            "threats": [],
            "summary": ""
        }
        
        # Basic strengths based on company data
        if 'key_metrics' in self.company_data:
            metrics = self.company_data['key_metrics']

            if metrics.get('profit_margin', 0) < 10:
                swot['weaknesses'].append({
                    "point": "Low profit margin",
                    "description": "The company's profit margins are below industry averages, indicating potential operational inefficiencies.",
                    "impact": "High"
                })
            
            if metrics.get('market_share', 0) > 10:
                swot['strengths'].append({
                    "point": "Strong market position",
                    "description": "The company has a significant market share, giving it competitive advantage.",
                    "impact": "High"
                })
                
            if metrics.get('customer_satisfaction', 0) > 8:
                swot['strengths'].append({
                    "point": "High customer satisfaction",
                    "description": "Strong customer satisfaction scores indicate good product-market fit and service quality.",
                    "impact": "High"
                })
                
            if metrics.get('employee_count', 0) > 100:
                swot['strengths'].append({
                    "point": "Established workforce",
                    "description": "The company has a substantial workforce to execute on strategic initiatives.",
                    "impact": "Medium"
                })
        
        # Basic weaknesses
        if self.financial_data:
            if self.financial_data.get('profit_margin', 0) < 10:
                if not any("profit margin" in w["point"].lower() for w in swot['weaknesses']):
                    swot['weaknesses'].append({
                        "point": "Low profit margins",
                        "description": "The company's profit margins are below industry averages, indicating potential operational inefficiencies.",
                        "impact": "High"
                    })
                
        if 'key_metrics' in self.company_data:
            metrics = self.company_data['key_metrics']
            
            if metrics.get('customer_acquisition_cost', 0) > 1000:
                swot['weaknesses'].append({
                    "point": "High customer acquisition costs",
                    "description": "The company spends significantly to acquire new customers, which may not be sustainable.",
                    "impact": "Medium"
                })
        
        # Basic opportunities
        if self.industry_data:
            if self.industry_data.get('growth_rate', 0) > 5:
                swot['opportunities'].append({
                    "point": "Growing industry",
                    "description": "The industry is experiencing strong growth, providing expansion opportunities.",
                    "impact": "High"
                })
                
            if 'trends' in self.industry_data and self.industry_data['trends']:
                swot['opportunities'].append({
                    "point": "Emerging industry trends",
                    "description": f"The company can capitalize on trends like {', '.join(self.industry_data['trends'][:2])}.",
                    "impact": "Medium"
                })
        
        # Basic threats
        if self.competitor_data and 'competitors' in self.competitor_data:
            if len(self.competitor_data['competitors']) > 5:
                swot['threats'].append({
                    "point": "Intense competition",
                    "description": "The market has many competitors, which could pressure prices and market share.",
                    "impact": "High"
                })
                
        if self.industry_data and 'challenges' in self.industry_data and self.industry_data['challenges']:
            swot['threats'].append({
                "point": "Industry challenges",
                "description": f"The industry faces challenges such as {', '.join(self.industry_data['challenges'][:2])}.",
                "impact": "Medium"
            })
        
        # Ensure minimum number of points in each category
        categories = ['strengths', 'weaknesses', 'opportunities', 'threats']
        generic_points = {
            'strengths': [
                {"point": "Brand recognition", "description": "The company has established brand recognition in its market.", "impact": "Medium"},
                {"point": "Product quality", "description": "The company offers high-quality products or services.", "impact": "High"},
                {"point": "Experienced management", "description": "The company has experienced leadership team.", "impact": "Medium"}
            ],
            'weaknesses': [
                {"point": "Limited resources", "description": "The company may have resource constraints compared to larger competitors.", "impact": "Medium"},
                {"point": "Geographic limitations", "description": "The company's operations are limited to specific geographic areas.", "impact": "Low"},
                {"point": "Product range", "description": "The company offers a limited range of products or services.", "impact": "Medium"}
            ],
            'opportunities': [
                {"point": "New markets", "description": "Potential to expand into new geographic or demographic markets.", "impact": "High"},
                {"point": "Strategic partnerships", "description": "Opportunities to form strategic partnerships to enhance capabilities.", "impact": "Medium"},
                {"point": "Technology adoption", "description": "Leveraging new technologies to improve operations or products.", "impact": "Medium"}
            ],
            'threats': [
                {"point": "Economic uncertainty", "description": "Economic fluctuations could impact customer spending.", "impact": "Medium"},
                {"point": "Regulatory changes", "description": "Changes in regulations could impact operations or costs.", "impact": "Medium"},
                {"point": "Changing customer preferences", "description": "Evolving customer needs and preferences require adaptation.", "impact": "High"}
            ]
        }
        
        for category in categories:
            while len(swot[category]) < 3:
                remaining_points = [p for p in generic_points[category] if p not in swot[category]]
                if remaining_points:
                    swot[category].append(remaining_points[0])
                    generic_points[category].remove(remaining_points[0])
                else:
                    break
        
        # Generate summary
        company_name = self.company_data.get('name', 'The company')
        industry = self.company_data.get('industry', 'its industry')
        
        swot['summary'] = (
            f"{company_name} has several strengths to leverage in {industry}, particularly "
            f"{swot['strengths'][0]['point'].lower()} and {swot['strengths'][1]['point'].lower()}. "
            f"However, it should address weaknesses such as {swot['weaknesses'][0]['point'].lower()}. "
            f"Key opportunities include {swot['opportunities'][0]['point'].lower()}, while being mindful of "
            f"threats like {swot['threats'][0]['point'].lower()}. Strategic focus should be on leveraging strengths "
            f"to capitalize on identified opportunities while mitigating weaknesses and preparing for potential threats."
        )
        
        return swot
    
    def generate_strategic_recommendations(self, focus_area: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate strategic recommendations for the business
        
        Args:
            focus_area: Optional area to focus recommendations on (growth, efficiency, etc.)
            
        Returns:
            Dictionary containing strategic recommendations
        """
        # Check if we have necessary data
        if not self.company_data:
            raise ValueError("Company data must be set before generating recommendations")
            
        # Prepare context for analysis
        context = self._prepare_context_for_recommendations(focus_area)
        
        if self.api_key:
            # Use OpenAI API for advanced analysis
            recommendations = self._generate_recommendations_with_api(context, focus_area)
        else:
            # Fallback to rule-based analysis
            recommendations = self._generate_recommendations_rule_based(focus_area)
        
        # Add metadata
        recommendations['timestamp'] = datetime.now().isoformat()
        recommendations['company_name'] = self.company_data.get('name', 'Unknown')
        recommendations['focus_area'] = focus_area or 'general'
        
        return recommendations
    
    def _prepare_context_for_recommendations(self, focus_area: Optional[str] = None) -> str:
        """Prepare context information for strategic recommendations"""
        # Start with the same context as SWOT analysis
        context = self._prepare_context_for_swot()
        
        # Add focus area if specified
        if focus_area:
            context += f"\nFocus Area for Recommendations: {focus_area}\n"
            
            # Add specific context based on focus area
            if focus_area.lower() == 'growth':
                if self.market_data:
                    context += f"\nMarket Growth Rate: {self.market_data.get('growth_rate', 'Unknown')}%\n"
                    context += f"Market Saturation: {self.market_data.get('saturation', 'Unknown')}%\n"
                    
                if 'key_metrics' in self.company_data:
                    metrics = self.company_data['key_metrics']
                    context += f"Current Growth Rate: {metrics.get('growth_rate', 'Unknown')}%\n"
                    context += f"Customer Acquisition Cost: ${metrics.get('customer_acquisition_cost', 'Unknown')}\n"
                    context += f"Customer Lifetime Value: ${metrics.get('customer_lifetime_value', 'Unknown')}\n"
                    
            elif focus_area.lower() == 'efficiency':
                if self.financial_data:
                    context += f"\nOperating Margin: {self.financial_data.get('operating_margin', 'Unknown')}%\n"
                    context += f"Cost Structure: {self.financial_data.get('cost_structure', 'Unknown')}\n"
                    
            elif focus_area.lower() == 'innovation':
                if 'research_development' in self.company_data:
                    rd = self.company_data['research_development']
                    context += f"\nR&D Budget: ${rd.get('budget', 'Unknown')}\n"
                    context += f"R&D as % of Revenue: {rd.get('percent_of_revenue', 'Unknown')}%\n"
                    
                    if 'current_projects' in rd:
                        context += "Current R&D Projects:\n"
                        for project in rd['current_projects']:
                            context += f"- {project}\n"
                            
            elif focus_area.lower() == 'market_expansion':
                if 'target_markets' in self.company_data:
                    context += "\nCurrent Target Markets:\n"
                    for market in self.company_data['target_markets']:
                        context += f"- {market}\n"
                        
                if 'potential_markets' in self.market_data:
                    context += "\nPotential New Markets:\n"
                    for market in self.market_data['potential_markets']:
                        context += f"- {market.get('name')}: {market.get('size')} (Growth: {market.get('growth_rate')}%)\n"
        
        return context
    
    def _generate_recommendations_with_api(self, context: str, focus_area: Optional[str] = None) -> Dict[str, Any]:
        """Generate strategic recommendations using OpenAI API"""
        try:
            # Construct the prompt
            focus_instruction = ""
            if focus_area:
                focus_instruction = f"Focus specifically on {focus_area} strategies. "
            
            prompt = f"""
            Based on the following information about a company and its industry, provide strategic business recommendations.
            {focus_instruction}
            
            {context}
            
            Format your response as a JSON object with the following structure:
            {{
                "executive_summary": "Brief overview of the strategic situation and key recommendations",
                "recommendations": [
                    {{
                        "title": "Recommendation 1",
                        "description": "Detailed explanation",
                        "implementation_steps": ["Step 1", "Step 2", "Step 3"],
                        "expected_impact": "Description of expected outcomes",
                        "timeframe": "Short-term/Medium-term/Long-term",
                        "resource_requirements": "Description of resources needed",
                        "risk_level": "Low/Medium/High",
                        "priority": "Low/Medium/High"
                    }},
                    ...
                ],
                "critical_success_factors": ["Factor 1", "Factor 2", "Factor 3"],
                "kpis_to_track": ["KPI 1", "KPI 2", "KPI 3"]
            }}
            
            Provide 3-5 specific, actionable recommendations that are directly relevant to the company's situation.
            """
            
            # Call OpenAI API
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a strategic business consultant with expertise in business strategy."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2500
            )
            
            # Parse the response
            response_text = response.choices[0].message.content
            
            # Extract JSON from response (handling potential text before/after the JSON)
            json_match = re.search(r'({[\s\S]*})', response_text)
            if json_match:
                json_str = json_match.group(1)
                recommendations = json.loads(json_str)
            else:
                raise ValueError("Could not extract valid JSON from API response")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations with API: {str(e)}")
            # Fallback to rule-based analysis
            return self._generate_recommendations_rule_based(focus_area)
    
    def _generate_recommendations_rule_based(self, focus_area: Optional[str] = None) -> Dict[str, Any]:
        """Generate basic strategic recommendations using rule-based approach (fallback)"""
        recommendations = {
            "executive_summary": "",
            "recommendations": [],
            "critical_success_factors": [],
            "kpis_to_track": []
        }
        
        company_name = self.company_data.get('name', 'the company')
        industry = self.company_data.get('industry', 'its industry')
        
        # Generate recommendations based on focus area
        if focus_area and focus_area.lower() == 'growth':
            recommendations['recommendations'] = [
                {
                    "title": "Expand target customer segments",
                    "description": "Identify and target adjacent customer segments that have similar needs to current customers.",
                    "implementation_steps": [
                        "Conduct market research to identify potential segments",
                        "Develop targeted value propositions",
                        "Create marketing campaigns for new segments"
                    ],
                    "expected_impact": "Increased customer base and revenue",
                    "timeframe": "Medium-term",
                    "resource_requirements": "Marketing budget and market research resources",
                    "risk_level": "Medium",
                    "priority": "High"
                },
                {
                    "title": "Enhance product/service offerings",
                    "description": "Develop new products or services that address additional customer needs.",
                    "implementation_steps": [
                        "Conduct customer needs analysis",
                        "Prioritize product development opportunities",
                        "Develop and test new offerings"
                    ],
                    "expected_impact": "Increased revenue per customer and market share",
                    "timeframe": "Medium-term",
                    "resource_requirements": "Product development resources and budget",
                    "risk_level": "Medium",
                    "priority": "High"
                },
                {
                    "title": "Expand to new market segments",  # Changed from "Geographic expansion"
                    "description": "Enter new geographic markets to reach additional customers.",
                    "implementation_steps": [
                        "Evaluate potential geographic markets",
                        "Develop market entry strategy",
                        "Allocate resources for expansion"
                    ],
                    "expected_impact": "Access to new customer base and increased revenue",
                    "timeframe": "Long-term",
                    "resource_requirements": "Significant capital and operational resources",
                    "risk_level": "High",
                    "priority": "Medium"
                }
            ]
            
            recommendations['critical_success_factors'] = [
                "Market research accuracy",
                "Product-market fit",
                "Effective marketing execution",
                "Sufficient capital for expansion"
            ]
            
            recommendations['kpis_to_track'] = [
                "Customer acquisition rate",
                "Revenue growth rate",
                "Market share",
                "Customer lifetime value",
                "Return on marketing investment"
            ]
            
            recommendations['executive_summary'] = (
                f"For {company_name} to accelerate growth in {industry}, we recommend a three-pronged approach: "
                f"expanding target customer segments, enhancing product/service offerings, and geographic expansion. "
                f"These strategies should be implemented sequentially, starting with customer segment expansion which "
                f"offers the highest near-term potential with moderate risk. Success will depend on strong market research, "
                f"achieving product-market fit, and effective marketing execution."
            )
            
        elif focus_area and focus_area.lower() == 'efficiency':
            recommendations['recommendations'] = [
                {
                    "title": "Streamline operational processes",
                    "description": "Identify and eliminate inefficiencies in core operational processes.",
                    "implementation_steps": [
                        "Conduct process audit and mapping",
                        "Identify bottlenecks and redundancies",
                        "Implement process improvements"
                    ],
                    "expected_impact": "Reduced operational costs and improved service delivery",
                    "timeframe": "Short-term",
                    "resource_requirements": "Process improvement expertise",
                    "risk_level": "Low",
                    "priority": "High"
                },
                {
                    "title": "Implement technology automation",
                    "description": "Adopt technologies to automate manual and repetitive tasks.",
                    "implementation_steps": [
                        "Identify automation opportunities",
                        "Evaluate technology solutions",
                        "Implement and train staff on new systems"
                    ],
                    "expected_impact": "Reduced labor costs and improved accuracy",
                    "timeframe": "Medium-term",
                    "resource_requirements": "Technology investment and implementation resources",
                    "risk_level": "Medium",
                    "priority": "High"
                },
                {
                    "title": "Optimize supply chain",
                    "description": "Improve supply chain efficiency through better supplier management and logistics.",
                    "implementation_steps": [
                        "Analyze current supply chain performance",
                        "Identify improvement opportunities",
                        "Renegotiate supplier contracts and optimize logistics"
                    ],
                    "expected_impact": "Reduced costs and improved inventory management",
                    "timeframe": "Medium-term",
                    "resource_requirements": "Supply chain expertise",
                    "risk_level": "Medium",
                    "priority": "Medium"
                }
            ]
            
            recommendations['critical_success_factors'] = [
                "Employee buy-in and adoption",
                "Effective change management",
                "Maintaining service quality during transitions",
                "Accurate process analysis"
            ]
            
            recommendations['kpis_to_track'] = [
                "Operating margin",
                "Cost per unit",
                "Employee productivity",
                "Process cycle time",
                "Error rates"
            ]
            recommendations['executive_summary'] = (
                f"To improve operational efficiency at {company_name}, we recommend focusing on streamlining operational "
                f"processes, implementing technology automation, and optimizing the supply chain. These initiatives should "
                f"be prioritized in that order, with process improvements offering immediate benefits at low risk. "
                f"Successful implementation will require strong change management and employee buy-in to ensure that "
                f"efficiency gains don't compromise service quality."
            )
            
        elif focus_area and focus_area.lower() == 'innovation':
            recommendations['recommendations'] = [
                {
                    "title": "Establish innovation process",
                    "description": "Create a structured innovation process to identify, evaluate, and develop new ideas.",
                    "implementation_steps": [
                        "Define innovation objectives and focus areas",
                        "Create idea submission and evaluation system",
                        "Allocate resources for prototyping and testing"
                    ],
                    "expected_impact": "Increased flow of viable new product/service ideas",
                    "timeframe": "Short-term",
                    "resource_requirements": "Innovation management expertise",
                    "risk_level": "Low",
                    "priority": "High"
                },
                {
                    "title": "Develop strategic partnerships",
                    "description": "Form partnerships with complementary businesses, research institutions, or startups.",
                    "implementation_steps": [
                        "Identify potential partners with complementary capabilities",
                        "Develop partnership proposals",
                        "Establish joint innovation initiatives"
                    ],
                    "expected_impact": "Access to external expertise and accelerated innovation",
                    "timeframe": "Medium-term",
                    "resource_requirements": "Partnership development resources",
                    "risk_level": "Medium",
                    "priority": "Medium"
                },
                {
                    "title": "Implement innovation incentives",
                    "description": "Create incentives for employees to contribute innovative ideas and solutions.",
                    "implementation_steps": [
                        "Design incentive program",
                        "Communicate program to employees",
                        "Recognize and reward successful innovations"
                    ],
                    "expected_impact": "Increased employee-driven innovation",
                    "timeframe": "Short-term",
                    "resource_requirements": "Incentive budget and program management",
                    "risk_level": "Low",
                    "priority": "High"
                }
            ]
            
            recommendations['critical_success_factors'] = [
                "Leadership commitment to innovation",
                "Tolerance for calculated risk",
                "Cross-functional collaboration",
                "Customer-centric innovation focus"
            ]
            
            recommendations['kpis_to_track'] = [
                "Number of new ideas generated",
                "Innovation conversion rate",
                "Time to market for new innovations",
                "Revenue from products less than 3 years old",
                "Return on innovation investment"
            ]
            
            recommendations['executive_summary'] = (
                f"To foster innovation at {company_name}, we recommend establishing a structured innovation process, "
                f"developing strategic partnerships, and implementing innovation incentives. The innovation process and "
                f"incentive program should be prioritized as they can be implemented quickly with immediate impact. "
                f"Success will require strong leadership commitment to innovation and a culture that tolerates calculated "
                f"risk-taking while maintaining focus on customer needs."
            )
            
        else:
            # General recommendations if no specific focus area
            recommendations['recommendations'] = [
                {
                    "title": "Enhance digital presence",
                    "description": "Strengthen online presence and digital marketing efforts to reach more customers.",
                    "implementation_steps": [
                        "Audit current digital presence",
                        "Develop comprehensive digital strategy",
                        "Implement website improvements and digital marketing campaigns"
                    ],
                    "expected_impact": "Increased brand awareness and customer acquisition",
                    "timeframe": "Short-term",
                    "resource_requirements": "Digital marketing expertise and budget",
                    "risk_level": "Low",
                    "priority": "High"
                },
                {
                    "title": "Customer experience improvement",
                    "description": "Enhance customer experience across all touchpoints to increase satisfaction and loyalty.",
                    "implementation_steps": [
                        "Map customer journey",
                        "Identify pain points and improvement opportunities",
                        "Implement targeted improvements"
                    ],
                    "expected_impact": "Improved customer retention and word-of-mouth referrals",
                    "timeframe": "Medium-term",
                    "resource_requirements": "Customer experience expertise",
                    "risk_level": "Low",
                    "priority": "High"
                },
                {
                    "title": "Talent development",
                    "description": "Invest in employee development to build capabilities and improve retention.",
                    "implementation_steps": [
                        "Assess current skills and future needs",
                        "Develop training and development programs",
                        "Implement career progression frameworks"
                    ],
                    "expected_impact": "Improved employee capabilities and retention",
                    "timeframe": "Medium-term",
                    "resource_requirements": "Training resources and budget",
                    "risk_level": "Low",
                    "priority": "Medium"
                }
            ]
            
            recommendations['critical_success_factors'] = [
                "Customer-centric approach",
                "Effective execution and follow-through",
                "Cross-functional alignment",
                "Data-driven decision making"
            ]
            
            recommendations['kpis_to_track'] = [
                "Revenue growth",
                "Customer satisfaction score",
                "Customer retention rate",
                "Employee engagement",
                "Market share"
            ]
            
            recommendations['executive_summary'] = (
                f"For {company_name} to strengthen its competitive position in {industry}, we recommend enhancing its "
                f"digital presence, improving customer experience, and investing in talent development. Digital presence "
                f"enhancements offer immediate benefits with relatively low investment, while customer experience and "
                f"talent initiatives build longer-term sustainable advantages. Success will require a consistent "
                f"customer-centric approach and strong execution capabilities."
            )
        
        return recommendations
    
    def get_market_trends(self, industry: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze current market trends relevant to the company's industry
        
        Args:
            industry: Optional industry to analyze (defaults to company's industry)
            
        Returns:
            Dictionary containing market trend analysis
        """
        if not industry:
            industry = self.company_data.get('industry')
            
        if not industry:
            raise ValueError("Industry must be specified or set in company data")
            
        # Try to get real-time data if possible
        trends = self._gather_market_trends(industry)
        
        if self.api_key:
            # Use OpenAI API to analyze trends
            analysis = self._analyze_trends_with_api(trends, industry)
        else:
            # Fallback to basic analysis
            analysis = self._analyze_trends_basic(trends, industry)
            
        result = {
            'industry': industry,
            'timestamp': datetime.now().isoformat(),
            'trends': trends,
            'analysis': analysis
        }
        
        return result
    
    def _gather_market_trends(self, industry: str) -> List[Dict[str, Any]]:
        """Gather current market trends from various sources"""
        trends = []
        
        # Try to gather data from web sources if possible
        try:
            # This would be replaced with actual API calls or web scraping in a production system
            # Here we'll use dummy data for demonstration
            
            if 'technology' in industry.lower():
                trends = [
                    {
                        "title": "AI and Machine Learning Integration",
                        "description": "Companies are increasingly integrating AI and ML capabilities into products and services.",
                        "source": "Industry reports",
                        "impact_level": "High"
                    },
                    {
                        "title": "Remote Work Technology",
                        "description": "Continued demand for solutions supporting distributed and remote work.",
                        "source": "Market analysis",
                        "impact_level": "High"
                    },
                    {
                        "title": "Cybersecurity Focus",
                        "description": "Growing emphasis on security solutions as cyber threats increase.",
                        "source": "Security reports",
                        "impact_level": "High"
                    },
                    {
                        "title": "Edge Computing Growth",
                        "description": "Processing data closer to the source rather than in centralized locations.",
                        "source": "Tech analysis",
                        "impact_level": "Medium"
                    }
                ]
            elif 'retail' in industry.lower():
                trends = [
                    {
                        "title": "Omnichannel Integration",
                        "description": "Seamless integration between online and offline shopping experiences.",
                        "source": "Retail reports",
                        "impact_level": "High"
                    },
                    {
                        "title": "Personalized Shopping Experiences",
                        "description": "Using data to create customized recommendations and experiences.",
                        "source": "Consumer behavior analysis",
                        "impact_level": "High"
                    },
                    {
                        "title": "Sustainable and Ethical Products",
                        "description": "Growing consumer demand for environmentally and socially responsible products.",
                        "source": "Consumer surveys",
                        "impact_level": "Medium"
                    },
                    {
                        "title": "Social Commerce",
                        "description": "Shopping directly through social media platforms.",
                        "source": "Digital marketing reports",
                        "impact_level": "Medium"
                    }
                ]
            elif 'finance' in industry.lower() or 'banking' in industry.lower():
                trends = [
                    {
                        "title": "Digital Banking Acceleration",
                        "description": "Rapid shift toward digital banking services and reduced physical locations.",
                        "source": "Banking industry reports",
                        "impact_level": "High"
                    },
                    {
                        "title": "Fintech Partnerships",
                        "description": "Traditional financial institutions partnering with fintech companies.",
                        "source": "Financial industry analysis",
                        "impact_level": "High"
                    },
                    {
                        "title": "Blockchain and Cryptocurrency Adoption",
                        "description": "Increasing integration of blockchain technology and digital currencies.",
                        "source": "Technology reports",
                        "impact_level": "Medium"
                    },
                    {
                        "title": "Financial Inclusion Initiatives",
                        "description": "Focus on serving underbanked populations with accessible financial services.",
                        "source": "Social impact reports",
                        "impact_level": "Medium"
                    }
                ]
            else:
                # Generic trends for other industries
                trends = [
                    {
                        "title": "Digital Transformation",
                        "description": "Accelerated adoption of digital technologies across business processes.",
                        "source": "Industry reports",
                        "impact_level": "High"
                    },
                    {
                        "title": "Sustainability Focus",
                        "description": "Growing emphasis on environmental sustainability in operations and products.",
                        "source": "Environmental analysis",
                        "impact_level": "Medium"
                    },
                    {
                        "title": "Supply Chain Resilience",
                        "description": "Companies investing in more robust and flexible supply chains.",
                        "source": "Supply chain reports",
                        "impact_level": "High"
                    },
                    {
                        "title": "Data-Driven Decision Making",
                        "description": "Increased use of analytics and data to inform strategic decisions.",
                        "source": "Business intelligence reports",
                        "impact_level": "Medium"
                    }
                ]
        except Exception as e:
            logger.error(f"Error gathering market trends: {str(e)}")
            # Fallback to generic trends
            trends = [
                {
                    "title": "Digital Transformation",
                    "description": "Accelerated adoption of digital technologies across business processes.",
                    "source": "Industry reports",
                    "impact_level": "High"
                },
                {
                    "title": "Customer Experience Focus",
                    "description": "Emphasis on creating seamless, personalized customer experiences.",
                    "source": "Market analysis",
                    "impact_level": "High"
                },
                {
                    "title": "Workforce Evolution",
                    "description": "Changes in workforce composition, skills, and work arrangements.",
                    "source": "Labor reports",
                    "impact_level": "Medium"
                }
            ]
            
        return trends
    
    def _analyze_trends_with_api(self, trends: List[Dict[str, Any]], industry: str) -> Dict[str, Any]:
        """Analyze market trends using OpenAI API"""
        try:
            # Prepare trends information
            trends_text = ""
            for i, trend in enumerate(trends, 1):
                trends_text += f"{i}. {trend['title']}: {trend['description']} (Impact: {trend['impact_level']})\n"
            
            # Construct the prompt
            prompt = f"""
            Analyze the following market trends for the {industry} industry:
            
            {trends_text}
            
            Provide an analysis in JSON format with the following structure:
            {{
                "summary": "Brief overview of the key trends and their implications",
                "opportunities": [
                    {{"trend": "Trend name", "opportunity": "Description of business opportunity", "urgency": "High/Medium/Low"}}
                ],
                "threats": [
                    {{"trend": "Trend name", "threat": "Description of potential threat", "severity": "High/Medium/Low"}}
                ],
                "strategic_implications": [
                    "Implication 1",
                    "Implication 2",
                    "Implication 3"
                ],
                "recommended_actions": [
                    {{"action": "Action description", "timeframe": "Short-term/Medium-term/Long-term", "priority": "High/Medium/Low"}}
                ]
            }}
            
            Focus on practical implications and actionable insights.
            """
            
            # Call OpenAI API
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a market intelligence analyst with expertise in industry trends."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            # Parse the response
            response_text = response.choices[0].message.content
            
            # Extract JSON from response
            json_match = re.search(r'({[\s\S]*})', response_text)
            if json_match:
                json_str = json_match.group(1)
                analysis = json.loads(json_str)
            else:
                raise ValueError("Could not extract valid JSON from API response")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing trends with API: {str(e)}")
            # Fallback to basic analysis
            return self._analyze_trends_basic(trends, industry)
    
    def _analyze_trends_basic(self, trends: List[Dict[str, Any]], industry: str) -> Dict[str, Any]:
        """Provide basic analysis of market trends (fallback method)"""
        analysis = {
            "summary": f"The {industry} industry is experiencing several significant trends that present both opportunities and challenges.",
            "opportunities": [],
            "threats": [],
            "strategic_implications": [],
            "recommended_actions": []
        }
        
        # Generate basic opportunities and threats based on trends
        for trend in trends:
            # Classify high impact trends as both opportunities and potential threats
            if trend['impact_level'] == 'High':
                opportunity = {
                    "trend": trend['title'],
                    "opportunity": f"Leverage {trend['title'].lower()} to enhance offerings and customer experience.",
                    "urgency": "High"
                }
                analysis['opportunities'].append(opportunity)
                
                threat = {
                    "trend": trend['title'],
                    "threat": f"Competitors may gain advantage through faster adoption of {trend['title'].lower()}.",
                    "severity": "Medium"
                }
                analysis['threats'].append(threat)
            else:
                # Medium impact trends as opportunities
                opportunity = {
                    "trend": trend['title'],
                    "opportunity": f"Explore how {trend['title'].lower()} can create new value for customers.",
                    "urgency": "Medium"
                }
                analysis['opportunities'].append(opportunity)
        
        # Generate generic strategic implications
        analysis['strategic_implications'] = [
            "Digital capabilities will be increasingly critical for competitive advantage",
            "Customer expectations are evolving, requiring more personalized and responsive experiences",
            "Operational agility is needed to adapt to rapidly changing market conditions"
        ]
        
        # Generate generic recommended actions
        analysis['recommended_actions'] = [
            {
                "action": "Conduct a digital capabilities assessment",
                "timeframe": "Short-term",
                "priority": "High"
            },
            {
                "action": "Develop a roadmap for addressing key trend areas",
                "timeframe": "Short-term",
                "priority": "High"
            },
            {
                "action": "Allocate innovation budget to explore emerging opportunities",
                "timeframe": "Medium-term",
                "priority": "Medium"
            },
            {
                "action": "Monitor competitor responses to market trends",
                "timeframe": "Ongoing",
                "priority": "Medium"
            }
        ]
        
        return analysis
    
    def answer_business_query(self, query: str) -> Dict[str, Any]:
        """
        Answer specific business questions using available data and AI analysis
        
        Args:
            query: Business question to answer
            
        Returns:
            Dictionary containing answer and supporting information
        """
        if not self.api_key:
            return {
                "answer": "I'm sorry, but advanced query answering requires an OpenAI API key to be configured.",
                "confidence": "Low",
                "sources": []
            }
        
        try:
            # Prepare context from available data
            context = self._prepare_context_for_query(query)
            
            # Construct the prompt
            prompt = f"""
            Based on the following business context, answer this question:
            
            Question: {query}
            
            Context:
            {context}
            
            Format your response as a JSON object with the following structure:
            {{
                "answer": "Comprehensive answer to the question",
                "confidence": "High/Medium/Low",
                "reasoning": "Explanation of how you arrived at the answer",
                "additional_information": "Any relevant additional context or caveats",
                "follow_up_questions": ["Suggested follow-up question 1", "Suggested follow-up question 2"]
            }}
            
            If you cannot answer the question based on the provided context, explain what information would be needed.
            """
            
            # Call OpenAI API
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a business intelligence assistant with expertise in strategic analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            # Parse the response
            response_text = response.choices[0].message.content
            
            # Extract JSON from response
            json_match = re.search(r'({[\s\S]*})', response_text)
            if json_match:
                json_str = json_match.group(1)
                result = json.loads(json_str)
            else:
                raise ValueError("Could not extract valid JSON from API response")
            
            # Add query and timestamp
            result['query'] = query
            result['timestamp'] = datetime.now().isoformat()
            
            # Add to conversation history
            self.conversation_history.append({
                'query': query,
                'response': result,
                'timestamp': datetime.now().isoformat()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error answering business query: {str(e)}")
            return {
                "query": query,
                "answer": "I'm sorry, but I encountered an error while processing your question. Please try rephrasing or ask a different question.",
                "confidence": "Low",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _prepare_context_for_query(self, query: str) -> str:
        """Prepare relevant context for answering a business query"""
        context_parts = []
        
        # Add company information
        if self.company_data:
            company_info = f"Company: {self.company_data.get('name', 'Unknown')}\n"
            company_info += f"Industry: {self.company_data.get('industry', 'Unknown')}\n"
            company_info += f"Size: {self.company_data.get('size', 'Unknown')}\n"
            company_info += f"Business Model: {self.company_data.get('business_model', 'Unknown')}\n"
            
            if 'key_metrics' in self.company_data:
                company_info += "Key Metrics:\n"
                for metric, value in self.company_data['key_metrics'].items():
                    company_info += f"- {metric}: {value}\n"
                    
            context_parts.append(company_info)
        
        # Add financial information if relevant to the query
        financial_keywords = ['financial', 'revenue', 'profit', 'margin', 'cost', 'expense', 'budget', 'investment']
        if self.financial_data and any(keyword in query.lower() for keyword in financial_keywords):
            financial_info = "Financial Information:\n"
            for key, value in self.financial_data.items():
                financial_info += f"- {key}: {value}\n"
                
            context_parts.append(financial_info)
        
        # Add market information if relevant to the query
        market_keywords = ['market', 'industry', 'trend', 'growth', 'competitor', 'competition', 'share']
        if self.market_data and any(keyword in query.lower() for keyword in market_keywords):
            market_info = "Market Information:\n"
            for key, value in self.market_data.items():
                if isinstance(value, dict) or isinstance(value, list):
                    market_info += f"- {key}: {json.dumps(value)[:200]}...\n"
                else:
                    market_info += f"- {key}: {value}\n"
                    
            context_parts.append(market_info)
        
        # Add competitor information if relevant to the query
        competitor_keywords = ['competitor', 'competition', 'rival', 'market share', 'competitive']
        if self.competitor_data and any(keyword in query.lower() for keyword in competitor_keywords):
            competitor_info = "Competitor Information:\n"
            if 'competitors' in self.competitor_data:
                for competitor in self.competitor_data['competitors']:
                    competitor_info += f"- {competitor.get('name')}: {competitor.get('description', '')}\n"
                    
            context_parts.append(competitor_info)
        
        # Add recent conversation history for context
        if self.conversation_history:
            recent_history = self.conversation_history[-3:]  # Last 3 exchanges
            history_info = "Recent Conversation History:\n"
            for exchange in recent_history:
                history_info += f"Q: {exchange['query']}\n"
                history_info += f"A: {exchange['response'].get('answer', '')[:100]}...\n"
                
            context_parts.append(history_info)
        
        return "\n\n".join(context_parts)
    
    def generate_business_plan(self, plan_type: str, timeframe: str) -> Dict[str, Any]:
        """
        Generate a business plan based on available data
        
        Args:
            plan_type: Type of plan (strategic, growth, turnaround, etc.)
            timeframe: Time horizon for the plan (1-year, 3-year, 5-year, etc.)
            
        Returns:
            Dictionary containing the business plan
        """
        if not self.company_data:
            raise ValueError("Company data must be set before generating a business plan")
            
        if not self.api_key:
            return {
                "error": "Business plan generation requires an OpenAI API key to be configured.",
                "plan_type": plan_type,
                "timeframe": timeframe
            }
            
        try:
            # Prepare context for plan generation
            context = self._prepare_context_for_business_plan(plan_type, timeframe)
            
            # Construct the prompt
            prompt = f"""
            Generate a {plan_type} business plan with a {timeframe} horizon based on the following information:
            
            {context}
            
            Format your response as a JSON object with the following structure:
            {{
                "executive_summary": "Brief overview of the plan",
                "vision_statement": "Vision for the company at the end of the plan period",
                "strategic_objectives": [
                    {{"objective": "Objective 1", "description": "Detailed explanation", "key_results": ["KR1", "KR2", "KR3"]}}
                ],
                "market_analysis": {{
                    "target_segments": ["Segment 1", "Segment 2"],
                    "market_size": "Description of addressable market",
                    "competitive_landscape": "Analysis of competitive position",
                    "growth_opportunities": ["Opportunity 1", "Opportunity 2"]
                }},
                "strategy": {{
                    "value_proposition": "Core value proposition",
                    "competitive_advantage": "Sources of competitive advantage",
                    "growth_strategy": "Approach to achieving growth"
                }},
                "implementation_roadmap": [
                    {{"phase": "Phase 1", "timeframe": "Q1-Q2", "key_initiatives": ["Initiative 1", "Initiative 2"], "expected_outcomes": "Description of outcomes"}}
                ],
                "financial_projections": {{
                    "revenue_growth": "Projected growth rate",
                    "profitability_targets": "Margin targets",
                    "investment_requirements": "Required investments",
                    "key_assumptions": ["Assumption 1", "Assumption 2"]
                }},
                "risk_assessment": [
                    {{"risk": "Risk 1", "likelihood": "High/Medium/Low", "impact": "High/Medium/Low", "mitigation": "Mitigation approach"}}
                ],
                "success_metrics": ["Metric 1", "Metric 2", "Metric 3"]
            }}
            
            Ensure the plan is realistic, actionable, and tailored to the company's specific situation and industry context.
            """
            
            # Call OpenAI API
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a strategic business consultant with expertise in business planning."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=3000
            )
            
            # Parse the response
            response_text = response.choices[0].message.content
            
            # Extract JSON from response
            json_match = re.search(r'({[\s\S]*})', response_text)
            if json_match:
                json_str = json_match.group(1)
                business_plan = json.loads(json_str)
            else:
                raise ValueError("Could not extract valid JSON from API response")
            
            # Add metadata
            business_plan['plan_type'] = plan_type
            business_plan['timeframe'] = timeframe
            business_plan['company_name'] = self.company_data.get('name', 'Unknown')
            business_plan['generation_date'] = datetime.now().isoformat()
            
            return business_plan
            
        except Exception as e:
            logger.error(f"Error generating business plan: {str(e)}")
            return {
                "error": f"Failed to generate business plan: {str(e)}",
                "plan_type": plan_type,
                "timeframe": timeframe
            }
    def _prepare_context_for_business_plan(self, plan_type: str, timeframe: str) -> str:
        """Prepare context information for business plan generation"""
        # Comprehensive context including all available data
        context_parts = []
        
        # Company information
        if self.company_data:
            company_info = f"Company: {self.company_data.get('name', 'Unknown')}\n"
            company_info += f"Industry: {self.company_data.get('industry', 'Unknown')}\n"
            company_info += f"Size: {self.company_data.get('size', 'Unknown')}\n"
            company_info += f"Business Model: {self.company_data.get('business_model', 'Unknown')}\n"
            company_info += f"Target Market: {self.company_data.get('target_market', 'Unknown')}\n"
            
            if 'mission' in self.company_data:
                company_info += f"Mission: {self.company_data['mission']}\n"
                
            if 'vision' in self.company_data:
                company_info += f"Vision: {self.company_data['vision']}\n"
                
            if 'values' in self.company_data:
                company_info += "Values:\n"
                for value in self.company_data['values']:
                    company_info += f"- {value}\n"
                    
            if 'products' in self.company_data:
                company_info += "Products/Services:\n"
                for product in self.company_data['products']:
                    company_info += f"- {product.get('name')}: {product.get('description')}\n"
                    
            if 'key_metrics' in self.company_data:
                company_info += "Key Metrics:\n"
                for metric, value in self.company_data['key_metrics'].items():
                    company_info += f"- {metric}: {value}\n"
                    
            context_parts.append(company_info)
        
        # Financial information
        if self.financial_data:
            financial_info = "Financial Information:\n"
            for key, value in self.financial_data.items():
                financial_info += f"- {key}: {value}\n"
                
            context_parts.append(financial_info)
        
        # Market information
        if self.market_data:
            market_info = "Market Information:\n"
            for key, value in self.market_data.items():
                if isinstance(value, dict) or isinstance(value, list):
                    continue  # Skip complex nested structures
                market_info += f"- {key}: {value}\n"
                
            if 'trends' in self.market_data:
                market_info += "Market Trends:\n"
                for trend in self.market_data['trends']:
                    market_info += f"- {trend}\n"
                    
            context_parts.append(market_info)
        
        # Competitor information
        if self.competitor_data and 'competitors' in self.competitor_data:
            competitor_info = "Key Competitors:\n"
            for competitor in self.competitor_data['competitors']:
                competitor_info += f"- {competitor.get('name')}: {competitor.get('description', '')}\n"
                if 'market_share' in competitor:
                    competitor_info += f"  Market Share: {competitor['market_share']}%\n"
                    
            context_parts.append(competitor_info)
        
        # Industry information
        if self.industry_data:
            industry_info = "Industry Information:\n"
            industry_info += f"Growth Rate: {self.industry_data.get('growth_rate', 'Unknown')}%\n"
            industry_info += f"Market Size: ${self.industry_data.get('market_size', 'Unknown')}\n"
            
            if 'trends' in self.industry_data:
                industry_info += "Industry Trends:\n"
                for trend in self.industry_data['trends']:
                    industry_info += f"- {trend}\n"
                    
            if 'challenges' in self.industry_data:
                industry_info += "Industry Challenges:\n"
                for challenge in self.industry_data['challenges']:
                    industry_info += f"- {challenge}\n"
                    
            context_parts.append(industry_info)
        
        # Add plan-specific context
        plan_context = f"Plan Type: {plan_type}\n"
        plan_context += f"Timeframe: {timeframe}\n"
        
        if plan_type.lower() == 'growth':
            plan_context += "Focus on strategies for accelerating business growth, expanding market share, and increasing revenue.\n"
        elif plan_type.lower() == 'turnaround':
            plan_context += "Focus on addressing business challenges, improving financial performance, and repositioning for future success.\n"
        elif plan_type.lower() == 'strategic':
            plan_context += "Focus on long-term strategic positioning, competitive advantage, and sustainable value creation.\n"
        elif plan_type.lower() == 'innovation':
            plan_context += "Focus on developing new products, services, or business models to create new sources of value.\n"
        
        context_parts.append(plan_context)
        
        return "\n\n".join(context_parts)