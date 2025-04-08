import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
import os
import openai
from datetime import datetime

logger = logging.getLogger(__name__)

class SWOTAnalyzer:
    """
    Specialized SWOT analysis module for business strategy
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            logger.warning("No OpenAI API key provided. SWOT analyzer will have limited functionality.")
    
    def analyze_company_data(self, company_data: Dict[str, Any],
                           industry_data: Optional[Dict[str, Any]] = None,
                           competitor_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate SWOT analysis from company, industry, and competitor data
        
        Args:
            company_data: Dictionary containing company information
            industry_data: Optional dictionary containing industry information
            competitor_data: Optional dictionary containing competitor information
            
        Returns:
            Dictionary containing SWOT analysis
        """
        # Prepare data for analysis
        analysis_data = self._prepare_analysis_data(company_data, industry_data, competitor_data)
        
        if self.api_key:
            # Use AI for advanced analysis
            swot = self._generate_ai_swot(analysis_data)
        else:
            # Use rule-based analysis as fallback
            swot = self._generate_rule_based_swot(analysis_data)
            
        # Add metadata
        swot['timestamp'] = datetime.now().isoformat()
        swot['company_name'] = company_data.get('name', 'Unknown')
        
        return swot
    
    def _prepare_analysis_data(self, company_data: Dict[str, Any],
                             industry_data: Optional[Dict[str, Any]],
                             competitor_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare data for SWOT analysis"""
        analysis_data = {
            'company': company_data,
            'industry': industry_data or {},
            'competitors': competitor_data or {}
        }
        
        return analysis_data
    
    def _generate_ai_swot(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate SWOT analysis using OpenAI API"""
        try:
            # Prepare the context from analysis data
            company = analysis_data['company']
            industry = analysis_data['industry']
            competitors = analysis_data['competitors']
            
            context = f"Company: {company.get('name', 'Unknown')}\n"
            context += f"Industry: {company.get('industry', 'Unknown')}\n"
            context += f"Size: {company.get('size', 'Unknown')}\n"
            context += f"Business Model: {company.get('business_model', 'Unknown')}\n"
            context += f"Target Market: {company.get('target_market', 'Unknown')}\n"
            
            if 'key_metrics' in company:
                context += "Key Metrics:\n"
                for key, value in company['key_metrics'].items():
                    context += f"- {key}: {value}\n"
            
            if industry:
                context += "\nIndustry Information:\n"
                if 'growth_rate' in industry:
                    context += f"- Growth Rate: {industry['growth_rate']}%\n"
                if 'market_size' in industry:
                    context += f"- Market Size: ${industry['market_size']}\n"
                if 'trends' in industry:
                    context += "- Trends:\n"
                    for trend in industry['trends']:
                        context += f"  * {trend}\n"
                if 'challenges' in industry:
                    context += "- Challenges:\n"
                    for challenge in industry['challenges']:
                        context += f"  * {challenge}\n"
            
            if competitors and 'competitors' in competitors:
                context += "\nKey Competitors:\n"
                for competitor in competitors['competitors']:
                    context += f"- {competitor.get('name')}: {competitor.get('description', '')}\n"
                    if 'strengths' in competitor:
                        context += f"  Strengths: {', '.join(competitor['strengths'])}\n"
                    if 'weaknesses' in competitor:
                        context += f"  Weaknesses: {', '.join(competitor['weaknesses'])}\n"
            
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
            
            # Extract and parse the response
            response_text = response.choices[0].message.content
            
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'({[\s\S]*})', response_text)
            if json_match:
                json_str = json_match.group(1)
                swot = json.loads(json_str)
            else:
                raise ValueError("Could not extract valid JSON from API response")
                
            return swot
            
        except Exception as e:
            logger.error(f"Error generating AI SWOT analysis: {str(e)}")
            # Fallback to rule-based analysis
            return self._generate_rule_based_swot(analysis_data)
    
    def _generate_rule_based_swot(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate rule-based SWOT analysis as fallback"""
        company = analysis_data['company']
        industry = analysis_data['industry']
        competitors = analysis_data['competitors']
        
        swot = {
            "strengths": [],
            "weaknesses": [],
            "opportunities": [],
            "threats": [],
            "summary": ""
        }
        
        # Generate strengths based on company data
        if 'key_metrics' in company:
            metrics = company['key_metrics']
            
            # Market share as potential strength
            if metrics.get('market_share', 0) > 10:
                swot['strengths'].append({
                    "point": "Strong market position",
                    "description": "The company has a significant market share, giving it competitive advantage.",
                    "impact": "High"
                })
            
            # Customer satisfaction as potential strength
            if metrics.get('customer_satisfaction', 0) > 8:
                swot['strengths'].append({
                    "point": "High customer satisfaction",
                    "description": "Strong customer satisfaction scores indicate good product-market fit and service quality.",
                    "impact": "High"
                })
            
            # Team size as potential strength
            if metrics.get('employee_count', 0) > 100:
                swot['strengths'].append({
                    "point": "Established workforce",
                    "description": "The company has a substantial workforce to execute on strategic initiatives.",
                    "impact": "Medium"
                })
                
            # R&D investment as potential strength
            if metrics.get('rd_investment_percentage', 0) > 10:
                swot['strengths'].append({
                    "point": "Strong R&D investment",
                    "description": "Above-average investment in research and development can drive innovation and new product development.",
                    "impact": "High"
                })
        
        # Brand recognition as potential strength
        if company.get('brand_recognition', 'low').lower() in ['high', 'strong', 'established']:
            swot['strengths'].append({
                "point": "Strong brand recognition",
                "description": "The company has an established brand that resonates with customers.",
                "impact": "High"
            })
            
        # Generate weaknesses based on company data
        if 'key_metrics' in company:
            metrics = company['key_metrics']
            
            # Low profit margins as weakness
            if metrics.get('profit_margin', 20) < 10:
                swot['weaknesses'].append({
                    "point": "Low profit margins",
                    "description": "The company's profit margins are below industry averages, indicating potential operational inefficiencies.",
                    "impact": "High"
                })
                
            # High customer acquisition cost as weakness
            if metrics.get('customer_acquisition_cost', 0) > 1000:
                swot['weaknesses'].append({
                    "point": "High customer acquisition costs",
                    "description": "The company spends significantly to acquire new customers, which may not be sustainable.",
                    "impact": "Medium"
                })
                
            # High employee turnover as weakness
            if metrics.get('employee_turnover', 0) > 20:
                swot['weaknesses'].append({
                    "point": "High employee turnover",
                    "description": "Above-average employee turnover can lead to knowledge loss and increased training costs.",
                    "impact": "Medium"
                })
                
            # Low market share as weakness
            if metrics.get('market_share', 15) < 5:
                swot['weaknesses'].append({
                    "point": "Limited market presence",
                    "description": "The company has a relatively small market share, which may limit economies of scale and market influence.",
                    "impact": "Medium"
                })
        
        # Generate opportunities based on industry data
        if industry:
            # Industry growth as opportunity
            if industry.get('growth_rate', 0) > 5:
                swot['opportunities'].append({
                    "point": "Growing industry",
                    "description": "The industry is experiencing strong growth, providing expansion opportunities.",
                    "impact": "High"
                })
                
            # Industry trends as opportunities
            if 'trends' in industry and industry['trends']:
                for i, trend in enumerate(industry['trends'][:2]):  # Take up to 2 trends
                    swot['opportunities'].append({
                        "point": f"Emerging trend: {trend}",
                        "description": f"The company can capitalize on the trend of {trend} to develop new offerings or reach new customers.",
                        "impact": "Medium"
                    })
                    
            # Digital transformation as opportunity (common in many industries)
            if 'digital_transformation' in industry.get('trends', []) or True:  # Default assumption
                swot['opportunities'].append({
                    "point": "Digital transformation",
                    "description": "Leveraging digital technologies to improve operations, customer experience, and create new business models.",
                    "impact": "High"
                })
        
        # Generate threats based on industry and competitor data
        if industry:
            # Industry challenges as threats
            if 'challenges' in industry and industry['challenges']:
                for i, challenge in enumerate(industry['challenges'][:2]):  # Take up to 2 challenges
                    swot['threats'].append({
                        "point": f"Industry challenge: {challenge}",
                        "description": f"The company must address the industry-wide challenge of {challenge}.",
                        "impact": "Medium"
                    })
        
        if competitors and 'competitors' in competitors:
            # Intense competition as threat
            if len(competitors['competitors']) > 5:
                swot['threats'].append({
                    "point": "Intense competition",
                    "description": "The market has many competitors, which could pressure prices and market share.",
                    "impact": "High"
                })
                
            # Strong competitor as threat
            for competitor in competitors['competitors']:
                if competitor.get('market_share', 0) > company.get('key_metrics', {}).get('market_share', 0):
                    swot['threats'].append({
                        "point": f"Strong competitor: {competitor.get('name', 'Unknown')}",
                        "description": f"{competitor.get('name', 'A competitor')} has a larger market share and may have greater resources or capabilities.",
                        "impact": "High"
                    })
                    break
        
        # Regulatory changes as a common threat
        swot['threats'].append({
            "point": "Regulatory changes",
            "description": "Changes in regulations could impact operations, compliance costs, or business model.",
            "impact": "Medium"
        })
        
        # Ensure minimum number of points in each category
        for category in ['strengths', 'weaknesses', 'opportunities', 'threats']:
            while len(swot[category]) < 3:
                if category == 'strengths':
                    swot[category].append({
                        "point": "Product/service quality",
                        "description": "The company offers high-quality products or services that meet customer needs.",
                        "impact": "Medium"
                    })
                elif category == 'weaknesses':
                    swot[category].append({
                        "point": "Limited resources",
                        "description": "The company may have resource constraints compared to larger competitors.",
                        "impact": "Medium"
                    })
                elif category == 'opportunities':
                    swot[category].append({
                        "point": "New market segments",
                        "description": "Potential to expand into new customer segments or geographic markets.",
                        "impact": "Medium"
                    })
                elif category == 'threats':
                    swot[category].append({
                        "point": "Economic uncertainty",
                        "description": "Economic fluctuations could impact customer spending and business performance.",
                        "impact": "Medium"
                    })
        
        # Generate summary
        company_name = company.get('name', 'The company')
        industry_name = company.get('industry', 'its industry')
        
        swot['summary'] = (
            f"{company_name} has several strengths to leverage in {industry_name}, particularly "
            f"{swot['strengths'][0]['point'].lower()} and {swot['strengths'][1]['point'].lower()}. "
            f"However, it should address weaknesses such as {swot['weaknesses'][0]['point'].lower()}. "
            f"Key opportunities include {swot['opportunities'][0]['point'].lower()}, while being mindful of "
            f"threats like {swot['threats'][0]['point'].lower()}. Strategic focus should be on leveraging strengths "
            f"to capitalize on identified opportunities while mitigating weaknesses and preparing for potential threats."
        )
        
        return swot
    
    def compare_with_competitors(self, company_data: Dict[str, Any],
                               competitor_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comparative analysis between company and competitors
        
        Args:
            company_data: Dictionary containing company information
            competitor_data: List of dictionaries containing competitor information
            
        Returns:
            Dictionary containing comparative analysis
        """
        # Prepare data for analysis
        company_name = company_data.get('name', 'Our Company')
        
        # Identify common metrics for comparison
        all_metrics = set()
        if 'key_metrics' in company_data:
            all_metrics.update(company_data['key_metrics'].keys())
            
        for competitor in competitor_data:
            if 'key_metrics' in competitor:
                all_metrics.update(competitor['key_metrics'].keys())
        
        # Filter to commonly available metrics
        common_metrics = []
        for metric in all_metrics:
            if ('key_metrics' in company_data and metric in company_data['key_metrics'] and
                all('key_metrics' in comp and metric in comp['key_metrics'] for comp in competitor_data)):
                common_metrics.append(metric)
        
        # Prepare comparison data
        comparison = {
            'company_name': company_name,
            'competitors': [comp.get('name', f'Competitor {i+1}') for i, comp in enumerate(competitor_data)],
            'metrics': {},
            'strengths_vs_competitors': [],
            'weaknesses_vs_competitors': [],
            'overall_assessment': ""
        }
        
        # Compare metrics
        for metric in common_metrics:
            comparison['metrics'][metric] = {
                'company': company_data['key_metrics'][metric],
                'competitors': [comp['key_metrics'][metric] for comp in competitor_data]
            }
            
            # Determine if this is a strength or weakness
            company_value = company_data['key_metrics'][metric]
            competitor_values = [comp['key_metrics'][metric] for comp in competitor_data]
            
            # For metrics where higher is better (assumed by default)
            higher_is_better = metric not in ['cost', 'expense', 'churn', 'turnover']
            
            if higher_is_better:
                if company_value > max(competitor_values):
                    comparison['strengths_vs_competitors'].append({
                        'metric': metric,
                        'description': f"Leading the market in {metric} with a value of {company_value} compared to competitor average of {sum(competitor_values)/len(competitor_values):.2f}"
                    })
                elif company_value < min(competitor_values):
                    comparison['weaknesses_vs_competitors'].append({
                        'metric': metric,
                        'description': f"Lagging behind competitors in {metric} with a value of {company_value} compared to competitor average of {sum(competitor_values)/len(competitor_values):.2f}"
                    })
            else:
                # For metrics where lower is better
                if company_value < min(competitor_values):
                    comparison['strengths_vs_competitors'].append({
                        'metric': metric,
                        'description': f"Leading the market in {metric} with a value of {company_value} compared to competitor average of {sum(competitor_values)/len(competitor_values):.2f}"
                    })
                elif company_value > max(competitor_values):
                    comparison['weaknesses_vs_competitors'].append({
                        'metric': metric,
                        'description': f"Lagging behind competitors in {metric} with a value of {company_value} compared to competitor average of {sum(competitor_values)/len(competitor_values):.2f}"
                    })
        
        # Generate overall assessment
        strength_count = len(comparison['strengths_vs_competitors'])
        weakness_count = len(comparison['weaknesses_vs_competitors'])
        
        if strength_count > weakness_count:
            comparison['overall_assessment'] = (
                f"{company_name} demonstrates competitive advantages in {strength_count} key metrics, "
                f"particularly in {', '.join([s['metric'] for s in comparison['strengths_vs_competitors'][:2]])}. "
                f"However, attention should be given to improving {', '.join([w['metric'] for w in comparison['weaknesses_vs_competitors'][:2]])} "
                f"to strengthen overall competitive position."
            )
        elif weakness_count > strength_count:
            comparison['overall_assessment'] = (
                f"{company_name} faces competitive challenges in {weakness_count} key metrics, "
                f"particularly in {', '.join([w['metric'] for w in comparison['weaknesses_vs_competitors'][:2]])}. "
                f"The company should leverage its strengths in {', '.join([s['metric'] for s in comparison['strengths_vs_competitors'][:2]] if comparison['strengths_vs_competitors'] else ['other areas'])} "
                f"while developing strategies to address competitive gaps."
            )
        else:
            comparison['overall_assessment'] = (
                f"{company_name} shows a balanced competitive position with equal numbers of strengths and weaknesses. "
                f"Key strengths include {', '.join([s['metric'] for s in comparison['strengths_vs_competitors'][:2]] if comparison['strengths_vs_competitors'] else ['other areas'])}, "
                f"while improvement opportunities exist in {', '.join([w['metric'] for w in comparison['weaknesses_vs_competitors'][:2]] if comparison['weaknesses_vs_competitors'] else ['other areas'])}."
            )
        
        return comparison