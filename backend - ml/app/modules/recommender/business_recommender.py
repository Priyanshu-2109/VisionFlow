# app/modules/recommender/business_recommender.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class BusinessRecommender:
    """
    Generate personalized business recommendations
    """
    
    def __init__(self):
        self.business_data = {}
        self.market_data = {}
        self.trend_data = {}
    
    def set_business_data(self, business_data: Dict[str, Any]) -> None:
        """Set business data for recommendation generation"""
        self.business_data = business_data
        logger.info(f"Business data set for {business_data.get('name', 'Unknown')}")
    
    def set_market_data(self, market_data: Dict[str, Any]) -> None:
        """Set market data for recommendation generation"""
        self.market_data = market_data
        logger.info(f"Market data set")
    
    def set_trend_data(self, trend_data: Dict[str, Any]) -> None:
        """Set trend data for recommendation generation"""
        self.trend_data = trend_data
        logger.info(f"Trend data set")
    
    def generate_growth_recommendations(self) -> Dict[str, Any]:
        """
        Generate growth strategy recommendations
        
        Returns:
            Dictionary containing growth recommendations
        """
        if not self.business_data:
            return {
                'error': 'Business data not set',
                'timestamp': datetime.now().isoformat()
            }
        
        recommendations = {
            'timestamp': datetime.now().isoformat(),
            'company_name': self.business_data.get('name', 'Unknown'),
            'industry': self.business_data.get('industry', 'Unknown'),
            'growth_strategies': [],
            'market_opportunities': [],
            'product_recommendations': [],
            'priority_actions': []
        }
        
        # Generate growth strategies
        growth_strategies = self._generate_growth_strategies()
        recommendations['growth_strategies'] = growth_strategies
        
        # Generate market opportunities
        market_opportunities = self._identify_market_opportunities()
        recommendations['market_opportunities'] = market_opportunities
        
        # Generate product recommendations
        product_recommendations = self._generate_product_recommendations()
        recommendations['product_recommendations'] = product_recommendations
        
        # Generate priority actions
        priority_actions = self._generate_priority_actions(growth_strategies, market_opportunities, product_recommendations)
        recommendations['priority_actions'] = priority_actions
        
        return recommendations
    
    def _generate_growth_strategies(self) -> List[Dict[str, Any]]:
        """Generate growth strategy recommendations"""
        strategies = []
        
        # Get business details
        business_size = self.business_data.get('size', 'small')
        business_model = self.business_data.get('business_model', 'unknown')
        industry = self.business_data.get('industry', 'unknown')
        
        # Market penetration strategy
        market_share = self.business_data.get('market_share', 0)
        market_growth = self.market_data.get('market_growth', 0)
        
        if market_share < 10 and market_growth > 0:
            strategies.append({
                'strategy': 'Market Penetration',
                'description': 'Increase market share in existing markets with current products/services',
                'rationale': f"Current market share ({market_share}%) suggests significant room for growth in a growing market ({market_growth}% growth rate)",
                'actions': [
                    'Enhance marketing and sales efforts to reach more customers',
                    'Optimize pricing strategy to increase competitiveness',
                    'Improve customer retention through enhanced service and loyalty programs',
                    'Consider strategic partnerships to expand distribution'
                ],
                'resource_requirements': 'Medium',
                'timeframe': 'Short-term',
                'expected_impact': 'High',
                'risk_level': 'Low'
            })
        
        # Market development strategy
        if self.market_data.get('adjacent_markets'):
            adjacent_markets = self.market_data.get('adjacent_markets', [])
            if adjacent_markets:
                top_market = adjacent_markets[0]
                strategies.append({
                    'strategy': 'Market Development',
                    'description': 'Expand into new markets with existing products/services',
                    'rationale': f"Opportunity to leverage existing offerings in new markets such as {top_market.get('name')} with {top_market.get('size')} size and {top_market.get('growth')}% growth",
                    'actions': [
                        'Conduct market research to understand new customer needs',
                        'Adapt marketing messaging for new market segments',
                        'Develop go-to-market strategy for target markets',
                        'Establish local partnerships or presence in new geographic markets'
                    ],
                    'resource_requirements': 'Medium',
                    'timeframe': 'Medium-term',
                    'expected_impact': 'Medium',
                    'risk_level': 'Medium'
                })
        
        # Product development strategy
        if 'r_and_d_capability' in self.business_data and self.business_data['r_and_d_capability'] > 3:
            strategies.append({
                'strategy': 'Product Development',
                'description': 'Develop new products/services for existing markets',
                'rationale': f"Strong R&D capabilities can be leveraged to expand offerings to current customer base",
                'actions': [
                    'Analyze customer needs and identify product gaps',
                    'Prioritize product development initiatives based on market potential',
                    'Accelerate innovation through agile development processes',
                    'Test new offerings with existing customers for feedback'
                ],
                'resource_requirements': 'High',
                'timeframe': 'Medium-term',
                'expected_impact': 'High',
                'risk_level': 'Medium'
            })
        
        # Diversification strategy
        if business_size == 'large' and self.business_data.get('financial_strength', 0) > 7:
            strategies.append({
                'strategy': 'Diversification',
                'description': 'Enter new markets with new products/services',
                'rationale': f"Strong financial position enables exploration of new business opportunities beyond current focus",
                'actions': [
                    'Identify high-potential markets aligned with company capabilities',
                    'Develop offerings tailored to new market needs',
                    'Consider acquisitions to accelerate market entry',
                    'Create separate business unit for new ventures'
                ],
                'resource_requirements': 'High',
                'timeframe': 'Long-term',
                'expected_impact': 'High',
                'risk_level': 'High'
            })
        
        # Digital transformation strategy
        if self.business_data.get('digital_maturity', 0) < 7:
            strategies.append({
                'strategy': 'Digital Transformation',
                'description': 'Leverage digital technologies to transform business model and customer experience',
                'rationale': f"Enhancing digital capabilities can create competitive advantage and operational efficiencies",
                'actions': [
                    'Assess current digital capabilities and gaps',
                    'Develop digital transformation roadmap',
                    'Implement key digital initiatives (e-commerce, automation, data analytics)',
                    'Build digital skills and culture throughout organization'
                ],
                'resource_requirements': 'High',
                'timeframe': 'Medium-term',
                'expected_impact': 'High',
                'risk_level': 'Medium'
            })
        
        # Acquisition strategy
        if business_size in ['medium', 'large'] and self.business_data.get('financial_strength', 0) > 8:
            strategies.append({
                'strategy': 'Strategic Acquisitions',
                'description': 'Accelerate growth through targeted acquisitions',
                'rationale': f"Strong financial position enables inorganic growth through acquiring complementary businesses",
                'actions': [
                    'Define acquisition criteria aligned with strategic objectives',
                    'Identify and evaluate potential acquisition targets',
                    'Develop integration plan for successful post-merger outcomes',
                    'Secure necessary financing and resources'
                ],
                'resource_requirements': 'High',
                'timeframe': 'Medium-term',
                'expected_impact': 'High',
                'risk_level': 'High'
            })
        
        # Strategic partnerships
        strategies.append({
            'strategy': 'Strategic Partnerships',
            'description': 'Form alliances to access new capabilities, customers, or markets',
            'rationale': f"Partnerships can provide access to complementary resources and accelerate growth",
            'actions': [
                'Identify potential partners with complementary capabilities',
                'Develop partnership strategy and objectives',
                'Create structured approach to partnership management',
                'Establish clear metrics for partnership success'
            ],
            'resource_requirements': 'Medium',
            'timeframe': 'Medium-term',
            'expected_impact': 'Medium',
            'risk_level': 'Medium'
        })
        
        return strategies
    def _identify_market_opportunities(self) -> List[Dict[str, Any]]:
        """Identify market opportunities based on trends and market data"""
        opportunities = []
        
        # Check if we have trend data
        if not self.trend_data:
            return opportunities
        
        # Extract trends
        trends = self.trend_data.get('top_trends', [])
        emerging_topics = self.trend_data.get('emerging_topics', [])
        
        # Convert business data to text for similarity matching
        business_text = ""
        business_text += self.business_data.get('description', '') + " "
        business_text += self.business_data.get('industry', '') + " "
        business_text += self.business_data.get('business_model', '') + " "
        
        # Add products/services
        for product in self.business_data.get('products', []):
            business_text += product.get('name', '') + " "
            business_text += product.get('description', '') + " "
        
        # Add capabilities
        for capability in self.business_data.get('capabilities', []):
            business_text += capability + " "
        
        # Find relevant trends
        relevant_trends = self._find_relevant_items(business_text, trends)
        
        # Create opportunities from relevant trends
        for trend in relevant_trends[:3]:  # Top 3 relevant trends
            trend_term = trend.get('term', '')
            trend_source = trend.get('source', 'Market Analysis')
            
            opportunity = {
                'opportunity': f"Capitalize on {trend_term} trend",
                'description': f"Align offerings with growing market interest in {trend_term}",
                'source': trend_source,
                'trend_strength': trend.get('trend_score', 0.5),
                'actions': [
                    f"Analyze how {trend_term} relates to current offerings",
                    f"Develop messaging highlighting relevance to {trend_term}",
                    f"Consider product enhancements to better address {trend_term}"
                ],
                'timeframe': 'Short-term',
                'potential_impact': 'Medium'
            }
            
            opportunities.append(opportunity)
        
        # Find relevant emerging topics
        relevant_emerging = self._find_relevant_items(business_text, emerging_topics)
        
        # Create opportunities from relevant emerging topics
        for topic in relevant_emerging[:2]:  # Top 2 relevant emerging topics
            topic_term = topic.get('term', '')
            
            opportunity = {
                'opportunity': f"Early mover in emerging {topic_term} space",
                'description': f"Position as early adopter of emerging {topic_term} trend before mainstream adoption",
                'source': 'Emerging Trend Analysis',
                'trend_strength': topic.get('emergence_score', 0.3),
                'actions': [
                    f"Monitor developments in {topic_term} space",
                    f"Develop pilot initiatives related to {topic_term}",
                    f"Build thought leadership around {topic_term}"
                ],
                'timeframe': 'Medium-term',
                'potential_impact': 'High'
            }
            
            opportunities.append(opportunity)
        
        # Add market-specific opportunities
        if self.market_data:
            # Underserved segments
            underserved_segments = self.market_data.get('underserved_segments', [])
            if underserved_segments:
                segment = underserved_segments[0]
                opportunities.append({
                    'opportunity': f"Target underserved {segment.get('name')} segment",
                    'description': f"Develop tailored offerings for {segment.get('name')} segment with limited current solutions",
                    'source': 'Market Analysis',
                    'trend_strength': 0.7,
                    'actions': [
                        f"Research specific needs of {segment.get('name')} segment",
                        "Develop targeted value proposition",
                        "Create specialized marketing and sales approach",
                        "Consider dedicated product features or services"
                    ],
                    'timeframe': 'Medium-term',
                    'potential_impact': 'Medium'
                })
            
            # Geographic expansion
            growing_regions = self.market_data.get('growing_regions', [])
            if growing_regions:
                region = growing_regions[0]
                opportunities.append({
                    'opportunity': f"Geographic expansion to {region.get('name')}",
                    'description': f"Enter {region.get('name')} market with {region.get('growth_rate')}% growth rate",
                    'source': 'Geographic Analysis',
                    'trend_strength': 0.6,
                    'actions': [
                        f"Conduct market assessment of {region.get('name')}",
                        "Develop region-specific go-to-market strategy",
                        "Identify potential partners or distribution channels",
                        "Adapt offerings to regional preferences if needed"
                    ],
                    'timeframe': 'Medium-term',
                    'potential_impact': 'Medium'
                })
            
            # Competitive gaps
            competitive_gaps = self.market_data.get('competitive_gaps', [])
            if competitive_gaps:
                gap = competitive_gaps[0]
                opportunities.append({
                    'opportunity': f"Address {gap.get('name')} gap in market",
                    'description': f"Develop offerings to address unmet need in {gap.get('name')}",
                    'source': 'Competitive Analysis',
                    'trend_strength': 0.8,
                    'actions': [
                        f"Validate market demand for {gap.get('name')} solution",
                        "Assess internal capabilities to address gap",
                        "Develop differentiated solution approach",
                        "Create go-to-market strategy emphasizing unique solution"
                    ],
                    'timeframe': 'Medium-term',
                    'potential_impact': 'High'
                })
        
        return opportunities
    
    def _find_relevant_items(self, business_text: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find items most relevant to the business based on text similarity"""
        if not items:
            return []
        
        # Extract terms from items
        item_terms = [item.get('term', '') for item in items]
        
        # Add business text as last item for comparison
        all_texts = item_terms + [business_text]
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            # Get similarity between business text and each item
            business_vector = tfidf_matrix[-1]
            item_vectors = tfidf_matrix[:-1]
            
            # Calculate cosine similarity
            similarities = cosine_similarity(business_vector, item_vectors)
            
            # Create list of (index, similarity) tuples and sort by similarity
            similarity_tuples = [(i, similarities[0, i]) for i in range(len(items))]
            similarity_tuples.sort(key=lambda x: x[1], reverse=True)
            
            # Return items sorted by relevance
            return [items[i] for i, _ in similarity_tuples]
            
        except Exception as e:
            logger.error(f"Error in finding relevant items: {str(e)}")
            return items  # Return original items if error occurs
    
    def _generate_product_recommendations(self) -> List[Dict[str, Any]]:
        """Generate product strategy recommendations"""
        recommendations = []
        
        # Get current products
        current_products = self.business_data.get('products', [])
        
        # Product enhancement recommendations
        if current_products:
            # Find products that could be enhanced
            for product in current_products:
                product_name = product.get('name', 'Product')
                product_performance = product.get('performance', {})
                
                # Check if product needs improvement
                if product_performance.get('growth', 0) < 5 or product_performance.get('satisfaction', 0) < 8:
                    recommendations.append({
                        'product': product_name,
                        'recommendation': f"Enhance {product_name} features and experience",
                        'rationale': f"Current performance metrics suggest opportunity for improvement",
                        'suggested_improvements': [
                            "Conduct customer research to identify pain points",
                            "Prioritize feature enhancements based on customer feedback",
                            "Improve user experience and interface design",
                            "Strengthen core value proposition"
                        ],
                        'expected_impact': 'Medium',
                        'timeframe': 'Short-term',
                        'resource_requirements': 'Medium'
                    })
                
                # Check if product could be expanded
                if product_performance.get('growth', 0) > 10 or product_performance.get('satisfaction', 0) > 8:
                    recommendations.append({
                        'product': product_name,
                        'recommendation': f"Expand {product_name} with additional features or versions",
                        'rationale': f"Strong performance suggests opportunity to build on success",
                        'suggested_improvements': [
                            "Develop premium version with advanced features",
                            "Create industry-specific variants",
                            "Expand to adjacent use cases",
                            "Develop complementary products or services"
                        ],
                        'expected_impact': 'High',
                        'timeframe': 'Medium-term',
                        'resource_requirements': 'Medium'
                    })
        
        # New product recommendations based on trends
        if self.trend_data and 'top_trends' in self.trend_data:
            top_trends = self.trend_data.get('top_trends', [])
            
            # Convert business capabilities to text
            capability_text = ""
            for capability in self.business_data.get('capabilities', []):
                capability_text += capability + " "
            
            # Find relevant trends for new product development
            relevant_trends = self._find_relevant_items(capability_text, top_trends)
            
            # Generate new product ideas based on trends
            for trend in relevant_trends[:2]:  # Top 2 relevant trends
                trend_term = trend.get('term', '')
                
                recommendations.append({
                    'product': f"New {trend_term} Solution",
                    'recommendation': f"Develop new offering focused on {trend_term}",
                    'rationale': f"Market trend analysis shows growing interest in {trend_term}",
                    'suggested_improvements': [
                        f"Conduct feasibility study for {trend_term} solution",
                        "Develop minimum viable product concept",
                        "Test with select customers for feedback",
                        "Create development and launch roadmap"
                    ],
                    'expected_impact': 'High',
                    'timeframe': 'Medium-term',
                    'resource_requirements': 'High'
                })
        
        # Pricing strategy recommendations
        if current_products:
            has_pricing_opportunity = False
            
            for product in current_products:
                if product.get('price_position', '') == 'low' and product.get('performance', {}).get('satisfaction', 0) > 7:
                    has_pricing_opportunity = True
            
            if has_pricing_opportunity:
                recommendations.append({
                    'product': 'Multiple Products',
                    'recommendation': "Optimize pricing strategy across product portfolio",
                    'rationale': "Current pricing may not fully capture value delivered to customers",
                    'suggested_improvements': [
                        "Conduct pricing analysis relative to perceived value",
                        "Implement value-based pricing strategy",
                        "Test price elasticity through controlled experiments",
                        "Consider tiered pricing structure for different segments"
                    ],
                    'expected_impact': 'High',
                    'timeframe': 'Short-term',
                    'resource_requirements': 'Low'
                })
        
        # Digital capabilities recommendation
        if self.business_data.get('digital_maturity', 0) < 7:
            recommendations.append({
                'product': 'All Products',
                'recommendation': "Enhance digital capabilities across product portfolio",
                'rationale': "Digital features can improve customer experience and create differentiation",
                'suggested_improvements': [
                    "Assess current digital capabilities in products",
                    "Identify opportunities for digital enhancement (mobile, API, automation)",
                    "Develop digital feature roadmap",
                    "Build or acquire necessary technical capabilities"
                ],
                'expected_impact': 'High',
                'timeframe': 'Medium-term',
                'resource_requirements': 'High'
            })
        
        # Product sunset recommendation
        if current_products and len(current_products) > 3:
            # Find underperforming products
            underperforming = []
            for product in current_products:
                if (product.get('performance', {}).get('growth', 0) < 0 and 
                    product.get('performance', {}).get('margin', 0) < 15):
                    underperforming.append(product.get('name', 'Product'))
            
            if underperforming:
                recommendations.append({
                    'product': ', '.join(underperforming),
                    'recommendation': f"Consider sunsetting or pivoting underperforming products",
                    'rationale': "Resources allocated to underperforming products could be redirected to higher-potential opportunities",
                    'suggested_improvements': [
                        "Evaluate strategic fit and future potential",
                        "Develop sunset plan for products without viable path forward",
                        "Consider pivoting products with valuable technology or customer base",
                        "Create migration plan for existing customers"
                    ],
                    'expected_impact': 'Medium',
                    'timeframe': 'Medium-term',
                    'resource_requirements': 'Medium'
                })
        
        return recommendations
    
    def _generate_priority_actions(self, growth_strategies: List[Dict[str, Any]],
                                  market_opportunities: List[Dict[str, Any]],
                                  product_recommendations: List[Dict[str, Any]]) -> List[str]:
        """Generate prioritized action list from all recommendations"""
        priority_actions = []
        
        # Add top growth strategy actions
        if growth_strategies:
            top_strategy = growth_strategies[0]
            priority_actions.append(f"Growth Strategy: {top_strategy.get('strategy')} - {top_strategy.get('actions')[0]}")
            
            if len(growth_strategies) > 1:
                second_strategy = growth_strategies[1]
                priority_actions.append(f"Growth Strategy: {second_strategy.get('strategy')} - {second_strategy.get('actions')[0]}")
        
        # Add top market opportunity actions
        if market_opportunities:
            top_opportunity = market_opportunities[0]
            priority_actions.append(f"Market Opportunity: {top_opportunity.get('opportunity')} - {top_opportunity.get('actions')[0]}")
        
        # Add top product recommendation actions
        if product_recommendations:
            top_product_rec = product_recommendations[0]
            priority_actions.append(f"Product Strategy: {top_product_rec.get('recommendation')} - {top_product_rec.get('suggested_improvements')[0]}")
        
        # Add additional high-impact actions
        for strategy in growth_strategies:
            if strategy.get('expected_impact') == 'High' and strategy.get('timeframe') == 'Short-term':
                action = f"High-Impact: {strategy.get('strategy')} - {strategy.get('actions')[0]}"
                if action not in priority_actions:
                    priority_actions.append(action)
        
        for opportunity in market_opportunities:
            if opportunity.get('potential_impact') == 'High' and opportunity.get('timeframe') == 'Short-term':
                action = f"High-Impact: {opportunity.get('opportunity')} - {opportunity.get('actions')[0]}"
                if action not in priority_actions:
                    priority_actions.append(action)
        
        for recommendation in product_recommendations:
            if recommendation.get('expected_impact') == 'High' and recommendation.get('timeframe') == 'Short-term':
                action = f"High-Impact: {recommendation.get('recommendation')} - {recommendation.get('suggested_improvements')[0]}"
                if action not in priority_actions:
                    priority_actions.append(action)
        
        return priority_actions[:10]  # Return top 10 priority actions
    
    def generate_competitive_strategy(self) -> Dict[str, Any]:
        """
        Generate competitive strategy recommendations
        
        Returns:
            Dictionary containing competitive strategy recommendations
        """
        if not self.business_data or not self.market_data:
            return {
                'error': 'Business data and market data must be set',
                'timestamp': datetime.now().isoformat()
            }
        
        recommendations = {
            'timestamp': datetime.now().isoformat(),
            'company_name': self.business_data.get('name', 'Unknown'),
            'industry': self.business_data.get('industry', 'Unknown'),
            'competitive_position': self._assess_competitive_position(),
            'differentiation_strategies': [],
            'competitor_specific_strategies': [],
            'defensive_strategies': [],
            'priority_actions': []
        }
        
        # Generate differentiation strategies
        differentiation_strategies = self._generate_differentiation_strategies()
        recommendations['differentiation_strategies'] = differentiation_strategies
        
        # Generate competitor-specific strategies
        competitor_strategies = self._generate_competitor_specific_strategies()
        recommendations['competitor_specific_strategies'] = competitor_strategies
        
        # Generate defensive strategies
        defensive_strategies = self._generate_defensive_strategies()
        recommendations['defensive_strategies'] = defensive_strategies
        
        # Generate priority actions
        priority_actions = self._generate_competitive_priority_actions(
            differentiation_strategies, competitor_strategies, defensive_strategies
        )
        recommendations['priority_actions'] = priority_actions
        
        return recommendations
    
    def _assess_competitive_position(self) -> Dict[str, Any]:
        """Assess current competitive position"""
        market_share = self.business_data.get('market_share', 0)
        market_growth = self.market_data.get('market_growth', 0)
        competitors = self.market_data.get('competitors', [])
        
        # Find largest competitor's market share
        largest_competitor_share = 0
        if competitors:
            largest_competitor_share = max(comp.get('market_share', 0) for comp in competitors)
        
        # Determine competitive position
        position = ""
        if market_share > largest_competitor_share:
            position = "Market Leader"
        elif market_share > largest_competitor_share * 0.7:
            position = "Strong Challenger"
        elif market_share > largest_competitor_share * 0.3:
            position = "Market Challenger"
        else:
            position = "Niche Player"
        
        # Determine market type
        market_type = ""
        if market_growth > 15:
            market_type = "High-Growth Market"
        elif market_growth > 5:
            market_type = "Growing Market"
        elif market_growth > 0:
            market_type = "Stable Market"
        else:
            market_type = "Declining Market"
        
        # Determine competitive intensity
        competitive_intensity = "Medium"
        if len(competitors) > 10:
            competitive_intensity = "High"
        elif len(competitors) < 5:
            competitive_intensity = "Low"
        
        # Calculate relative strength
        strengths = self.business_data.get('strengths', [])
        weaknesses = self.business_data.get('weaknesses', [])
        relative_strength = "Medium"
        if len(strengths) > len(weaknesses) * 2:
            relative_strength = "Strong"
        elif len(weaknesses) > len(strengths):
            relative_strength = "Weak"
        
        return {
            'position': position,
            'market_type': market_type,
            'market_share': market_share,
            'competitive_intensity': competitive_intensity,
            'relative_strength': relative_strength,
            'assessment': f"The company is a {position} in a {market_type} with {competitive_intensity} competitive intensity. Overall competitive strength is {relative_strength}."
        }
    
    def _generate_differentiation_strategies(self) -> List[Dict[str, Any]]:
        """Generate differentiation strategy recommendations"""
        strategies = []
        
        # Get business details
        strengths = self.business_data.get('strengths', [])
        business_model = self.business_data.get('business_model', 'unknown')
        capabilities = self.business_data.get('capabilities', [])
        
        # Value proposition differentiation
        strategies.append({
            'strategy': 'Value Proposition Differentiation',
            'description': 'Clearly articulate and strengthen unique value proposition',
            'rationale': 'A distinctive value proposition creates competitive advantage and reduces price sensitivity',
            'actions': [
                'Define and document core value proposition',
                'Validate value proposition with customer research',
                'Ensure consistent communication across all channels',
                'Train all customer-facing staff on value proposition'
            ],
            'expected_impact': 'High',
            'timeframe': 'Short-term',
            'resource_requirements': 'Low'
        })
        
        # Feature differentiation
        if 'product_innovation' in capabilities or 'r_and_d' in capabilities:
            strategies.append({
                'strategy': 'Feature and Functionality Differentiation',
                'description': 'Develop unique features that address unmet customer needs',
                'rationale': 'Distinctive features create competitive barriers and reduce direct comparison',
                'actions': [
                    'Identify high-value features not offered by competitors',
                    'Prioritize feature development based on customer impact',
                    'Patent or protect innovative features where possible',
                    'Highlight unique features in marketing and sales materials'
                ],
                'expected_impact': 'High',
                'timeframe': 'Medium-term',
                'resource_requirements': 'Medium'
            })
        
        # Service differentiation
        if 'customer_service' in strengths or 'customer_support' in capabilities:
            strategies.append({
                'strategy': 'Service Excellence Differentiation',
                'description': 'Create competitive advantage through superior service experience',
                'rationale': 'Service quality can be a powerful differentiator, especially in commoditized markets',
                'actions': [
                    'Define service standards that exceed industry norms',
                    'Invest in service training and capabilities',
                    'Implement service metrics and accountability',
                    'Create service guarantees or promises'
                ],
                'expected_impact': 'Medium',
                'timeframe': 'Medium-term',
                'resource_requirements': 'Medium'
            })
        
        # Brand differentiation
        strategies.append({
            'strategy': 'Brand Differentiation',
            'description': 'Build distinctive brand positioning and recognition',
            'rationale': 'Strong brand creates emotional connection and reduces price sensitivity',
            'actions': [
                'Define distinctive brand positioning',
                'Ensure consistent brand expression across touchpoints',
                'Invest in brand-building activities',
                'Measure and track brand equity over time'
            ],
            'expected_impact': 'High',
            'timeframe': 'Long-term',
            'resource_requirements': 'Medium'
        })
        
        # Channel differentiation
        if 'distribution' in capabilities or 'sales' in capabilities:
            strategies.append({
                'strategy': 'Channel and Access Differentiation',
                'description': 'Create advantage through superior access and distribution',
                'rationale': 'Unique distribution channels can provide competitive advantage',
                'actions': [
                    'Identify underserved access points or channels',
                    'Develop channel partnerships or capabilities',
                    'Create seamless omnichannel experience',
                    'Optimize channel economics and performance'
                ],
                'expected_impact': 'Medium',
                'timeframe': 'Medium-term',
                'resource_requirements': 'Medium'
            })
        
        # Price differentiation
        if 'cost_efficiency' in strengths or business_model in ['low_cost', 'volume_based']:
            strategies.append({
                'strategy': 'Price Leadership',
                'description': 'Maintain competitive advantage through superior pricing',
                'rationale': 'Price advantage can be compelling for price-sensitive segments',
                'actions': [
                    'Optimize cost structure to enable competitive pricing',
                    'Implement strategic pricing model',
                    'Communicate value-to-price ratio effectively',
                    'Monitor and respond to competitive pricing'
                ],
                'expected_impact': 'Medium',
                'timeframe': 'Short-term',
                'resource_requirements': 'Low'
            })
        elif 'premium_quality' in strengths or business_model in ['premium', 'luxury']:
            strategies.append({
                'strategy': 'Premium Positioning',
                'description': 'Maintain premium position with corresponding value delivery',
                'rationale': 'Premium positioning can yield higher margins and customer loyalty',
                'actions': [
                    'Ensure product quality and experience justifies premium',
                    'Create premium brand signals and experiences',
                    'Develop exclusive features or services',
                    'Avoid discounting that undermines premium positioning'
                ],
                'expected_impact': 'High',
                'timeframe': 'Medium-term',
                'resource_requirements': 'Medium'
            })
        
        return strategies
    
    def _generate_competitor_specific_strategies(self) -> List[Dict[str, Any]]:
        """Generate competitor-specific strategy recommendations"""
        strategies = []
        
        # Get competitors
        competitors = self.market_data.get('competitors', [])
        
        if not competitors:
            return strategies
        
        # Identify top competitors
        top_competitors = sorted(competitors, key=lambda x: x.get('market_share', 0), reverse=True)[:3]
        
        for competitor in top_competitors:
            competitor_name = competitor.get('name', 'Competitor')
            competitor_strengths = competitor.get('strengths', [])
            competitor_weaknesses = competitor.get('weaknesses', [])
            
            # Skip if we don't have enough information
            if not competitor_strengths and not competitor_weaknesses:
                continue
            
            # Create strategy based on competitor weaknesses
            if competitor_weaknesses:
                top_weakness = competitor_weaknesses[0] if competitor_weaknesses else "Unknown"
                
                strategies.append({
                    'competitor': competitor_name,
                    'strategy': f"Exploit {competitor_name}'s weaknesses",
                    'description': f"Target {competitor_name}'s vulnerability in {top_weakness}",
                    'rationale': f"Directly addressing competitor weakness can win customers and market share",
                    'actions': [
                        f"Highlight your strength in {top_weakness} area in competitive situations",
                        f"Develop specific messaging addressing {top_weakness}",
                        f"Train sales team on competitive positioning against {competitor_name}",
                        f"Monitor {competitor_name}'s efforts to address this weakness"
                    ],
                    'expected_impact': 'Medium',
                    'timeframe': 'Short-term',
                    'resource_requirements': 'Low'
                })
            
            # Create strategy based on competitor strengths
            if competitor_strengths:
                top_strength = competitor_strengths[0] if competitor_strengths else "Unknown"
                
                strategies.append({
                    'competitor': competitor_name,
                    'strategy': f"Neutralize {competitor_name}'s strengths",
                    'description': f"Reduce competitive disadvantage in {top_strength}",
                    'rationale': f"Minimizing competitive advantage reduces customer loss and protects market share",
                    'actions': [
                        f"Assess capability gap in {top_strength} area",
                        f"Develop plan to improve capabilities or offer alternatives",
                        f"Create messaging that reframes the importance of {top_strength}",
                        f"Develop partnerships or solutions that address this gap"
                    ],
                    'expected_impact': 'Medium',
                    'timeframe': 'Medium-term',
                    'resource_requirements': 'Medium'
                })
        
        # Add general competitive strategy
        strategies.append({
            'competitor': 'All Competitors',
            'strategy': 'Competitive Intelligence Program',
            'description': 'Develop systematic approach to gathering and using competitive intelligence',
            'rationale': 'Better competitive information enables more effective strategy and tactics',
            'actions': [
                'Establish competitive intelligence gathering process',
                'Create competitive intelligence database',
                'Develop regular competitive analysis reports',
                'Integrate competitive insights into strategy and planning'
            ],
            'expected_impact': 'High',
            'timeframe': 'Medium-term',
            'resource_requirements': 'Medium'
        })
        
        return strategies
    def _generate_defensive_strategies(self) -> List[Dict[str, Any]]:
        """Generate defensive strategy recommendations"""
        strategies = []
        
        # Get business details
        market_share = self.business_data.get('market_share', 0)
        market_position = self._assess_competitive_position().get('position', '')
        
        # Customer retention strategy
        strategies.append({
            'strategy': 'Customer Retention Program',
            'description': 'Strengthen relationships with existing customers to reduce churn',
            'rationale': 'Retaining existing customers is more cost-effective than acquiring new ones',
            'actions': [
                'Implement customer success program for key accounts',
                'Develop early warning system for at-risk customers',
                'Create loyalty incentives and programs',
                'Establish regular customer feedback mechanisms'
            ],
            'expected_impact': 'High',
            'timeframe': 'Short-term',
            'resource_requirements': 'Medium'
        })
        
        # Entry barriers strategy
        if market_position in ['Market Leader', 'Strong Challenger']:
            strategies.append({
                'strategy': 'Raise Competitive Entry Barriers',
                'description': 'Create structural advantages that are difficult for competitors to overcome',
                'rationale': 'Strong entry barriers protect market position and margins',
                'actions': [
                    'Secure exclusive partnerships or distribution channels',
                    'Invest in proprietary technology or patents',
                    'Create network effects or ecosystem advantages',
                    'Establish economies of scale in key areas'
                ],
                'expected_impact': 'High',
                'timeframe': 'Long-term',
                'resource_requirements': 'High'
            })
        
        # Switching cost strategy
        strategies.append({
            'strategy': 'Increase Customer Switching Costs',
            'description': 'Create valuable features and integrations that make switching costly',
            'rationale': 'Higher switching costs reduce customer churn and competitive threats',
            'actions': [
                'Develop deep product integrations with customer systems',
                'Create proprietary data formats or repositories',
                'Build customer-specific customizations',
                'Implement volume-based incentives or long-term contracts'
            ],
            'expected_impact': 'Medium',
            'timeframe': 'Medium-term',
            'resource_requirements': 'Medium'
        })
        
        # Competitive signaling strategy
        if market_position in ['Market Leader', 'Strong Challenger']:
            strategies.append({
                'strategy': 'Strategic Competitive Signaling',
                'description': 'Signal competitive intent to discourage competitive threats',
                'rationale': 'Clear signals can deter competitive actions and protect market position',
                'actions': [
                    'Publicly announce strategic investments or initiatives',
                    'Respond quickly and visibly to competitive threats',
                    'Demonstrate commitment to key market segments',
                    'Establish reputation for competitive response'
                ],
                'expected_impact': 'Medium',
                'timeframe': 'Medium-term',
                'resource_requirements': 'Low'
            })
        
        # Ecosystem strategy
        strategies.append({
            'strategy': 'Build Defensive Ecosystem',
            'description': 'Create network of partners and complementary offerings',
            'rationale': 'Strong ecosystem creates mutual dependencies and competitive barriers',
            'actions': [
                'Identify strategic partnership opportunities',
                'Develop partner program and incentives',
                'Create integration capabilities and APIs',
                'Co-market with ecosystem partners'
            ],
            'expected_impact': 'High',
            'timeframe': 'Long-term',
            'resource_requirements': 'Medium'
        })
        
        # Preemptive innovation strategy
        strategies.append({
            'strategy': 'Preemptive Innovation',
            'description': 'Anticipate and address emerging customer needs before competitors',
            'rationale': 'First-mover advantage can create lasting competitive advantage',
            'actions': [
                'Establish innovation scouting and research program',
                'Create rapid prototyping and testing capabilities',
                'Develop fast-track process for promising innovations',
                'Allocate resources for strategic innovation initiatives'
            ],
            'expected_impact': 'High',
            'timeframe': 'Medium-term',
            'resource_requirements': 'High'
        })
        
        return strategies
    
    def _generate_competitive_priority_actions(self, differentiation_strategies: List[Dict[str, Any]],
                                             competitor_strategies: List[Dict[str, Any]],
                                             defensive_strategies: List[Dict[str, Any]]) -> List[str]:
        """Generate prioritized competitive action list"""
        priority_actions = []
        
        # Add top differentiation strategy actions
        if differentiation_strategies:
            top_strategy = differentiation_strategies[0]
            priority_actions.append(f"Differentiation: {top_strategy.get('strategy')} - {top_strategy.get('actions')[0]}")
            
            if len(differentiation_strategies) > 1:
                second_strategy = differentiation_strategies[1]
                priority_actions.append(f"Differentiation: {second_strategy.get('strategy')} - {second_strategy.get('actions')[0]}")
        
        # Add top competitor strategy actions
        if competitor_strategies:
            top_strategy = competitor_strategies[0]
            priority_actions.append(f"Competitive: {top_strategy.get('strategy')} - {top_strategy.get('actions')[0]}")
        
        # Add top defensive strategy actions
        if defensive_strategies:
            top_strategy = defensive_strategies[0]
            priority_actions.append(f"Defensive: {top_strategy.get('strategy')} - {top_strategy.get('actions')[0]}")
        
        # Add additional high-impact actions
        for strategy in differentiation_strategies + competitor_strategies + defensive_strategies:
            if strategy.get('expected_impact') == 'High' and strategy.get('timeframe') == 'Short-term':
                action = f"High-Impact: {strategy.get('strategy')} - {strategy.get('actions')[0]}"
                if action not in priority_actions:
                    priority_actions.append(action)
        
        return priority_actions[:10]  # Return top 10 priority actions