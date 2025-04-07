# app/modules/trend_detector/trend_analyzer.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
import os
import re
from datetime import datetime, timedelta
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import io
import base64

logger = logging.getLogger(__name__)

# Initialize NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    logger.warning("Could not download NLTK resources. Text analysis may be limited.")

class TrendAnalyzer:
    """
    Analyze scraped data to identify business trends
    """
    
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        except:
            logger.warning("Could not initialize NLTK components. Using simplified text processing.")
            self.stop_words = set(['a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 'when', 'where', 'how', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'can', 'could', 'will', 'would', 'shall', 'should', 'may', 'might', 'must', 'to', 'of', 'in', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'from', 'up', 'down', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'just', 'don', 'now'])
            self.lemmatizer = None
    
    def analyze_news_articles(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze news articles to identify trends and key topics
        
        Args:
            articles: List of dictionaries containing article data
            
        Returns:
            Dictionary containing trend analysis results
        """
        if not articles:
            return {
                'trends': [],
                'top_sources': [],
                'topic_clusters': [],
                'sentiment': {'positive': 0, 'neutral': 0, 'negative': 0},
                'word_frequencies': {},
                'timestamp': datetime.now().isoformat()
            }
        
        # Extract text for analysis
        texts = []
        titles = []
        sources = []
        dates = []
        
        for article in articles:
            titles.append(article.get('title', ''))
            texts.append(article.get('content', '') or article.get('description', ''))
            sources.append(article.get('source', 'Unknown'))
            dates.append(article.get('published_at', ''))
        
        # Identify top sources
        source_counts = Counter(sources)
        top_sources = [{'source': source, 'count': count} for source, count in source_counts.most_common(10)]
        
        # Process text
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # Extract key phrases and topics
        topics = self._extract_topics(processed_texts)
        word_freq = self._calculate_word_frequencies(processed_texts)
        
        # Identify trends based on frequency and recency
        trends = self._identify_trends(processed_texts, titles, dates)
        
        # Generate word cloud
        wordcloud_img = self._generate_wordcloud(word_freq)
        
        # Perform sentiment analysis
        sentiment = self._analyze_sentiment(texts)
        
        # Create result object
        result = {
            'trends': trends,
            'top_sources': top_sources,
            'topic_clusters': topics,
            'sentiment': sentiment,
            'word_frequencies': dict(word_freq.most_common(50)),
            'wordcloud': wordcloud_img,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis"""
        if not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Tokenize
        try:
            tokens = word_tokenize(text)
        except:
            # Fallback to simple splitting
            tokens = text.split()
        
        # Remove stopwords
        tokens = [word for word in tokens if word not in self.stop_words]
        
        # Lemmatize
        if self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        # Rejoin
        return ' '.join(tokens)
    
    def _calculate_word_frequencies(self, texts: List[str]) -> Counter:
        """Calculate word frequencies across all texts"""
        all_words = []
        
        for text in texts:
            words = text.split()
            # Filter out very short words
            words = [word for word in words if len(word) > 2]
            all_words.extend(words)
        
        return Counter(all_words)
    
    def _extract_topics(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Extract main topics using LDA"""
        try:
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(max_features=1000, min_df=2, max_df=0.8)
            X = vectorizer.fit_transform(texts)
            
            # Apply LDA
            n_topics = min(5, len(texts) // 2) if len(texts) > 2 else 2
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            lda.fit(X)
            
            # Extract feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Extract topics
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[:-11:-1]  # Top 10 words
                top_words = [feature_names[i] for i in top_words_idx]
                
                topics.append({
                    'id': topic_idx,
                    'top_words': top_words,
                    'weight': float(topic.sum())
                })
            
            return topics
            
        except Exception as e:
            logger.error(f"Error extracting topics: {str(e)}")
            return []
    def _identify_trends(self, processed_texts: List[str], titles: List[str], 
                        dates: List[str]) -> List[Dict[str, Any]]:
        """Identify trends based on frequency and recency"""
        # Combine processed texts with titles for better trend identification
        combined_texts = []
        for i in range(len(processed_texts)):
            title = titles[i].lower() if titles[i] else ""
            # Give more weight to words in titles by repeating them
            combined_texts.append(title + " " + title + " " + processed_texts[i])
        
        # Calculate word frequencies
        word_freq = self._calculate_word_frequencies(combined_texts)
        
        # Extract bigrams (two-word phrases)
        bigrams = []
        for text in combined_texts:
            words = text.split()
            if len(words) > 1:
                for i in range(len(words) - 1):
                    bigrams.append(words[i] + " " + words[i+1])
        
        bigram_freq = Counter(bigrams)
        
        # Combine unigrams and bigrams, favoring bigrams
        trend_candidates = {}
        
        # Add top bigrams
        for bigram, count in bigram_freq.most_common(20):
            if count >= 2:  # Require at least 2 occurrences
                trend_candidates[bigram] = {
                    'count': count,
                    'is_bigram': True
                }
        
        # Add top unigrams that aren't part of frequent bigrams
        for word, count in word_freq.most_common(30):
            if count >= 3:  # Require at least 3 occurrences
                # Check if word is part of a frequent bigram
                is_in_bigram = any(word in bigram for bigram in trend_candidates.keys())
                
                if not is_in_bigram:
                    trend_candidates[word] = {
                        'count': count,
                        'is_bigram': False
                    }
        
        # Score trends based on frequency and recency
        trends = []
        
        for term, data in trend_candidates.items():
            # Calculate recency score based on when the term appears in articles
            recency_score = 0
            term_articles = 0
            
            for i, text in enumerate(combined_texts):
                if term in text:
                    term_articles += 1
                    # Give higher weight to more recent articles
                    try:
                        article_date = datetime.fromisoformat(dates[i].replace('Z', '+00:00'))
                        days_ago = (datetime.now() - article_date).days
                        recency_score += max(0, 7 - days_ago) / 7  # Scale from 0 to 1
                    except:
                        recency_score += 0.5  # Default if date parsing fails
            
            # Normalize recency score
            if term_articles > 0:
                recency_score /= term_articles
            
            # Calculate overall trend score
            frequency_score = min(1.0, data['count'] / 10)  # Scale frequency, cap at 1.0
            trend_score = (frequency_score * 0.7) + (recency_score * 0.3)  # Weight frequency more
            
            trends.append({
                'term': term,
                'frequency': data['count'],
                'recency_score': float(recency_score),
                'trend_score': float(trend_score),
                'is_bigram': data['is_bigram']
            })
        
        # Sort by trend score
        trends.sort(key=lambda x: x['trend_score'], reverse=True)
        
        # Return top trends
        return trends[:15]
    
    def _analyze_sentiment(self, texts: List[str]) -> Dict[str, float]:
        """Perform basic sentiment analysis"""
        try:
            # Load sentiment lexicons
            positive_words = set([
                'good', 'great', 'excellent', 'positive', 'success', 'successful', 'profit', 'profitable',
                'growth', 'growing', 'increase', 'increasing', 'improved', 'improving', 'improvement',
                'opportunity', 'opportunities', 'benefit', 'beneficial', 'advantage', 'advantages',
                'innovation', 'innovative', 'efficient', 'efficiency', 'effective', 'gain', 'gains',
                'progress', 'progressive', 'leading', 'leader', 'best', 'better', 'strong', 'strength',
                'strengthen', 'strengthening', 'up', 'higher', 'highest', 'top', 'optimal', 'optimistic'
            ])
            
            negative_words = set([
                'bad', 'poor', 'negative', 'fail', 'failure', 'loss', 'losing', 'decline', 'declining',
                'decrease', 'decreasing', 'worsen', 'worsening', 'worst', 'worse', 'problem', 'problematic',
                'challenge', 'challenging', 'difficult', 'difficulty', 'threat', 'threatening', 'risk', 'risky',
                'danger', 'dangerous', 'concern', 'concerning', 'worried', 'worry', 'anxious', 'anxiety',
                'fear', 'fearful', 'weak', 'weakness', 'weakening', 'down', 'lower', 'lowest', 'bottom',
                'crisis', 'critical', 'emergency', 'recession', 'depression', 'bankrupt', 'bankruptcy'
            ])
            
            # Count sentiments
            positive_count = 0
            negative_count = 0
            neutral_count = 0
            
            for text in texts:
                text_lower = text.lower()
                
                # Count positive and negative words
                pos_words = sum(1 for word in positive_words if f" {word} " in f" {text_lower} ")
                neg_words = sum(1 for word in negative_words if f" {word} " in f" {text_lower} ")
                
                # Determine sentiment of this text
                if pos_words > neg_words:
                    positive_count += 1
                elif neg_words > pos_words:
                    negative_count += 1
                else:
                    neutral_count += 1
            
            # Calculate percentages
            total = len(texts)
            if total > 0:
                positive_pct = (positive_count / total) * 100
                negative_pct = (negative_count / total) * 100
                neutral_pct = (neutral_count / total) * 100
            else:
                positive_pct = negative_pct = neutral_pct = 0
                
            return {
                'positive': float(positive_pct),
                'negative': float(negative_pct),
                'neutral': float(neutral_pct)
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return {'positive': 0, 'negative': 0, 'neutral': 100}
    
    def _generate_wordcloud(self, word_freq: Counter) -> str:
        """Generate word cloud image from word frequencies"""
        try:
            # Create word cloud
            wordcloud = WordCloud(width=800, height=400, background_color='white',
                                 max_words=100, contour_width=3, contour_color='steelblue')
            
            # Generate from frequencies
            wordcloud.generate_from_frequencies(word_freq)
            
            # Convert to image
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            
            # Save to bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # Convert to base64 for embedding in JSON
            img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
            
            return img_str
            
        except Exception as e:
            logger.error(f"Error generating word cloud: {str(e)}")
            return ""
    
    def analyze_social_trends(self, social_data: Dict[str, Any], 
                             industry: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze social media trend data
        
        Args:
            social_data: Dictionary containing social media data
            industry: Optional industry for context
            
        Returns:
            Dictionary containing trend analysis results
        """
        result = {
            'top_trends': [],
            'platform_insights': {},
            'industry_context': industry,
            'sentiment': {'positive': 0, 'neutral': 0, 'negative': 0},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Process Twitter data if available
            twitter_trends = []
            if 'twitter' in social_data and social_data['twitter']:
                twitter_trends = self._analyze_twitter_trends(social_data['twitter'])
                result['platform_insights']['twitter'] = twitter_trends
            
            # Process Reddit data if available
            reddit_trends = []
            if 'reddit' in social_data and social_data['reddit']:
                reddit_trends = self._analyze_reddit_trends(social_data['reddit'])
                result['platform_insights']['reddit'] = reddit_trends
            
            # Combine trends from all platforms
            all_trends = twitter_trends + reddit_trends
            
            # Sort by overall trend score
            all_trends.sort(key=lambda x: x.get('trend_score', 0), reverse=True)
            
            # Take top trends
            result['top_trends'] = all_trends[:10]
            
            # Calculate overall sentiment
            sentiments = []
            if 'twitter' in result['platform_insights']:
                sentiments.append(result['platform_insights']['twitter'].get('sentiment', {}))
            if 'reddit' in result['platform_insights']:
                sentiments.append(result['platform_insights']['reddit'].get('sentiment', {}))
                
            if sentiments:
                result['sentiment'] = {
                    'positive': sum(s.get('positive', 0) for s in sentiments) / len(sentiments),
                    'negative': sum(s.get('negative', 0) for s in sentiments) / len(sentiments),
                    'neutral': sum(s.get('neutral', 0) for s in sentiments) / len(sentiments)
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing social trends: {str(e)}")
            return result
    
    def _analyze_twitter_trends(self, twitter_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze Twitter trend data"""
        try:
            trends = []
            
            for item in twitter_data:
                keyword = item.get('keyword', '')
                tweet_count = item.get('tweet_count', 0)
                trending_score = item.get('trending_score', 0)
                
                # Skip items with no activity
                if tweet_count == 0:
                    continue
                
                # Create trend object
                trend = {
                    'term': keyword,
                    'source': 'Twitter',
                    'frequency': tweet_count,
                    'trend_score': trending_score,
                    'sentiment': item.get('sentiment', {})
                }
                
                trends.append(trend)
            
            # Calculate overall sentiment
            sentiments = [item.get('sentiment', {}) for item in twitter_data if item.get('sentiment')]
            overall_sentiment = {}
            
            if sentiments:
                overall_sentiment = {
                    'positive': sum(s.get('positive', 0) for s in sentiments) / len(sentiments),
                    'negative': sum(s.get('negative', 0) for s in sentiments) / len(sentiments),
                    'neutral': sum(s.get('neutral', 0) for s in sentiments) / len(sentiments)
                }
            
            return {
                'trends': trends,
                'sentiment': overall_sentiment
            }
            
        except Exception as e:
            logger.error(f"Error analyzing Twitter trends: {str(e)}")
            return {'trends': [], 'sentiment': {}}
    
    def _analyze_reddit_trends(self, reddit_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze Reddit trend data"""
        try:
            trends = []
            
            for item in reddit_data:
                keyword = item.get('keyword', '')
                post_count = item.get('post_count', 0)
                
                # Skip items with no activity
                if post_count == 0:
                    continue
                
                # Calculate trend score based on post count and engagement
                subreddit_data = item.get('subreddits', {})
                total_score = sum(data.get('total_score', 0) for data in subreddit_data.values())
                total_comments = sum(data.get('total_comments', 0) for data in subreddit_data.values())
                
                # Normalize to a 0-1 scale
                normalized_posts = min(1.0, post_count / 50)
                normalized_score = min(1.0, total_score / 1000)
                normalized_comments = min(1.0, total_comments / 500)
                
                # Weight the factors
                trend_score = (normalized_posts * 0.3) + (normalized_score * 0.4) + (normalized_comments * 0.3)
                
                # Get top posts
                top_posts = item.get('top_posts', [])
                
                # Create trend object
                trend = {
                    'term': keyword,
                    'source': 'Reddit',
                    'frequency': post_count,
                    'trend_score': float(trend_score),
                    'engagement': {
                        'total_score': total_score,
                        'total_comments': total_comments
                    },
                    'top_posts': top_posts[:3] if top_posts else []
                }
                
                trends.append(trend)
            
            # Sort by trend score
            trends.sort(key=lambda x: x['trend_score'], reverse=True)
            
            # Estimate sentiment (very basic approach)
            positive_count = negative_count = neutral_count = 0
            
            for trend in trends:
                # Look at post titles for sentiment cues
                for post in trend.get('top_posts', []):
                    title = post.get('title', '').lower()
                    
                    # Very simple sentiment detection
                    positive_words = ['good', 'great', 'awesome', 'excellent', 'amazing', 'positive', 'success']
                    negative_words = ['bad', 'poor', 'terrible', 'awful', 'negative', 'fail', 'failure']
                    
                    pos_count = sum(1 for word in positive_words if word in title)
                    neg_count = sum(1 for word in negative_words if word in title)
                    
                    if pos_count > neg_count:
                        positive_count += 1
                    elif neg_count > pos_count:
                        negative_count += 1
                    else:
                        neutral_count += 1
            
            # Calculate sentiment percentages
            total_posts = positive_count + negative_count + neutral_count
            sentiment = {}
            
            if total_posts > 0:
                sentiment = {
                    'positive': (positive_count / total_posts) * 100,
                    'negative': (negative_count / total_posts) * 100,
                    'neutral': (neutral_count / total_posts) * 100
                }
            else:
                sentiment = {'positive': 0, 'negative': 0, 'neutral': 100}
            
            return {
                'trends': trends,
                'sentiment': sentiment
            }
            
        except Exception as e:
            logger.error(f"Error analyzing Reddit trends: {str(e)}")
            return {'trends': [], 'sentiment': {}}
    
    def analyze_industry_reports(self, reports: List[Dict[str, Any]], 
                               industry: str) -> Dict[str, Any]:
        """
        Analyze industry reports to extract key insights
        
        Args:
            reports: List of dictionaries containing report data
            industry: Industry context
            
        Returns:
            Dictionary containing analysis results
        """
        result = {
            'industry': industry,
            'report_count': len(reports),
            'key_sources': [],
            'major_themes': [],
            'timestamp': datetime.now().isoformat()
        }
        
        if not reports:
            return result
            
        try:
            # Extract sources
            sources = [report.get('source', 'Unknown') for report in reports]
            source_counts = Counter(sources)
            result['key_sources'] = [{'name': source, 'count': count} 
                                   for source, count in source_counts.most_common()]
            
            # Extract text for analysis
            texts = []
            for report in reports:
                title = report.get('title', '')
                summary = report.get('summary', '')
                texts.append(title + " " + summary)
            
            # Preprocess texts
            processed_texts = [self._preprocess_text(text) for text in texts]
            
            # Extract themes using topic modeling
            if len(processed_texts) >= 3:  # Need enough documents for meaningful topics
                try:
                    vectorizer = TfidfVectorizer(max_features=1000, min_df=1, max_df=0.9)
                    X = vectorizer.fit_transform(processed_texts)
                    
                    n_topics = min(5, len(processed_texts))
                    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
                    lda.fit(X)
                    
                    feature_names = vectorizer.get_feature_names_out()
                    
                    for topic_idx, topic in enumerate(lda.components_):
                        top_words_idx = topic.argsort()[:-11:-1]
                        top_words = [feature_names[i] for i in top_words_idx]
                        
                        result['major_themes'].append({
                            'id': topic_idx,
                            'keywords': top_words,
                            'weight': float(topic.sum())
                        })
                except Exception as e:
                    logger.warning(f"Could not perform topic modeling on reports: {str(e)}")
            
            # Fallback to simple word frequency if topic modeling fails
            if not result['major_themes']:
                word_freq = self._calculate_word_frequencies(processed_texts)
                top_words = [word for word, _ in word_freq.most_common(20)]
                
                # Group into pseudo-themes
                chunk_size = 5
                for i in range(0, len(top_words), chunk_size):
                    theme_words = top_words[i:i+chunk_size]
                    if theme_words:
                        result['major_themes'].append({
                            'id': i // chunk_size,
                            'keywords': theme_words,
                            'weight': 1.0 - (i / len(top_words))  # Higher weight to earlier words
                        })
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing industry reports: {str(e)}")
            return result
    
    def analyze_economic_indicators(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze economic indicators for business impact
        
        Args:
            indicators: Dictionary containing economic indicator data
            
        Returns:
            Dictionary containing analysis results
        """
        result = {
            'indicators': indicators,
            'business_impact': {},
            'recommendations': [],
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Analyze GDP growth
            gdp_growth = indicators.get('gdp_growth')
            if gdp_growth is not None:
                if gdp_growth >= 3:
                    impact = "Positive - Strong economic growth suggests favorable business conditions"
                    recommendation = "Consider expansion strategies to capitalize on economic growth"
                elif gdp_growth > 0:
                    impact = "Neutral to Positive - Moderate economic growth provides stability"
                    recommendation = "Maintain current strategies with cautious optimism"
                elif gdp_growth > -2:
                    impact = "Neutral to Negative - Slow or negative growth may constrain business opportunities"
                    recommendation = "Focus on efficiency and cost management"
                else:
                    impact = "Negative - Significant economic contraction may reduce demand"
                    recommendation = "Implement defensive strategies to preserve cash and market position"
                    
                result['business_impact']['gdp_growth'] = impact
                result['recommendations'].append(recommendation)
            
            # Analyze inflation rate
            inflation_rate = indicators.get('inflation_rate')
            if inflation_rate is not None:
                if inflation_rate > 5:
                    impact = "Negative - High inflation may increase costs and reduce purchasing power"
                    recommendation = "Review pricing strategies and supply chain costs"
                elif inflation_rate > 2:
                    impact = "Neutral - Moderate inflation is manageable but warrants attention"
                    recommendation = "Monitor cost trends and consider gradual price adjustments"
                else:
                    impact = "Positive - Low inflation provides stable cost environment"
                    recommendation = "Focus on volume growth rather than price increases"
                    
                result['business_impact']['inflation_rate'] = impact
                result['recommendations'].append(recommendation)
            
            # Analyze interest rate
            interest_rate = indicators.get('interest_rate')
            if interest_rate is not None:
                if interest_rate > 5:
                    impact = "Negative - High interest rates increase borrowing costs and may reduce investment"
                    recommendation = "Minimize debt and focus on organic growth"
                elif interest_rate > 2:
                    impact = "Neutral - Moderate interest rates allow for strategic borrowing"
                    recommendation = "Balance debt and equity financing for growth initiatives"
                else:
                    impact = "Positive - Low interest rates provide favorable financing conditions"
                    recommendation = "Consider strategic investments and long-term financing"
                    
                result['business_impact']['interest_rate'] = impact
                result['recommendations'].append(recommendation)
            
            # Analyze unemployment rate
            unemployment_rate = indicators.get('unemployment_rate')
            if unemployment_rate is not None:
                if unemployment_rate > 7:
                    impact = "Mixed - High unemployment may reduce consumer spending but increase labor availability"
                    recommendation = "Evaluate talent acquisition opportunities while monitoring consumer segments"
                elif unemployment_rate > 4:
                    impact = "Neutral - Moderate unemployment provides labor market balance"
                    recommendation = "Maintain competitive compensation while monitoring productivity"
                else:
                    impact = "Mixed - Low unemployment supports consumer spending but may increase labor costs"
                    recommendation = "Focus on employee retention and productivity improvements"
                    
                result['business_impact']['unemployment_rate'] = impact
                result['recommendations'].append(recommendation)
            
            # Analyze stock market performance
            stock_indices = indicators.get('stock_market_indices', {})
            if stock_indices:
                # Calculate average market performance
                changes = [data.get('change_percent', 0) for data in stock_indices.values()]
                avg_change = sum(changes) / len(changes) if changes else 0
                
                if avg_change > 10:
                    impact = "Positive - Strong market performance indicates investor confidence"
                    recommendation = "Consider equity financing for growth initiatives"
                elif avg_change > 0:
                    impact = "Neutral to Positive - Stable market conditions support business confidence"
                    recommendation = "Maintain balanced capital structure"
                else:
                    impact = "Neutral to Negative - Market weakness may affect investor sentiment"
                    recommendation = "Focus on operational performance to maintain investor confidence"
                    
                result['business_impact']['stock_market'] = impact
                result['recommendations'].append(recommendation)
            
            # Overall economic assessment
            impacts = list(result['business_impact'].values())
            positive_count = sum(1 for impact in impacts if impact.startswith("Positive"))
            negative_count = sum(1 for impact in impacts if impact.startswith("Negative"))
            
            if positive_count > negative_count:
                result['overall_assessment'] = "Generally favorable economic conditions for business"
            elif negative_count > positive_count:
                result['overall_assessment'] = "Challenging economic conditions requiring careful management"
            else:
                result['overall_assessment'] = "Mixed economic signals suggesting a balanced approach"
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing economic indicators: {str(e)}")
            return result
    
    def generate_trend_report(self, news_analysis: Dict[str, Any], 
                             social_analysis: Dict[str, Any],
                             industry_analysis: Dict[str, Any],
                             economic_analysis: Dict[str, Any],
                             industry: str) -> Dict[str, Any]:
        """
        Generate comprehensive trend report combining all analyses
        
        Args:
            news_analysis: Results from analyze_news_articles
            social_analysis: Results from analyze_social_trends
            industry_analysis: Results from analyze_industry_reports
            economic_analysis: Results from analyze_economic_indicators
            industry: Industry context
            
        Returns:
            Dictionary containing comprehensive trend report
        """
        report = {
            'industry': industry,
            'timestamp': datetime.now().isoformat(),
            'top_trends': [],
            'emerging_topics': [],
            'economic_context': economic_analysis.get('overall_assessment', ''),
            'sentiment_analysis': {},
            'strategic_implications': [],
            'data_sources': {
                'news_articles': news_analysis.get('trends', []),
                'social_media': social_analysis.get('top_trends', []),
                'industry_reports': industry_analysis.get('report_count', 0),
                'economic_indicators': list(economic_analysis.get('indicators', {}).keys())
            }
        }
        
        try:
            # Combine trends from different sources
            all_trends = []
            
            # Add news trends
            for trend in news_analysis.get('trends', [])[:10]:
                all_trends.append({
                    'term': trend.get('term', ''),
                    'source': 'News',
                    'score': trend.get('trend_score', 0),
                    'frequency': trend.get('frequency', 0)
                })
            
            # Add social trends
            for trend in social_analysis.get('top_trends', [])[:10]:
                all_trends.append({
                    'term': trend.get('term', ''),
                    'source': trend.get('source', 'Social Media'),
                    'score': trend.get('trend_score', 0),
                    'frequency': trend.get('frequency', 0)
                })
            
            # Add industry themes as trends
            for i, theme in enumerate(industry_analysis.get('major_themes', [])[:5]):
                theme_term = ' '.join(theme.get('keywords', [])[:3])
                all_trends.append({
                    'term': theme_term,
                    'source': 'Industry Reports',
                    'score': theme.get('weight', 0),
                    'keywords': theme.get('keywords', [])
                })
            
            # Sort by score and remove duplicates
            seen_terms = set()
            unique_trends = []
            
            for trend in sorted(all_trends, key=lambda x: x.get('score', 0), reverse=True):
                term = trend.get('term', '').lower()
                # Check for similar terms (to avoid near-duplicates)
                if not any(term in seen or seen in term for seen in seen_terms):
                    seen_terms.add(term)
                    unique_trends.append(trend)
            
            # Take top trends
            report['top_trends'] = unique_trends[:15]
            
            # Identify emerging topics (lower frequency but growing)
            news_topics = news_analysis.get('topic_clusters', [])
            social_trends = social_analysis.get('top_trends', [])
            
            emerging_candidates = []
            
            # Add lower-ranked news trends that show recency
            for trend in news_analysis.get('trends', [])[10:20]:
                if trend.get('recency_score', 0) > 0.7:  # High recency
                    emerging_candidates.append({
                        'term': trend.get('term', ''),
                        'source': 'News',
                        'recency': trend.get('recency_score', 0),
                        'emergence_score': trend.get('recency_score', 0) * 0.8  # Weight recency highly
                    })
            
            # Add topics from news that aren't in top trends
            top_terms = {t.get('term', '').lower() for t in report['top_trends']}
            for topic in news_topics:
                topic_terms = topic.get('top_words', [])
                if topic_terms and not any(term in top_terms for term in topic_terms[:3]):
                    emerging_candidates.append({
                        'term': ' '.join(topic_terms[:3]),
                        'source': 'News Topics',
                        'keywords': topic_terms,
                        'emergence_score': 0.6  # Default score
                    })
            
            # Sort by emergence score
            emerging_candidates.sort(key=lambda x: x.get('emergence_score', 0), reverse=True)
            report['emerging_topics'] = emerging_candidates[:8]
            
            # Combine sentiment analysis
            news_sentiment = news_analysis.get('sentiment', {})
            social_sentiment = social_analysis.get('sentiment', {})
            
            if news_sentiment and social_sentiment:
                # Weight news higher than social for overall sentiment
                report['sentiment_analysis'] = {
                    'positive': (news_sentiment.get('positive', 0) * 0.6) + (social_sentiment.get('positive', 0) * 0.4),
                    'neutral': (news_sentiment.get('neutral', 0) * 0.6) + (social_sentiment.get('neutral', 0) * 0.4),
                    'negative': (news_sentiment.get('negative', 0) * 0.6) + (social_sentiment.get('negative', 0) * 0.4),
                    'sources': {
                        'news': news_sentiment,
                        'social': social_sentiment
                    }
                }
            elif news_sentiment:
                report['sentiment_analysis'] = news_sentiment
            elif social_sentiment:
                report['sentiment_analysis'] = social_sentiment
            
            # Generate strategic implications
            report['strategic_implications'] = self._generate_strategic_implications(
                report['top_trends'],
                report['emerging_topics'],
                economic_analysis,
                report['sentiment_analysis'],
                industry
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating trend report: {str(e)}")
            return report
    def _generate_strategic_implications(self, top_trends: List[Dict[str, Any]],
                                        emerging_topics: List[Dict[str, Any]],
                                        economic_analysis: Dict[str, Any],
                                        sentiment_analysis: Dict[str, Any],
                                        industry: str) -> List[Dict[str, Any]]:
        """Generate strategic implications from trend analysis"""
        implications = []
        
        try:
            # Implication from top trends
            if top_trends:
                top_trend_terms = [t.get('term', '') for t in top_trends[:3]]
                trend_implication = {
                    'area': 'Market Trends',
                    'implication': f"The {industry} industry is seeing significant focus on {', '.join(top_trend_terms)}",
                    'recommendation': f"Evaluate how your business strategy addresses these dominant trends",
                    'urgency': 'High'
                }
                implications.append(trend_implication)
            
            # Implication from emerging topics
            if emerging_topics:
                emerging_terms = [t.get('term', '') for t in emerging_topics[:3]]
                emerging_implication = {
                    'area': 'Emerging Opportunities',
                    'implication': f"Early signals indicate growing interest in {', '.join(emerging_terms)}",
                    'recommendation': f"Monitor these emerging topics for potential opportunities",
                    'urgency': 'Medium'
                }
                implications.append(emerging_implication)
            
            # Implication from economic context
            if economic_analysis.get('overall_assessment'):
                econ_recommendations = economic_analysis.get('recommendations', [])
                top_econ_recommendation = econ_recommendations[0] if econ_recommendations else "Adjust strategies based on economic conditions"
                
                econ_implication = {
                    'area': 'Economic Environment',
                    'implication': economic_analysis.get('overall_assessment'),
                    'recommendation': top_econ_recommendation,
                    'urgency': 'Medium'
                }
                implications.append(econ_implication)
            
            # Implication from sentiment analysis
            if sentiment_analysis:
                positive = sentiment_analysis.get('positive', 0)
                negative = sentiment_analysis.get('negative', 0)
                
                sentiment_message = ""
                sentiment_recommendation = ""
                urgency = "Medium"
                
                if positive > 60:
                    sentiment_message = f"Overall sentiment toward {industry} is strongly positive"
                    sentiment_recommendation = "Leverage positive sentiment in marketing and communications"
                    urgency = "Low"
                elif positive > negative:
                    sentiment_message = f"Overall sentiment toward {industry} is moderately positive"
                    sentiment_recommendation = "Maintain current positioning while addressing negative factors"
                    urgency = "Low"
                elif negative > 60:
                    sentiment_message = f"Overall sentiment toward {industry} shows significant concerns"
                    sentiment_recommendation = "Address negative sentiment factors in communications and strategy"
                    urgency = "High"
                else:
                    sentiment_message = f"Overall sentiment toward {industry} is mixed or neutral"
                    sentiment_recommendation = "Differentiate by addressing specific concerns in your market"
                    urgency = "Medium"
                
                sentiment_implication = {
                    'area': 'Market Sentiment',
                    'implication': sentiment_message,
                    'recommendation': sentiment_recommendation,
                    'urgency': urgency
                }
                implications.append(sentiment_implication)
            
            # Add industry-specific implication
            industry_lower = industry.lower()
            if 'technology' in industry_lower or 'software' in industry_lower:
                tech_implication = {
                    'area': 'Technology Evolution',
                    'implication': "Technology sectors experience rapid innovation cycles and competitive pressure",
                    'recommendation': "Maintain innovation focus and monitor competitive landscape closely",
                    'urgency': 'High'
                }
                implications.append(tech_implication)
            elif 'retail' in industry_lower or 'consumer' in industry_lower:
                retail_implication = {
                    'area': 'Consumer Behavior',
                    'implication': "Consumer preferences and shopping behaviors continue to evolve",
                    'recommendation': "Focus on omnichannel experience and personalization capabilities",
                    'urgency': 'High'
                }
                implications.append(retail_implication)
            elif 'finance' in industry_lower or 'banking' in industry_lower:
                finance_implication = {
                    'area': 'Regulatory Environment',
                    'implication': "Financial sectors face ongoing regulatory scrutiny and compliance requirements",
                    'recommendation': "Prioritize regulatory compliance and risk management frameworks",
                    'urgency': 'High'
                }
                implications.append(finance_implication)
            elif 'health' in industry_lower or 'medical' in industry_lower:
                health_implication = {
                    'area': 'Healthcare Innovation',
                    'implication': "Healthcare continues to see technological innovation and cost pressures",
                    'recommendation': "Balance innovation with cost-effectiveness in solutions",
                    'urgency': 'Medium'
                }
                implications.append(health_implication)
            
            return implications
            
        except Exception as e:
            logger.error(f"Error generating strategic implications: {str(e)}")
            return [
                {
                    'area': 'General Strategy',
                    'implication': f"Market conditions in {industry} continue to evolve",
                    'recommendation': "Maintain flexibility and monitor industry developments",
                    'urgency': 'Medium'
                }
            ]