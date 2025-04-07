# app/modules/trend_detector/scraper.py
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
import os
import time
from datetime import datetime, timedelta
import re
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

class TrendScraper:
    """
    Scraper for collecting trend data from various sources
    """
    
    def __init__(self, news_api_key: Optional[str] = None, twitter_api_key: Optional[str] = None):
        self.news_api_key = news_api_key or os.environ.get('NEWS_API_KEY')
        self.twitter_api_key = twitter_api_key or os.environ.get('TWITTER_API_KEY')
        self.twitter_api_secret = os.environ.get('TWITTER_API_SECRET')
        self.user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        
        if not self.news_api_key:
            logger.warning("No News API key provided. News scraping will be limited.")
            
        if not self.twitter_api_key or not self.twitter_api_secret:
            logger.warning("Twitter API credentials not provided. Twitter scraping will be limited.")
    
    def scrape_news_articles(self, keywords: List[str], days_back: int = 7, 
                            limit: int = 100) -> List[Dict[str, Any]]:
        """
        Scrape news articles related to specified keywords
        
        Args:
            keywords: List of keywords to search for
            days_back: Number of days to look back
            limit: Maximum number of articles to return
            
        Returns:
            List of dictionaries containing article data
        """
        articles = []
        
        if self.news_api_key:
            # Use News API if key is available
            articles = self._scrape_with_news_api(keywords, days_back, limit)
        else:
            # Fallback to web scraping (with caution regarding rate limits)
            articles = self._scrape_news_from_web(keywords, days_back, limit)
            
        return articles
    
    def _scrape_with_news_api(self, keywords: List[str], days_back: int = 7, 
                             limit: int = 100) -> List[Dict[str, Any]]:
        """Scrape news using News API"""
        try:
            # Prepare query parameters
            query = ' OR '.join(keywords)
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            # Make API request
            url = 'https://newsapi.org/v2/everything'
            params = {
                'q': query,
                'from': from_date,
                'sortBy': 'relevancy',
                'language': 'en',
                'apiKey': self.news_api_key,
                'pageSize': min(100, limit)  # API limit is 100 per request
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if data['status'] == 'ok':
                    articles = []
                    
                    for article in data['articles'][:limit]:
                        processed_article = {
                            'title': article.get('title', ''),
                            'description': article.get('description', ''),
                            'content': article.get('content', ''),
                            'url': article.get('url', ''),
                            'source': article.get('source', {}).get('name', 'Unknown'),
                            'published_at': article.get('publishedAt', ''),
                            'keywords': keywords,
                            'scraped_at': datetime.now().isoformat()
                        }
                        articles.append(processed_article)
                        
                    logger.info(f"Scraped {len(articles)} articles using News API")
                    return articles
                else:
                    logger.error(f"News API error: {data.get('message', 'Unknown error')}")
                    return []
            else:
                logger.error(f"News API HTTP error: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error scraping news with News API: {str(e)}")
            return []
        
    def _scrape_news_from_web(self, keywords: List[str], days_back: int = 7, 
                             limit: int = 100) -> List[Dict[str, Any]]:
        """Fallback method to scrape news from web sources"""
        articles = []
        
        # List of news sources to scrape (public pages only, respecting robots.txt)
        news_sources = [
            {'name': 'Reuters Business', 'url': 'https://www.reuters.com/business/'},
            {'name': 'BBC Business', 'url': 'https://www.bbc.com/news/business'},
            {'name': 'CNBC', 'url': 'https://www.cnbc.com/business/'}
        ]
        
        try:
            for source in news_sources:
                if len(articles) >= limit:
                    break
                    
                logger.info(f"Scraping news from {source['name']}")
                
                # Send request with appropriate headers
                headers = {'User-Agent': self.user_agent}
                response = requests.get(source['url'], headers=headers)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Extract links - this is a simplified approach and needs customization for each site
                    links = soup.find_all('a', href=True)
                    article_links = []
                    
                    for link in links:
                        # Filter for article links (this pattern varies by site)
                        href = link['href']
                        if any(pattern in href for pattern in ['/article/', '/story/', '/news/']):
                            # Make absolute URL if needed
                            if not href.startswith('http'):
                                href = urljoin(source['url'], href)
                                
                            article_links.append(href)
                    
                    # Remove duplicates
                    article_links = list(set(article_links))
                    
                    # Process each article
                    for link in article_links[:min(10, limit - len(articles))]:  # Limit per source
                        try:
                            # Add delay to avoid overloading the server
                            time.sleep(1)
                            
                            # Fetch article content
                            article_response = requests.get(link, headers=headers)
                            
                            if article_response.status_code == 200:
                                article_soup = BeautifulSoup(article_response.text, 'html.parser')
                                
                                # Extract article details (this varies by site)
                                title = article_soup.find('h1')
                                title_text = title.text.strip() if title else "Unknown Title"
                                
                                # Check if article matches keywords
                                if any(keyword.lower() in title_text.lower() for keyword in keywords):
                                    # Extract other elements
                                    description = ""
                                    desc_elem = article_soup.find('meta', {'name': 'description'}) or article_soup.find('meta', {'property': 'og:description'})
                                    if desc_elem and 'content' in desc_elem.attrs:
                                        description = desc_elem['content']
                                        
                                    # Extract date (this varies by site)
                                    date_text = ""
                                    date_elem = article_soup.find('time') or article_soup.find('span', {'class': 'date'})
                                    if date_elem:
                                        date_text = date_elem.text.strip()
                                    
                                    # Extract content (simplified)
                                    content = ""
                                    article_body = article_soup.find('article') or article_soup.find('div', {'class': ['article-body', 'story-body']})
                                    if article_body:
                                        paragraphs = article_body.find_all('p')
                                        content = ' '.join([p.text.strip() for p in paragraphs])
                                    
                                    # Create article object
                                    article = {
                                        'title': title_text,
                                        'description': description,
                                        'content': content[:1000] + '...' if len(content) > 1000 else content,
                                        'url': link,
                                        'source': source['name'],
                                        'published_at': date_text,
                                        'keywords': [k for k in keywords if k.lower() in (title_text + ' ' + description + ' ' + content).lower()],
                                        'scraped_at': datetime.now().isoformat()
                                    }
                                    
                                    articles.append(article)
                                    
                                    if len(articles) >= limit:
                                        break
                        except Exception as e:
                            logger.warning(f"Error processing article {link}: {str(e)}")
                            continue
                else:
                    logger.warning(f"Failed to fetch {source['name']}: HTTP {response.status_code}")
            
            logger.info(f"Scraped {len(articles)} articles from web sources")
            return articles
                
        except Exception as e:
            logger.error(f"Error scraping news from web: {str(e)}")
            return []
    
    def scrape_social_media_trends(self, keywords: List[str], 
                                  days_back: int = 7) -> Dict[str, Any]:
        """
        Scrape trending topics and posts from social media
        
        Args:
            keywords: List of keywords to search for
            days_back: Number of days to look back
            
        Returns:
            Dictionary containing trend data from social media
        """
        results = {
            'twitter': [],
            'linkedin': [],
            'reddit': [],
            'scraped_at': datetime.now().isoformat()
        }
        
        # Try to get Twitter data if API keys are available
        if self.twitter_api_key and self.twitter_api_secret:
            results['twitter'] = self._scrape_twitter_trends(keywords, days_back)
        else:
            logger.warning("Skipping Twitter trend collection due to missing API credentials")
        
        # Reddit data collection (public API)
        results['reddit'] = self._scrape_reddit_trends(keywords, days_back)
        
        return results
    
    def _scrape_twitter_trends(self, keywords: List[str], days_back: int = 7) -> List[Dict[str, Any]]:
        """Scrape trending topics and tweets from Twitter"""
        try:
            # Note: This is a placeholder for Twitter API v2 integration
            # In a real implementation, you would use the Twitter API client
            # This would require proper authentication with API keys
            
            logger.warning("Twitter API integration not fully implemented")
            
            # Return placeholder data
            return [
                {
                    'keyword': keyword,
                    'tweet_count': 0,
                    'trending_score': 0,
                    'top_tweets': [],
                    'sentiment': {'positive': 0, 'neutral': 0, 'negative': 0}
                }
                for keyword in keywords
            ]
            
        except Exception as e:
            logger.error(f"Error scraping Twitter trends: {str(e)}")
            return []
    
    def _scrape_reddit_trends(self, keywords: List[str], days_back: int = 7) -> List[Dict[str, Any]]:
        """Scrape trending topics and posts from Reddit"""
        reddit_trends = []
        
        try:
            # Relevant subreddits for business trends
            subreddits = ['business', 'economics', 'finance', 'technology', 'startups', 'entrepreneur']
            
            for keyword in keywords:
                # Initialize data structure for this keyword
                keyword_data = {
                    'keyword': keyword,
                    'post_count': 0,
                    'top_posts': [],
                    'subreddits': {},
                    'sentiment': {'positive': 0, 'neutral': 0, 'negative': 0}
                }
                
                # Search across subreddits
                for subreddit in subreddits:
                    try:
                        # Use Reddit's JSON API (no authentication required for public data)
                        url = f"https://www.reddit.com/r/{subreddit}/search.json"
                        params = {
                            'q': keyword,
                            't': 'week',  # Time period (week ~= 7 days)
                            'sort': 'relevance',
                            'limit': 10
                        }
                        
                        headers = {'User-Agent': self.user_agent}
                        response = requests.get(url, params=params, headers=headers)
                        
                        if response.status_code == 200:
                            data = response.json()
                            posts = data.get('data', {}).get('children', [])
                            
                            subreddit_posts = []
                            for post in posts:
                                post_data = post.get('data', {})
                                
                                # Extract relevant post data
                                post_info = {
                                    'title': post_data.get('title', ''),
                                    'url': f"https://www.reddit.com{post_data.get('permalink', '')}",
                                    'score': post_data.get('score', 0),
                                    'comments': post_data.get('num_comments', 0),
                                    'created_utc': post_data.get('created_utc', 0),
                                    'author': post_data.get('author', 'unknown')
                                }
                                
                                subreddit_posts.append(post_info)
                                
                                # Add to top posts if it's among the highest scoring
                                keyword_data['top_posts'].append(post_info)
                            
                            # Sort top posts by score and limit to top 10
                            keyword_data['top_posts'] = sorted(
                                keyword_data['top_posts'], 
                                key=lambda x: x['score'], 
                                reverse=True
                            )[:10]
                            
                            # Add subreddit-specific data
                            keyword_data['subreddits'][subreddit] = {
                                'post_count': len(subreddit_posts),
                                'total_score': sum(p['score'] for p in subreddit_posts),
                                'total_comments': sum(p['comments'] for p in subreddit_posts)
                            }
                            
                            # Update overall post count
                            keyword_data['post_count'] += len(subreddit_posts)
                            
                            # Add delay to avoid rate limiting
                            time.sleep(1)
                            
                    except Exception as e:
                        logger.warning(f"Error scraping Reddit for {keyword} in r/{subreddit}: {str(e)}")
                        continue
                
                # Add this keyword's data to results
                reddit_trends.append(keyword_data)
                
            return reddit_trends
            
        except Exception as e:
            logger.error(f"Error scraping Reddit trends: {str(e)}")
            return []
    
    def scrape_industry_reports(self, industry: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find recent industry reports and analyses
        
        Args:
            industry: Industry to search for
            limit: Maximum number of reports to return
            
        Returns:
            List of dictionaries containing report data
        """
        reports = []
        
        try:
            # List of sources for industry reports
            report_sources = [
                {'name': 'McKinsey & Company', 'url': 'https://www.mckinsey.com/industries/'},
                {'name': 'Deloitte Insights', 'url': 'https://www2.deloitte.com/us/en/insights/industry/'},
                {'name': 'PwC', 'url': 'https://www.pwc.com/gx/en/industries/'}
            ]
            
            for source in report_sources:
                if len(reports) >= limit:
                    break
                    
                # Construct industry-specific URL (this varies by source)
                industry_slug = industry.lower().replace(' ', '-')
                industry_url = urljoin(source['url'], industry_slug)
                
                # Send request
                headers = {'User-Agent': self.user_agent}
                response = requests.get(industry_url, headers=headers)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Look for report links (this pattern varies by site)
                    # Targeting elements that likely contain reports
                    report_containers = soup.find_all(['article', 'div'], class_=re.compile(r'(report|insight|article|publication)'))
                    
                    if not report_containers:
                        # Fallback to generic link search
                        links = soup.find_all('a', href=True)
                        
                        for link in links:
                            if len(reports) >= limit:
                                break
                                
                            href = link['href']
                            title = link.text.strip()
                            
                            # Filter for likely report links
                            if (('report' in href.lower() or 'insight' in href.lower() or 'analysis' in href.lower()) and
                                len(title) > 20):  # Assuming report titles are reasonably long
                                
                                # Make absolute URL if needed
                                if not href.startswith('http'):
                                    href = urljoin(industry_url, href)
                                
                                # Create report object
                                report = {
                                    'title': title,
                                    'url': href,
                                    'source': source['name'],
                                    'industry': industry,
                                    'scraped_at': datetime.now().isoformat(),
                                    'summary': ''  # Would require additional request to get summary
                                }
                                
                                reports.append(report)
                    else:
                        # Process structured report containers
                        for container in report_containers:
                            if len(reports) >= limit:
                                break
                                
                            # Extract report details
                            title_elem = container.find(['h1', 'h2', 'h3', 'h4', 'h5'])
                            title = title_elem.text.strip() if title_elem else "Unknown Report"
                            
                            link_elem = container.find('a', href=True)
                            if not link_elem:
                                continue
                                
                            href = link_elem['href']
                            
                            # Make absolute URL if needed
                            if not href.startswith('http'):
                                href = urljoin(industry_url, href)
                            
                            # Extract summary if available
                            summary_elem = container.find(['p', 'div'], class_=re.compile(r'(summary|description|excerpt)'))
                            summary = summary_elem.text.strip() if summary_elem else ""
                            
                            # Create report object
                            report = {
                                'title': title,
                                'url': href,
                                'source': source['name'],
                                'industry': industry,
                                'scraped_at': datetime.now().isoformat(),
                                'summary': summary
                            }
                            
                            reports.append(report)
                            
                else:
                    logger.warning(f"Failed to fetch {source['name']}: HTTP {response.status_code}")
            
            logger.info(f"Scraped {len(reports)} industry reports")
            return reports
                
        except Exception as e:
            logger.error(f"Error scraping industry reports: {str(e)}")
            return []
    
    def scrape_economic_indicators(self) -> Dict[str, Any]:
        """
        Scrape current economic indicators
        
        Returns:
            Dictionary containing economic indicator data
        """
        indicators = {
            'gdp_growth': None,
            'inflation_rate': None,
            'unemployment_rate': None,
            'interest_rate': None,
            'consumer_confidence': None,
            'stock_market_indices': {},
            'currency_exchange_rates': {},
            'scraped_at': datetime.now().isoformat()
        }
        
        try:
            # Scrape from reliable public source
            url = 'https://tradingeconomics.com/united-states/indicators'
            headers = {'User-Agent': self.user_agent}
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract indicators (this is specific to the Trading Economics layout)
                tables = soup.find_all('table', {'class': 'table-striped'})
                
                for table in tables:
                    rows = table.find_all('tr')
                    
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) >= 3:
                            indicator_name = cells[0].text.strip().lower()
                            indicator_value = cells[1].text.strip()
                            
                            # Map to our indicator structure
                            if 'gdp growth' in indicator_name:
                                indicators['gdp_growth'] = self._extract_numeric_value(indicator_value)
                            elif 'inflation' in indicator_name:
                                indicators['inflation_rate'] = self._extract_numeric_value(indicator_value)
                            elif 'unemployment' in indicator_name:
                                indicators['unemployment_rate'] = self._extract_numeric_value(indicator_value)
                            elif 'interest rate' in indicator_name:
                                indicators['interest_rate'] = self._extract_numeric_value(indicator_value)
                            elif 'consumer confidence' in indicator_name:
                                indicators['consumer_confidence'] = self._extract_numeric_value(indicator_value)
            
            # Scrape stock market indices
            stock_url = 'https://finance.yahoo.com/world-indices/'
            stock_response = requests.get(stock_url, headers=headers)
            
            if stock_response.status_code == 200:
                soup = BeautifulSoup(stock_response.text, 'html.parser')
                
                # Find the main table
                table = soup.find('table')
                
                if table:
                    rows = table.find_all('tr')
                    
                    for row in rows[1:]:  # Skip header row
                        cells = row.find_all('td')
                        if len(cells) >= 3:
                            try:
                                index_name = cells[0].text.strip()
                                index_value = self._extract_numeric_value(cells[1].text.strip())
                                index_change = self._extract_numeric_value(cells[2].text.strip())
                                
                                indicators['stock_market_indices'][index_name] = {
                                    'value': index_value,
                                    'change_percent': index_change
                                }
                            except:
                                continue
            
            # Scrape currency exchange rates
            currency_url = 'https://finance.yahoo.com/currencies/'
            currency_response = requests.get(currency_url, headers=headers)
            
            if currency_response.status_code == 200:
                soup = BeautifulSoup(currency_response.text, 'html.parser')
                
                # Find the main table
                table = soup.find('table')
                
                if table:
                    rows = table.find_all('tr')
                    
                    for row in rows[1:]:  # Skip header row
                        cells = row.find_all('td')
                        if len(cells) >= 3:
                            try:
                                currency_pair = cells[0].text.strip()
                                rate = self._extract_numeric_value(cells[1].text.strip())
                                
                                indicators['currency_exchange_rates'][currency_pair] = rate
                            except:
                                continue
            
            return indicators
                
        except Exception as e:
            logger.error(f"Error scraping economic indicators: {str(e)}")
            return indicators
    
    def _extract_numeric_value(self, text: str) -> Optional[float]:
        """Extract numeric value from text"""
        try:
            # Remove % signs and commas
            cleaned = text.replace('%', '').replace(',', '')
            # Extract numeric part
            match = re.search(r'-?\d+\.?\d*', cleaned)
            if match:
                return float(match.group())
            return None
        except:
            return None