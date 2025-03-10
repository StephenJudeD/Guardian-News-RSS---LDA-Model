from functools import lru_cache
import requests
import pandas as pd
from datetime import datetime, timedelta
import os

class GuardianFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://content.guardianapis.com/search"
    
    #@lru_cache(maxsize=32)
    def _fetch_page(self, page, start_date_str, end_date_str, page_size):
        """Cached method for fetching individual pages"""
        params = {
            'api-key': self.api_key,
            'show-fields': 'bodyText,headline,byline,wordcount,thumbnail',
            'page-size': page_size,
            'order-by': 'relevance',
            'from-date': start_date_str,
            'to-date': end_date_str,
            'page': page
        }
        response = requests.get(self.base_url, params=params)
        return response.json()['response']

    def fetch_articles(self, start_date, end_date, page_size=100, max_pages=10):
        """
        Fetch Guardian articles within the specified date range.
        """
        all_articles = []
        page = 1
        total_pages = 1
    
        while page <= total_pages and page <= max_pages:
            try:
                start_date_str = start_date.strftime('%Y-%m-%d')
                end_date_str = end_date.strftime('%Y-%m-%d')
    
                params = {
                    'api-key': self.api_key,
                    'show-fields': 'bodyText,headline,byline,wordcount,thumbnail',
                    'page-size': page_size,
                    'order-by': 'relevance',
                    'from-date': start_date_str,
                    'to-date': end_date_str,
                    'page': page
                }
                response = requests.get(self.base_url, params=params)
                data = response.json()['response']
    
                if page == 1:
                    total_pages = data['pages']
    
                for article in data['results']:
                    if 'fields' in article and article['sectionName'] in ["World news", "US news", "Football", "Sport", "UK news"]:
                        all_articles.append({
                            'title': article['webTitle'],
                            'content': article['fields'].get('bodyText', ''),
                            'section': article['sectionName'],
                            'published': datetime.strptime(
                                article['webPublicationDate'], 
                                '%Y-%m-%dT%H:%M:%SZ'
                            ),
                            'wordcount': article['fields'].get('wordcount', 0),
                            'byline': article['fields'].get('byline', ''),
                            'thumbnail': article['fields'].get('thumbnail', '')
                        })
    
                page += 1
                time.sleep(1)  # Add a delay to avoid hitting rate limits
    
            except Exception as e:
                print(f"Error fetching page {page}: {str(e)}")
                break
    
        df = pd.DataFrame(all_articles)
        return df
