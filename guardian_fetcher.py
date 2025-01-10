from functools import lru_cache
import requests
import pandas as pd
from datetime import datetime, timedelta
import os

class GuardianFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://content.guardianapis.com/search"
    
    @lru_cache(maxsize=32)
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

    def fetch_articles(self, days_back=30, page_size=50):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        all_articles = []
        page = 1
        total_pages = 1

        while page <= total_pages and page <= 5:
            try:
                data = self._fetch_page(page, start_date_str, end_date_str, page_size)
                
                if page == 1:
                    total_pages = min(data['pages'], 5)
                
                for article in data['results']:
                    if 'fields' in article:
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
                
            except Exception as e:
                print(f"Error fetching page {page}: {str(e)}")
                break

        df = pd.DataFrame(all_articles)
        df['days_ago'] = (datetime.now() - df['published']).dt.days
        
        print(f"📰 Fetched {len(df)} articles")
        return df
