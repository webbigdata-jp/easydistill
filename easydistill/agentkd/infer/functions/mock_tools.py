
import re
import os
import time
import json
import requests
from typing import Any, Optional, Dict, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import sqlite3
from tavily import TavilyClient
from pathlib import Path

load_dotenv()


class SearchResult(BaseModel):
    id: str
    title: str
    url: str
    snippet: str
    source: str
    display_link: Optional[str] = None
    formatted_url: Optional[str] = None

class SearchResponse(BaseModel):
    success: bool
    query: str
    results: List[SearchResult]
    count: int
    search_time: float
    message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class GoogleSearch:
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("missing GOOGLE_API_KEY")
        
        # your search url
        self.api_url = None
        if not self.api_url:
            raise ValueError("missing api_url")
    
    def search(
        self,
        query: str,
        num_results: int = 5,
        language: str = "zh-cn",
        country: str = "cn",
        safe_search: bool = True
    ) -> SearchResponse:
        if not query or not query.strip():
            return SearchResponse(
                success=False,
                query=query,
                results=[],
                count=0,
                search_time=0,
                message="search query is None"
            )
        
        query = query.strip()
        num_results = max(1, min(num_results, 10))
        
        start_time = time.time()
        
        try:
            headers = {
                'X-AK': self.api_key,
                'Content-Type': 'application/json'
            }
            
            data = {
                'query': query,
                'num': num_results,
                'extendParams': {
                    'country': country,
                    'locale': language,
                },
                'platformInput': {
                    'model': 'google-search'
                }
            }
            
           
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            
            search_time = time.time() - start_time
            
            
            json_data = response.json()
            search_results = []
            
            if "data" in json_data and 'originalOutput' in json_data['data']:
                organic_results = json_data['data']['originalOutput'].get('organic', [])
                
                for i, item in enumerate(organic_results):
                    search_results.append(SearchResult(
                        id=f"google-{i}",
                        title=item.get("title", ""),
                        url=item.get("link", ""),
                        snippet=item.get("snippet", ""),
                        source="google",
                        display_link=item.get("displayLink"),
                        formatted_url=item.get("formattedUrl")
                    ))
            
            return SearchResponse(
                success=True,
                query=query,
                results=search_results,
                count=len(search_results),
                search_time=search_time,
                metadata={
                    "language": language,
                    "country": country,
                    "safe_search": safe_search,
                    "search_engine": "google"
                }
            )
            
        except requests.exceptions.Timeout:
            return SearchResponse(
                success=False,
                query=query,
                results=[],
                count=0,
                search_time=time.time() - start_time,
                message="time out"
            )
        except requests.exceptions.RequestException as e:
            return SearchResponse(
                success=False,
                query=query,
                results=[],
                count=0,
                search_time=time.time() - start_time,
                message=f"wrong api: {str(e)}"
            )
        except Exception as e:
            return SearchResponse(
                success=False,
                query=query,
                results=[],
                count=0,
                search_time=time.time() - start_time,
                message=f"search failed: {str(e)}"
            )
    
    def search_simple(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:

        response = self.search(query, num_results)
        
        if response.success:
            return [
                {
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet
                }
                for r in response.results
            ]
        else:
            return []

def quick_search(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    try:
        searcher = GoogleSearch()
        return searcher.search_simple(query, num_results)
    except Exception as e:
        print(f"search Error: {e}")
        return []



class MockTools:
    _cache_file = _cache_file = Path(__file__).parent / "search_cache.db"
    _db_initialized = False
    
    @staticmethod
    def _init_cache_db():
        """Initialize the cache database only once"""
        if MockTools._db_initialized:
            return
            
        with sqlite3.connect(MockTools._cache_file) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS search_cache (
                    query TEXT PRIMARY KEY,
                    results TEXT,
                    timestamp REAL
                )
                """
            )
            conn.commit()
        MockTools._db_initialized = True
    
    @staticmethod
    def _get_cached_results(query: str) -> Optional[list]:
        """Retrieve cached search results"""
        try:
            with sqlite3.connect(MockTools._cache_file) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT results FROM search_cache WHERE query = ?",
                    (query,)
                )
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
            return None
        except Exception:
            return None
    
    @staticmethod
    def _cache_results(query: str, results: list):
        """Cache search results"""
        try:
            with sqlite3.connect(MockTools._cache_file) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO search_cache (query, results, timestamp)
                    VALUES (?, ?, ?)
                    """,
                    (query, json.dumps(results), time.time())
                )
                conn.commit()
        except Exception:
            pass 
    
    @staticmethod
    def web_search(query):
        """web search with cache support"""
        MockTools._init_cache_db()

        query = query.strip().lower()
        query = re.sub(r"[^\w\s]", "", query)
        
        search_dicts = MockTools._get_cached_results(query)
        if search_dicts is None:
            search_dicts = quick_search(query, num_results=10)
        else:
            print("Retrieved from cache.\n")

        retrieved_content = ""
        if len(search_dicts) == 0:
            retrieved_content = "Failed to retrieve content. Please try again later."
        else:
            MockTools._cache_results(query, search_dicts)
            for idx, search_res in enumerate(search_dicts):
                retrieved_content += f"Page: {idx}\nTitle: {search_res['title']}\nSnippet: {search_res['snippet']}\n"
        print(retrieved_content.strip())
    @staticmethod
    def final_answer_print(answer):
        """Only for final answer"""
        print(answer)


if __name__ == "__main__":
    small_agent_input_query = "Tim Russert death location"
    content = MockTools.web_search(small_agent_input_query)
    print(content)
