import asyncio
import aiohttp
import feedparser
from datetime import datetime, timedelta
from typing import List, Dict
import json
from pathlib import Path
import logging
from parsers.parser_manager import NewsParserManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RSSParser:
    def __init__(self, urls_file: str, output_file: str = 'rss_feed_24h.json', fetch_content: bool = True):
        self.urls_file = urls_file
        self.output_file = output_file
        self.time_threshold = datetime.now() - timedelta(hours=24)
        self.fetch_content = fetch_content
        self.news_parser = NewsParserManager() if fetch_content else None
        
    async def fetch_rss(self, session: aiohttp.ClientSession, url: str) -> Dict:
        try:
            logger.info(f"Загрузка RSS: {url}")
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    content = await response.text()
                    return {'url': url, 'content': content, 'error': None}
                else:
                    logger.error(f"Ошибка {response.status} для {url}")
                    return {'url': url, 'content': None, 'error': f"HTTP {response.status}"}
        except Exception as e:
            logger.error(f"Ошибка при загрузке {url}: {str(e)}")
            return {'url': url, 'content': None, 'error': str(e)}
    
    async def fetch_article_content(self, session: aiohttp.ClientSession, url: str) -> str:
        try:
            # logger.info(f"Извлечение текста из: {url}")
            text = await self.news_parser.extract_text(session, url)
            # logger.info(f"{url}: {text}")
            return text if text else ""
        except Exception as e:
            # logger.error(f"Ошибка извлечения текста из {url}: {str(e)}")
            return ""
    
    def parse_feed(self, feed_data: Dict) -> List[Dict]:
        if feed_data['content'] is None:
            return []
        
        try:
            feed = feedparser.parse(feed_data['content'])
            recent_entries = []
            
            for entry in feed.entries:
                published_date = None
                
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    published_date = datetime(*entry.published_parsed[:6])
                elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                    published_date = datetime(*entry.updated_parsed[:6])
                
                if published_date is None or published_date >= self.time_threshold:
                    entry_data = {
                        'source_url': feed_data['url'],
                        'source_title': feed.feed.get('title', 'Unknown'),
                        'title': entry.get('title', 'No title'),
                        'link': entry.get('link', ''),
                        'published': published_date.isoformat() if published_date else 'Unknown',
                        'summary': entry.get('summary', entry.get('description', ''))[:500],
                        'author': entry.get('author', entry.get("dc:creator", feed.feed.get('title', 'Unknown'))),
                        'full_text': None
                    }
                    recent_entries.append(entry_data)
            
            logger.info(f"Найдено {len(recent_entries)} записей за последние 24ч из {feed_data['url']}")
            return recent_entries
            
        except Exception as e:
            logger.error(f"Ошибка парсинга {feed_data['url']}: {str(e)}")
            return []
    
    def load_urls(self) -> List[str]:
        try:
            with open(self.urls_file, 'r', encoding='utf-8') as f:
                urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            logger.info(f"Загружено {len(urls)} URL-адресов")
            return urls
        except FileNotFoundError:
            logger.error(f"Файл {self.urls_file} не найден")
            return []
    
    async def parse_all_feeds(self):
        urls = self.load_urls()
        
        if not urls:
            logger.error("Нет URL для парсинга")
            return
        
        all_entries = []
        
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_rss(session, url) for url in urls]
            feed_results = await asyncio.gather(*tasks)

            for feed_data in feed_results:
                entries = self.parse_feed(feed_data)
                all_entries.extend(entries)
            
            if self.fetch_content and all_entries:
                logger.info(f"Начинаем извлечение текста из {len(all_entries)} статей...")

                semaphore = asyncio.Semaphore(20)
                
                async def fetch_with_semaphore(entry):
                    async with semaphore:
                        if entry['link']:
                            entry['full_text'] = await self.fetch_article_content(session, entry['link'])
                            await asyncio.sleep(0.5)
                        return entry
                
                tasks = [fetch_with_semaphore(entry) for entry in all_entries]
                all_entries = await asyncio.gather(*tasks)
        
        all_entries = list(filter(lambda x: x.get('full_text'), all_entries))
        self.save_results(all_entries)
        # logger.info(f"Всего найдено {len(all_entries)} записей за последние 24 часа")
    
    def save_results(self, entries: List[Dict]):
        output_data = {
            'generated_at': datetime.now().isoformat(),
            'total_entries': len(entries),
            'time_range': '24 hours',
            'entries': entries
        }
        
        Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Результаты сохранены в {self.output_file}")


async def main():
    input_dir = Path("input")
    input_dir.mkdir(exist_ok=True)
    urls_file = input_dir / 'rss_urls.txt'
    # urls_file = r'rss_parser\input\rss_urls.txt'
    if not Path(urls_file).exists():
        logger.info(f"Нет файла: {urls_file}")
        return 0

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'rss_feed_24h.json'
    # output_file = r'rss_parser\output\rss_feed_24h.json'
    parser = RSSParser(
        urls_file=urls_file, 
        output_file=output_file,
        fetch_content=True
    )
    await parser.parse_all_feeds()


if __name__ == "__main__":
    asyncio.run(main())