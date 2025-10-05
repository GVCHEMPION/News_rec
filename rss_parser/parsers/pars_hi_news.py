import asyncio
from random import uniform 

import aiohttp
from bs4 import BeautifulSoup

from .abstract_pars import AbstractNewsParser


class HiNewsParser(AbstractNewsParser):
    def __init__(self, link: str):
        super().__init__(link)
        if 'hi-news.ru' not in self.domain:
            raise ValueError(f"Ссылка '{link}' не принадлежит домену hi-news.ru")

    @staticmethod
    def _parse_html_sync(html_content: str) -> str|None:
        soup = BeautifulSoup(html_content, 'lxml')
        
        text_div = soup.select_one('article[id="main"] div[class="text"]')
        if not text_div:
            return None

        texts = []
        for element in text_div.find_all(['h2', 'p']):
            text = element.get_text(strip=True)
            if text:
                if element.name == 'h2':
                    texts.append(f"\n{text}:")
                else:
                    texts.append(text)
        
        if not texts:
            return None
        
        return " ".join(texts)

    async def parse(self, session: aiohttp.ClientSession) -> str | None:
        MAX_RETRIES = 5
        RETRY_DELAY = 1
        REQUEST_TIMEOUT = 20
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with session.get(self.link, headers=self.headers, timeout=REQUEST_TIMEOUT) as response:
                    response.raise_for_status()
                    html_content = await response.text()
                    loop = asyncio.get_running_loop()
                    parsed_text = await loop.run_in_executor(
                        None, self._parse_html_sync, html_content
                    )
                    return parsed_text

            except aiohttp.ClientError as e:
                if attempt < MAX_RETRIES:
                    delay = RETRY_DELAY * (2 ** (attempt))
                    jitter = uniform(0, 0.5)
                    await asyncio.sleep(delay + jitter)
                    continue
                else:
                    return None
            except Exception as e:
                if attempt < MAX_RETRIES:
                    delay = RETRY_DELAY * (2 ** (attempt))
                    jitter = uniform(0, 0.5)
                    await asyncio.sleep(delay + jitter)
                    continue
                else:
                    return None
        return None


async def main():
    hi_news_link = "https://hi-news.ru/eto-interesno/pochemu-osenyu-i-zimoj-lyudi-stavyat-na-podokonnik-tarelku-s-solyu.html"
    
    try:
        parser = HiNewsParser(hi_news_link)
        article_text = await parser.parse()

        if article_text:
            print("\n--- Текст статьи ---")
            print(article_text)
            print("--------------------\n")
        else:
            print("Не удалось извлечь текст статьи.")

    except ValueError as e:
        print(f"Ошибка инициализации парсера: {e}")


if __name__ == "__main__":
    asyncio.run(main())