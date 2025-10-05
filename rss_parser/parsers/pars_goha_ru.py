import asyncio
from random import uniform

import aiohttp
from bs4 import BeautifulSoup

from .abstract_pars import AbstractNewsParser


class GohaRuParser(AbstractNewsParser):
    def __init__(self, link: str):
        super().__init__(link)
        if 'goha.ru' not in self.domain:
            raise ValueError(f"Ссылка '{link}' не принадлежит домену goha.ru")

    @staticmethod
    def _parse_html_sync(html_content: str) -> str|None:
        soup = BeautifulSoup(html_content, 'lxml')
        
        text_div = soup.find('div', class_="editor-body entry-article__article-body")
        if not text_div:            
            return None

        texts = []

        paragraphs = text_div.find_all('p')
        for p in paragraphs:
            texts.append(p.get_text(strip=True))
        
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
    goha_ru_link = "https://www.goha.ru/najden-redchajshij-processor-intel-pentium-4-s-chastotoj-40-ggc-dYzo6n"
    
    try:
        async with aiohttp.ClientSession() as session:
            parser = GohaRuParser(goha_ru_link)
            article_text = await parser.parse(session)

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