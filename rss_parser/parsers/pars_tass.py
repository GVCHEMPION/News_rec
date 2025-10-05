import aiohttp
from random import uniform
import asyncio
from bs4 import BeautifulSoup

from .abstract_pars import AbstractNewsParser


class TassParser(AbstractNewsParser):
    def __init__(self, link: str):
        super().__init__(link)

    def _parse_html_sync(self, html: str) -> str|None:
        soup = BeautifulSoup(html, 'lxml')
        article_body = soup.find('article')
        if not article_body:
            return None
        texts = []
        summary = article_body.find('summary')
        if summary:
            texts.append(summary.get_text(strip=True))

        paragraphs = article_body.find_all('p')
        for p in paragraphs:
            texts.append(p.get_text(strip=True))
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
    tass_url = 'https://tass.ru/politika/25247237'
    parser = TassParser(link=tass_url)

    article_text = await parser.parse()

    print("\n--- Результат парсинга ---")
    print(article_text)
    print("--------------------------")


if __name__ == "__main__":
    asyncio.run(main())