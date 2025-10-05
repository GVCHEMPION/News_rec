import asyncio
from typing import Optional, Type
import aiohttp
from urllib.parse import urlparse
from random import uniform

from parsers.pars_goha_ru import GohaRuParser
from parsers.pars_hi_news import HiNewsParser
from parsers.pars_ixbt import IxbtParser
from parsers.pars_kommersant import KommersantParser
from parsers.pars_lenta_ru import LentaRuParser
from parsers.pars_mail_ru import MailRuParser
from parsers.pars_rambler import RamblerParser
from parsers.pars_ria_ru import RiaRuParser
from parsers.pars_tass import TassParser
from parsers.pars_vedomosti import VedomostiParser
from parsers.pars_vz_ru import VzRuParser
from parsers.abstract_pars import AbstractNewsParser


class NewsParserManager:
    PARSERS = {
        'vz.ru': VzRuParser,
        'goha.ru': GohaRuParser,
        'vedomosti.ru': VedomostiParser,
        'tass.ru': TassParser,
        'ria.ru': RiaRuParser,
        'rambler.ru': RamblerParser,
        'mail.ru': MailRuParser,
        'lenta.ru': LentaRuParser,
        'kommersant.ru': KommersantParser,
        'ixbt.com': IxbtParser,
        'hi-news.ru': HiNewsParser,
    }

    def __init__(self):
        pass

    @staticmethod
    def _extract_domain(link: str) -> str:
        try:
            netloc = urlparse(link).netloc
            if netloc.startswith('www.'):
                return netloc[4:]
            return netloc
        except:
            return ""

    def _get_parser_class(self, url: str) -> Optional[Type[AbstractNewsParser]]:
        domain = self._extract_domain(url)
        if not domain:
            return None

        for supported_domain, parser_class in self.PARSERS.items():
            if domain == supported_domain or domain.endswith('.' + supported_domain):
                return parser_class
        
        return None

    async def extract_text(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        parser_class = self._get_parser_class(url)

        if not parser_class:
            return None

        try:
            parser_instance = parser_class(url)
            return await parser_instance.parse(session)
        except Exception as e:
            
            return None

    @classmethod
    def is_supported(cls, link: str) -> bool:
        domain = cls._extract_domain(link)
        return any(domain == s_domain or domain.endswith('.' + s_domain) for s_domain in cls.PARSERS.keys())

    @classmethod
    def get_supported_domains(cls) -> list[str]:
        return list(cls.PARSERS.keys())



async def main():
    links = [
        "https://www.vedomosti.ru/society/news/2025/10/04/1144323-vlasti-belgoroda",
        "https://tass.ru/politika/25247237",
        "https://lenta.ru/news/2025/10/04/13-letniy-pyanyy-malchik-ugnal-avtomobil-i-popal-v-avariyu/",
        "https://unsupported-domain.com/some-article"
    ]
    
    print("Поддерживаемые домены:")
    print(", ".join(NewsParserManager.get_supported_domains()))
    print()

    manager = NewsParserManager()
    async with aiohttp.ClientSession() as session:
        for link in links:
            print(f"\n{'='*80}")
            print(f"Парсинг: {link}")
            print(f"{'='*80}")
            
            article_text = await manager.extract_text(session, link)
            
            if article_text:
                print("\n--- Текст статьи ---")
                preview = article_text[:500] + "..." if len(article_text) > 500 else article_text
                print(preview)
                print(f"\nОбщая длина текста: {len(article_text)} символов")
                print("--------------------\n")
            else:
                print("Не удалось извлечь текст статьи (возможно, домен не поддерживается или произошла ошибка).")

if __name__ == "__main__":
    pass