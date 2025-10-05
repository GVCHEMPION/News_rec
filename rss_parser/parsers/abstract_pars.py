from abc import ABC, abstractmethod
from fake_useragent import UserAgent
from random import uniform

class AbstractNewsParser(ABC):
    def __init__(self, link: str):
        if not link.startswith(('http://', 'https://')):
            raise ValueError("Некорректная ссылка. Она должна начинаться с http:// или https://")
        
        self.link = link
        self.domain = link.split('//')[-1].split('/')[0]
        ua = UserAgent()
        self.headers = {
            'User-Agent': ua.ff
        }
    
    @abstractmethod
    async def parse(self) -> str | None:
        """Абстрактный метод: парсинг статьи"""
        pass

