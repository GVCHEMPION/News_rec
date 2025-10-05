import asyncio
import schedule
import time
from datetime import datetime
import logging
from rss_parser import RSSParser
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RSSScheduler:
    def __init__(self, urls_file: str, output_file: str, run_on_start: bool = True):
        self.urls_file = urls_file
        self.output_file = output_file
        self.run_on_start = run_on_start
        
    async def run_parser(self):
        try:
            logger.info("=" * 60)
            logger.info(f"Запуск парсера в {datetime.now()}")
            logger.info("=" * 60)
            
            parser = RSSParser(
                urls_file=self.urls_file,
                output_file=self.output_file,
                fetch_content=True
            )
            
            await parser.parse_all_feeds()
            
            logger.info("=" * 60)
            logger.info(f"Парсинг завершен в {datetime.now()}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Ошибка при выполнении парсинга: {str(e)}", exc_info=True)
    
    def schedule_job(self):
        asyncio.run(self.run_parser())
    
    def start(self):
        logger.info("Запуск RSS планировщика")
        logger.info(f"Парсер будет запускаться каждые 24 часа")
        
        if not Path(self.urls_file).exists():
            logger.error(f"Файл {self.urls_file} не найден!")
            return
        
        if self.run_on_start:
            logger.info("Выполнение первого запуска...")
            self.schedule_job()
        
        schedule.every(24).hours.do(self.schedule_job)
        
        # Альтернативный вариант - запуск в определенное время каждый день
        # schedule.every().day.at("00:00").do(self.schedule_job)
        
        logger.info("Планировщик активен. Ожидание следующего запуска...")
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Планировщик остановлен пользователем")


def main():
    input_dir = Path("input")
    input_dir.mkdir(exist_ok=True)
    urls_file = input_dir / 'rss_urls.txt'
    if not Path(urls_file).exists():
        logger.info(f"Нет файла: {urls_file}")
        return 0

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'rss_feed_24h.json'
    scheduler = RSSScheduler(
        urls_file=urls_file,
        output_file=output_file,
        run_on_start=True
    )
    scheduler.start()


if __name__ == "__main__":
    main()