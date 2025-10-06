import os
import asyncio
from typing import List
from pydantic import BaseModel, Field
from openai import AsyncOpenAI


class ClusterAnalysis(BaseModel):
    reasoning_about_title: List[str] = Field(
        ..., 
        min_length=2, 
        max_length=3,
        description="Рассуждения о выборе заголовка. Учитываем, что заголовок должен читаться"
    )
    title: str = Field(
        ...,
        description="Красивый заголовок кластера, объединяющий все темы"
    )
    reasoning_about_summary: List[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="Рассуждения о создании саммари по всему класстеру"
    )
    summary: str = Field(
        ...,
        description="Краткое содержание кластера, которое показывает читателю, о чём класстер"
    )


class TextClusterAnalyzer:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = "gpt-4o"):

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        
        client_kwargs = {"api_key": self.api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
            
        self.client = AsyncOpenAI(**client_kwargs)
    
    async def analyze_cluster(self, texts: List[str]) -> ClusterAnalysis:
        if not texts:
            raise ValueError("Список текстов не может быть пустым")
    
        texts_formatted = "\n\n".join([f"Текст {i+1}:\n{text}" for i, text in enumerate(texts)])[:3500]
        
        prompt = f"""
Проанализируй следующий кластер текстов и создай для него заголовок и краткое содержание.

{texts_formatted}

Анализируй темы, общие идеи и ключевые моменты всех текстов."""
        
        completion = await self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format=ClusterAnalysis,
            temperature=0.7
        )
        # print(type(completion.choices[0].message.parsed))
        return completion.choices[0].message.parsed
    
    async def analyze_multiple_clusters(self, clusters: List[List[str]]) -> List[ClusterAnalysis]:
        tasks = [self.analyze_cluster(cluster) for cluster in clusters]
        return await asyncio.gather(*tasks)
    
    async def close(self):
        await self.client.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


if __name__ == "__main__":
    
    async def main():

        async with TextClusterAnalyzer(

            base_url="http://localhost:30000/v1",

            api_key="not-needed",

            model="Qwen/Qwen3-14B-AWQ" 
        ) as analyzer:
            
            sample_texts = [
                "Сегодня запустили новую функцию в приложении. Пользователи очень довольны.",
                "Обновление приложения получило положительные отзывы от бета-тестеров.",
                "Новая фича увеличила вовлеченность пользователей на 25%."
            ]
            
            try:
                print("--- Анализ одного кластера ---")
                result = await analyzer.analyze_cluster(sample_texts)
                
                print("Заголовок:", result.title)
                print("\nРассуждения о заголовке:")
                for r in result.reasoning_about_title:
                    print(f"- {r}")
                
                print("\nСаммари:", result.summary)
                print("\nРассуждения о саммари:")
                for r in result.reasoning_about_summary:
                    print(f"- {r}")
                
                clusters = [
                    [
                        "Продажи выросли на 15% в этом квартале.",
                        "Выручка достигла рекордных значений."
                    ],
                    [
                        "Команда разработки расширилась до 50 человек.",
                        "Наняли трех новых senior разработчиков."
                    ]
                ]
                
                results = await analyzer.analyze_multiple_clusters(clusters)
                print("\n\n=== Анализ нескольких кластеров ===")
                for i, res in enumerate(results, 1):
                    print(f"\nКластер {i}:")
                    print(f"Заголовок: {res.title}")
                    print(f"Саммари: {res.summary}")

            except Exception as e:
                print(f"\nПроизошла ошибка: {e}")
                print("Убедитесь, что ваш локальный сервер запущен, доступен по указанному адресу и поддерживает structured output (tool_calls).")

    
    asyncio.run(main())