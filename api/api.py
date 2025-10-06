from fastapi import FastAPI, HTTPException, Query, Body, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, HttpUrl, validator
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
import uuid
import logging
from collections import defaultdict
import hashlib
import secrets
import jwt
import os
import asyncio
import re
from pathlib import Path

from sqlalchemy import create_engine, Column, String, Integer, Text, DateTime, ForeignKey, Table, ARRAY, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB

from llm import TextClusterAnalyzer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== DATABASE SETUP ====================
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://news_user:secure_password_123@postgres:5432/news_db")
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
RSS_JSON_PATH = Path(os.getenv("RSS_JSON_PATH", "/app/rss_parser/output/rss_feed_24h.json"))

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ==================== DATABASE MODELS ====================
article_relations = Table(
    'article_relations',
    Base.metadata,
    Column('article_id', UUID(as_uuid=True), ForeignKey('articles.id'), primary_key=True),
    Column('related_article_id', UUID(as_uuid=True), ForeignKey('articles.id'), primary_key=True)
)

user_read_articles = Table(
    'user_read_articles',
    Base.metadata,
    Column('user_id', UUID(as_uuid=True), ForeignKey('users.id'), primary_key=True),
    Column('article_id', UUID(as_uuid=True), ForeignKey('articles.id'), primary_key=True),
    Column('read_at', DateTime, default=datetime.utcnow)
)

class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    login = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(256), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    interested_hashtags = Column(JSONB, default={}) 
    interested_clusters = Column(JSONB, default={})
    interested_authors = Column(JSONB, default={})
    
    read_articles = relationship("Article", secondary=user_read_articles, back_populates="read_by_users")

class Cluster(Base):
    __tablename__ = "clusters"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    cluster_number = Column(Integer, nullable=False)
    name = Column(String(500), nullable=False)
    summary = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    articles = relationship("Article", back_populates="cluster")

class Article(Base):
    __tablename__ = "articles"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    link = Column(String(1000), unique=True, nullable=False, index=True)
    title = Column(String(1000), nullable=False)
    summary = Column(Text)
    short_summary = Column(Text)
    source = Column(String(200))
    published = Column(String(100))
    author = Column(String(200), nullable=True)

    hashtags = Column(ARRAY(String), default=[])
    entities = Column(JSONB, default={})

    cluster_id = Column(UUID(as_uuid=True), ForeignKey('clusters.id'), nullable=True)
    cluster = relationship("Cluster", back_populates="articles")

    related_articles = relationship(
        "Article",
        secondary=article_relations,
        primaryjoin=id == article_relations.c.article_id,
        secondaryjoin=id == article_relations.c.related_article_id,
        backref="related_to"
    )
    
    read_by_users = relationship("User", secondary=user_read_articles, back_populates="read_articles")
    
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# ==================== PYDANTIC MODELS ====================

class UserCreate(BaseModel):
    login: str = Field(..., min_length=3, max_length=100)
    password: str = Field(..., min_length=6, max_length=100)

class UserLogin(BaseModel):
    login: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class ArticleCreate(BaseModel):
    title: str = Field(..., description="Заголовок статьи")
    summary: Optional[str] = Field(None, description="Краткое содержание")
    link: str = Field(..., description="Ссылка на статью")
    source: Optional[str] = Field("Unknown", description="Источник")
    published: Optional[str] = Field(None, description="Дата публикации")
    author: Optional[str] = Field(None, description="Автор статьи")

class ProcessRequest(BaseModel):
    articles: List[ArticleCreate] = Field(..., description="Список статей для обработки")
    eps: float = Field(0.4, ge=0.1, le=1.0, description="Параметр кластеризации DBSCAN")
    min_samples: int = Field(1, ge=1, le=10, description="Минимум статей в кластере")

class RSSProcessRequest(BaseModel):
    eps: float = Field(0.4, ge=0.1, le=1.0, description="Параметр кластеризации DBSCAN")
    min_samples: int = Field(1, ge=1, le=10, description="Минимум статей в кластере")
    json_path: Optional[str] = Field(None, description="Путь к JSON файлу (по умолчанию: /app/rss_parser/output/rss_feed_24h.json)")

class ArticleResponse(BaseModel):
    id: uuid.UUID
    title: str
    summary: Optional[str]
    short_summary: Optional[str]
    link: str
    source: Optional[str]
    published: Optional[str]
    author: Optional[str]
    hashtags: List[str]
    entities: Dict[str, List[str]]
    cluster_id: Optional[uuid.UUID]
    related_articles: List[Dict[str, str]]
    
    class Config:
        from_attributes = True

class ClusterResponse(BaseModel):
    id: uuid.UUID
    cluster_number: int
    name: str
    summary: str
    articles_count: int
    
    class Config:
        from_attributes = True

class RecommendationResponse(BaseModel):
    article: ArticleResponse
    score: float
    reason: str

# ==================== UTILITY FUNCTIONS ====================

def normalize_hashtag(tag: str) -> Optional[str]:
    tag = tag.lstrip('#')
    
    # Оставляем только буквы, цифры и _
    clean_tag = re.sub(r'[^\w]', '', tag, flags=re.UNICODE)
    
    if not clean_tag:
        return None
    
    return f"#{clean_tag.lower()}"


def normalize_hashtags(hashtags: List[str]) -> List[str]:
    normalized = []
    for tag in hashtags:
        normalized_tag = normalize_hashtag(tag)
        if normalized_tag and normalized_tag not in normalized:
            normalized.append(normalized_tag)
    return normalized


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, hashed: str) -> bool:
    return hash_password(password) == hashed


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.PyJWTError:
        return None


def load_rss_json(file_path: str = None) -> Dict[str, Any]:
    if file_path is None:
        file_path = RSS_JSON_PATH
    else:
        file_path = Path(file_path)
    
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"RSS JSON файл не найден: {file_path}"
        )
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Ошибка парсинга JSON: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка чтения файла: {str(e)}"
        )


def convert_rss_entries_to_articles(entries: List[Dict]) -> List[Dict]:
    articles = []
    
    for entry in entries:
        article = {
            "title": entry.get("title", "Без названия"),
            "summary": entry.get("full_text") or entry.get("summary"),
            "link": entry.get("link", ""),
            "source": entry.get("source_title", "Unknown"),
            "published": entry.get("published", ""),
            "author": entry.get("author", "Unknown")
        }
        
        if article["link"]:
            articles.append(article)
    
    return articles

# ==================== FASTAPI APP ====================

app = FastAPI(
    title="News Processing & Recommendation API",
    description="API для кластеризации новостей с PostgreSQL и автоматической обработкой RSS",
    version="2.0.0" 
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

cluster_analyzer = None 

async def generate_cluster_summary(articles_summaries: List[str]) -> Dict[str, Any]:
    if not cluster_analyzer:
        logger.error("Cluster analyzer не был инициализирован.")
        raise HTTPException(status_code=503, detail="Сервис анализа кластеров недоступен.")
    
    try:
        analysis = await cluster_analyzer.analyze_cluster(articles_summaries)
        logger.info(f"LLM сгенерировал заголовок: '{analysis.title}'")
        return {
            "title": analysis.title,
            "summary": analysis.summary
        }
    except Exception as e:
        logger.error(f"Ошибка при генерации саммари кластера через LLM: {e}")
        return {
            "title": f"Кластер из {len(articles_summaries)} новостей",
            "summary": "Автоматическое резюме временно недоступно. Основные темы можно понять по заголовкам статей."
        }

async def process_rss_on_startup():
    logger.info("Запуск фоновой обработки RSS при старте...")
    db = SessionLocal()
    try:
        # return #Выключение загрузки
        rss_data = load_rss_json()
        entries = rss_data.get("entries", [])
        if not entries:
            logger.info("В RSS файле нет статей для обработки.")
            return

        articles_dict = convert_rss_entries_to_articles(entries)
        
        try:
            from service_code import UnifiedNewsService
        except ImportError:
            logger.error("Модуль 'service_code' не найден. Обработка на старте отменена.")
            return

        service = UnifiedNewsService()
        result = service.process_articles(articles_dict, eps=0.4, min_samples=1)
        
        cluster_summaries_tasks = []
        cluster_data_list = []
        for cluster_data in result['clusters']:
            articles_summaries = [a.get('summary', a['title']) for a in cluster_data['articles']]
            cluster_data_list.append(cluster_data)
            cluster_summaries_tasks.append(generate_cluster_summary(articles_summaries))
        
        cluster_summaries = await asyncio.gather(*cluster_summaries_tasks)

        created_clusters = {}
        created_articles = {}

        for idx, cluster_data in enumerate(cluster_data_list):
            llm_summary = cluster_summaries[idx]
            
            cluster = Cluster(
                cluster_number=int(cluster_data['cluster_id']),
                name=llm_summary['title'],
                summary=llm_summary['summary']
            )
            db.add(cluster)
            db.flush()
            created_clusters[cluster_data['cluster_id']] = cluster

        # Логика сохранения статей (остается прежней)
        for cluster_data in result['clusters']:
            cluster = created_clusters[cluster_data['cluster_id']]
            for article_data in cluster_data['articles']:
                normalized_tags = normalize_hashtags(article_data.get('hashtags', []))
                existing_article = db.query(Article).filter(Article.link == article_data['link']).first()
                if existing_article:
                    article = existing_article
                    # Обновляем поля, если статья уже есть
                    article.title = article_data['title']
                    article.summary = article_data.get('summary')
                    article.short_summary = article_data.get('short_summary')
                    article.hashtags = normalized_tags
                    article.entities = article_data.get('entities', {})
                    article.author = article_data.get('author')
                    article.cluster_id = cluster.id
                else:
                    article = Article(
                        link=article_data['link'],
                        title=article_data['title'],
                        summary=article_data.get('summary'),
                        short_summary=article_data.get('short_summary'),
                        source=article_data.get('source'),
                        published=article_data.get('published'),
                        author=article_data.get('author'),
                        hashtags=normalized_tags,
                        entities=article_data.get('entities', {}),
                        cluster_id=cluster.id
                    )
                    db.add(article)
                db.flush()
                created_articles[article_data['link']] = article
        
        # Логика связывания статей (остается прежней)
        for cluster_data in result['clusters']:
            for article_data in cluster_data['articles']:
                article = created_articles[article_data['link']]
                related = article_data.get('related_articles_in_cluster', [])
                if related:
                    for rel_data in related:
                        related_article = created_articles.get(rel_data['link'])
                        if related_article and related_article not in article.related_articles:
                            article.related_articles.append(related_article)
        
        db.commit()
        logger.info(f"Обработка RSS на старте успешно завершена. Создано кластеров: {len(created_clusters)}, статей: {len(created_articles)}")

    except HTTPException as e:
        logger.warning(f"Ошибка при обработке RSS на старте: {e.detail}")
        db.rollback()
    except Exception as e:
        logger.critical(f"Критическая ошибка при обработке RSS на старте: {e}", exc_info=True)
        db.rollback()
    finally:
        db.close()

@app.on_event("startup")
async def startup_event():
    global cluster_analyzer
    logger.info("Запуск приложения...")
    cluster_analyzer = TextClusterAnalyzer(
        api_key=os.getenv("OPENAI_API_KEY", "not-needed"),
        base_url=os.getenv("SGLANG_URL", "http://sglang:30000/v1"),
        model=os.getenv("OPENAI_MODEL", "Qwen/Qwen3-14B-AWQ")
    )
    asyncio.create_task(process_rss_on_startup())
    logger.info("Приложение запущено. Обработка RSS выполняется в фоновом режиме.")

@app.on_event("shutdown")
async def shutdown_event():
    global cluster_analyzer
    if cluster_analyzer:
        await cluster_analyzer.close()
    logger.info("Приложение остановлено.")

# ==================== DEPENDENCY ====================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)) -> User:
    token = credentials.credentials
    payload = decode_token(token)
    if not payload or not (user_id := payload.get("sub")):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")
    
    user = db.query(User).filter(User.id == uuid.UUID(user_id)).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    
    return user

# ==================== AUTH ENDPOINTS ====================

@app.post("/api/v1/auth/register", response_model=Token, tags=["Authentication"])
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.login == user_data.login).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Login already registered"
        )
    
    user = User(
        login=user_data.login,
        password_hash=hash_password(user_data.password)
    )
    
    db.add(user)
    db.commit()
    db.refresh(user)
    
    access_token = create_access_token(
        data={"sub": str(user.id)},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/api/v1/auth/login", response_model=Token, tags=["Authentication"])
async def login(user_data: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.login == user_data.login).first()
    
    if not user or not verify_password(user_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect login or password"
        )
    
    access_token = create_access_token(
        data={"sub": str(user.id)},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

# ==================== ARTICLES ENDPOINTS ====================
async def _process_and_save_articles(articles_dict: List[Dict], eps: float, min_samples: int, db: Session):
    try:
        from service_code import UnifiedNewsService
    except ImportError:
         raise HTTPException(status_code=500, detail="Модуль 'service_code' не найден.")

    service = UnifiedNewsService()
    result = service.process_articles(articles_dict, eps=eps, min_samples=min_samples)

    cluster_summaries_tasks = []
    cluster_data_list = []
    for cluster_data in result['clusters']:
        articles_summaries = [a.get('summary', a['title']) for a in cluster_data['articles']]
        cluster_data_list.append(cluster_data)
        cluster_summaries_tasks.append(generate_cluster_summary(articles_summaries))

    cluster_summaries = await asyncio.gather(*cluster_summaries_tasks, return_exceptions=True)

    created_clusters = {}
    created_articles = {}
    
    for idx, cluster_data in enumerate(cluster_data_list):
        llm_summary = cluster_summaries[idx]
        
        if isinstance(llm_summary, Exception):
            logger.error(f"Задача LLM для кластера {cluster_data['cluster_id']} провалилась: {llm_summary}")
            cluster_name = cluster_data.get('cluster_name', f"Кластер #{cluster_data['cluster_id']}")
            cluster_summary_text = cluster_data.get('cluster_summary', 'Автоматическое резюме недоступно.')
        else:
            cluster_name = llm_summary['title']
            cluster_summary_text = llm_summary['summary']
        
        cluster = Cluster(
            cluster_number=int(cluster_data['cluster_id']),
            name=cluster_name,
            summary=cluster_summary_text
        )
        db.add(cluster)
        db.flush()
        created_clusters[cluster_data['cluster_id']] = cluster

    for cluster_data in result['clusters']:
        cluster = created_clusters[cluster_data['cluster_id']]
        for article_data in cluster_data['articles']:
            normalized_tags = normalize_hashtags(article_data.get('hashtags', []))
            existing_article = db.query(Article).filter(Article.link == article_data['link']).first()
            if existing_article:
                article = existing_article
                article.title, article.summary, article.short_summary, article.hashtags, article.entities, article.author, article.cluster_id = \
                    article_data['title'], article_data.get('summary'), article_data.get('short_summary'), normalized_tags, article_data.get('entities', {}), article_data.get('author'), cluster.id
            else:
                article = Article(
                    link=article_data['link'], title=article_data['title'], summary=article_data.get('summary'),
                    short_summary=article_data.get('short_summary'), source=article_data.get('source'),
                    published=article_data.get('published'), author=article_data.get('author'),
                    hashtags=normalized_tags, entities=article_data.get('entities', {}), cluster_id=cluster.id
                )
                db.add(article)
            db.flush()
            created_articles[article_data['link']] = article

    for cluster_data in result['clusters']:
        for article_data in cluster_data['articles']:
            article = created_articles[article_data['link']]
            related = article_data.get('related_articles_in_cluster', [])
            if related:
                for rel_data in related:
                    related_article = created_articles.get(rel_data['link'])
                    if related_article and related_article not in article.related_articles:
                        article.related_articles.append(related_article)
    
    db.commit()
    
    return {
        "processed_articles": result['total_articles'],
        "total_clusters": result['total_clusters'],
        "clusters_created": len(created_clusters),
        "articles_created_or_updated": len(created_articles),
        "llm_summaries_generated": sum(1 for s in cluster_summaries if not isinstance(s, Exception))
    }

@app.post("/api/v1/articles/process-rss", tags=["Articles"])
async def process_rss_articles(request: RSSProcessRequest = Body(...), current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        rss_data = load_rss_json(request.json_path)
        entries = rss_data.get("entries", [])
        if not entries:
            raise HTTPException(status_code=400, detail="В RSS JSON файле нет статей")
        
        articles_dict = convert_rss_entries_to_articles(entries)
        if not articles_dict:
            raise HTTPException(status_code=400, detail="Не удалось извлечь статьи из JSON")

        stats = await _process_and_save_articles(articles_dict, request.eps, request.min_samples, db)

        return {
            "status": "success",
            "message": "RSS статьи успешно обработаны с LLM-суммаризацией",
            "statistics": {
                "total_entries_in_file": rss_data.get("total_entries", 0),
                "time_range_in_file": rss_data.get("time_range", "unknown"),
                **stats
            }
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Ошибка при обработке RSS: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")


@app.post("/api/v1/articles/process", tags=["Articles"])
async def process_articles(request: ProcessRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        articles_dict = [article.dict() for article in request.articles]
        stats = await _process_and_save_articles(articles_dict, request.eps, request.min_samples, db)
        
        return {
            "status": "success",
            "message": "Статьи успешно обработаны с LLM-суммаризацией",
            "statistics": stats
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Ошибка при обработке статей: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")

@app.get("/api/v1/articles", response_model=List[ArticleResponse], tags=["Articles"])
async def get_articles(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    cluster_id: Optional[uuid.UUID] = Query(None, description="Фильтр по ID кластера"),
    hashtags: Optional[List[str]] = Query(None, description="Список хэштегов для фильтрации (логика И). Пример: ?hashtags=политика&hashtags=выборы"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    query = db.query(Article)
    
    if cluster_id:
        query = query.filter(Article.cluster_id == cluster_id)

    if hashtags:
        normalized_tags = normalize_hashtags(hashtags)
        if normalized_tags:
            query = query.filter(Article.hashtags.contains(normalized_tags))
        
    articles = query.offset(skip).limit(limit).all()
    
    result = []
    for article in articles:
        related = [{"title": a.title, "link": a.link} for a in article.related_articles]
        result.append({
            "id": article.id,
            "title": article.title,
            "summary": article.summary,
            "short_summary": article.short_summary,
            "link": article.link,
            "source": article.source,
            "published": article.published,
            "author": article.author,
            "hashtags": article.hashtags or [],
            "entities": article.entities or {},
            "cluster_id": article.cluster_id,
            "related_articles": related
        })
    print(f"articles: {len(result)}")
    return result


@app.get("/api/v1/articles/{article_id}", response_model=ArticleResponse, tags=["Articles"])
async def get_article(
    article_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    article = db.query(Article).filter(Article.id == article_id).first()
    
    if not article:
        raise HTTPException(status_code=404, detail="Статья не найдена")
    
    related = [{"title": a.title, "link": a.link} for a in article.related_articles]
    
    return {
        "id": article.id,
        "title": article.title,
        "summary": article.summary,
        "short_summary": article.short_summary,
        "link": article.link,
        "source": article.source,
        "published": article.published,
        "author": article.author,
        "hashtags": article.hashtags or [],
        "entities": article.entities or {},
        "cluster_id": article.cluster_id,
        "related_articles": related
    }


@app.post("/api/v1/articles/{article_id}/read", tags=["User Actions"])
async def mark_article_read(
    article_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    article = db.query(Article).filter(Article.id == article_id).first()
    
    if not article:
        raise HTTPException(status_code=404, detail="Статья не найдена")
    
    if article in current_user.read_articles:
        return {"status": "info", "message": "Статья уже отмечена как прочитанная"}
    
    current_user.read_articles.append(article)
    
    topics_dict = current_user.interested_hashtags or {}
    
    for tag in article.hashtags or []:
        topics_dict[tag] = topics_dict.get(tag, 0) + 1
    
    for entity_list in (article.entities or {}).values():
        for entity in entity_list:
            normalized_entity = normalize_hashtag(entity)
            if normalized_entity:
                topics_dict[normalized_entity] = topics_dict.get(normalized_entity, 0) + 1
            
    current_user.interested_hashtags = topics_dict
    
    if article.author:
        authors_dict = current_user.interested_authors or {}
        authors_dict[article.author] = authors_dict.get(article.author, 0) + 1
        current_user.interested_authors = authors_dict
    
    if article.cluster_id:
        clusters_dict = current_user.interested_clusters or {}
        cluster_key = str(article.cluster_id)
        clusters_dict[cluster_key] = clusters_dict.get(cluster_key, 0) + 1
        current_user.interested_clusters = clusters_dict
    
    db.commit()
    
    return {
        "status": "success",
        "message": "Статья отмечена как прочитанная",
        "article_title": article.title
    }

# ==================== CLUSTERS ENDPOINTS ====================

@app.get("/api/v1/clusters", response_model=List[ClusterResponse], tags=["Clusters"])
async def get_clusters(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    clusters = db.query(Cluster).offset(skip).limit(limit).all()
    
    result = []
    for cluster in clusters:
        result.append({
            "id": cluster.id,
            "cluster_number": cluster.cluster_number,
            "name": cluster.name,
            "summary": cluster.summary,
            "articles_count": len(cluster.articles)
        })
    print(f"clusters: {len(result)}")
    return result


@app.get("/api/v1/clusters/{cluster_id}", tags=["Clusters"])
async def get_cluster(
    cluster_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    cluster = db.query(Cluster).filter(Cluster.id == cluster_id).first()
    
    if not cluster:
        raise HTTPException(status_code=404, detail="Кластер не найден")
    
    articles = []
    for article in cluster.articles:
        related = [{"title": a.title, "link": a.link} for a in article.related_articles]
        articles.append({
            "id": article.id,
            "title": article.title,
            "summary": article.summary,
            "short_summary": article.short_summary,
            "link": article.link,
            "author": article.author,
            "hashtags": article.hashtags or [],
            "related_articles": related
        })
    
    return {
        "id": cluster.id,
        "cluster_number": cluster.cluster_number,
        "name": cluster.name,
        "summary": cluster.summary,
        "articles_count": len(articles),
        "articles": articles
    }

# ==================== HASHTAGS ENDPOINTS ====================

@app.get("/api/v1/hashtags", response_model=List[str], tags=["Hashtags"])
async def get_hashtags(
    search: Optional[str] = Query(None, description="Поиск по части хэштега (без учета регистра)"),
    limit: int = Query(100, ge=1, le=500, description="Максимальное количество хэштегов для возврата"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Возвращает список уникальных хэштегов из всех статей.
    Можно использовать для автодополнения на фронтенде.
    """
    query = db.query(func.unnest(Article.hashtags).label("hashtag")).distinct()
    
    if search:
        normalized_search = normalize_hashtag(search)
        if normalized_search:
            query = query.filter(func.unnest(Article.hashtags).ilike(f"%{normalized_search}%"))

    results = query.limit(limit).all()
    
    return [row.hashtag for row in results]


# ==================== RECOMMENDATIONS ====================

@app.get("/api/v1/recommendations", response_model=Dict[str, Any], tags=["Recommendations"])
async def get_recommendations(
    top_n: int = Query(10, ge=1, le=50),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    read_ids = [a.id for a in current_user.read_articles]
    articles = db.query(Article).filter(~Article.id.in_(read_ids)).all()
    
    if not articles:
        return {
            "recommendations": [],
            "message": "Нет новых статей для рекомендаций"
        }
    
    scored_articles = []
    
    for article in articles:
        score = 0.0
        reasons = []
        
        user_topics = current_user.interested_hashtags or {}
        topic_score = 0.0
        common_topics = []
        
        article_topics = set(article.hashtags or [])
        for entity_list in (article.entities or {}).values():
            for entity in entity_list:
                normalized_entity = normalize_hashtag(entity)
                if normalized_entity:
                    article_topics.add(normalized_entity)

        for topic in article_topics:
            if topic in user_topics:
                topic_score += user_topics[topic]
                common_topics.append(topic)
        
        if common_topics:
            reasons.append(f"Темы: {', '.join(common_topics[:2])}")
        
        score += 0.55 * topic_score 
        
        user_authors = current_user.interested_authors or {}
        if article.author and article.author in user_authors:
            author_score = user_authors[article.author]
            score += 0.45 * author_score
            reasons.append(f"Автор: {article.author}")
        
        if article.cluster_id:
            user_clusters = current_user.interested_clusters or {}
            cluster_key = str(article.cluster_id)
            if cluster_key in user_clusters:
                score += 0.15 * user_clusters[cluster_key]
                reasons.append("Похожие новости")
        
        if not reasons:
            reasons.append("Новая тема для вас")
        
        scored_articles.append({
            "article": article,
            "score": score,
            "reason": " | ".join(reasons)
        })
    
    scored_articles.sort(key=lambda x: x['score'], reverse=True)
    
    recommendations = []
    for item in scored_articles[:top_n]:
        article = item['article']
        related = [{"title": a.title, "link": a.link} for a in article.related_articles]
        
        recommendations.append({
            "article": {
                "id": article.id,
                "title": article.title,
                "summary": article.summary,
                "short_summary": article.short_summary,
                "link": article.link,
                "source": article.source,
                "published": article.published,
                "author": article.author,
                "hashtags": article.hashtags or [],
                "entities": article.entities or {},
                "cluster_id": article.cluster_id,
                "related_articles": related
            },
            "score": round(item['score'], 2),
            "reason": item['reason']
        })
    print(f"recommendations: {len(recommendations)}")
    return {"recommendations": recommendations}

# ==================== USER PROFILE ====================

@app.get("/api/v1/profile", tags=["User Profile"])
async def get_profile(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    hashtags = current_user.interested_hashtags or {}
    authors = current_user.interested_authors or {}
    clusters = current_user.interested_clusters or {}
    
    top_hashtags = dict(sorted(hashtags.items(), key=lambda x: x[1], reverse=True)[:10])
    top_authors = dict(sorted(authors.items(), key=lambda x: x[1], reverse=True)[:10])
    
    return {
        "user_id": str(current_user.id),
        "login": current_user.login,
        "statistics": {
            "read_articles_count": len(current_user.read_articles),
            "unique_topics_count": len(hashtags),
            "unique_authors": len(authors),
            "preferred_clusters": len(clusters)
        },
        "interests": {
            "top_topics": top_hashtags,
            "top_authors": top_authors
        },
        "recent_articles": [
            {"id": str(a.id), "title": a.title, "link": a.link}
            for a in current_user.read_articles[-20:]
        ]
    }

# ==================== RSS INFO ====================

@app.get("/api/v1/rss/info", tags=["RSS"])
async def get_rss_info(
    json_path: Optional[str] = Query(None, description="Путь к JSON файлу"),
    current_user: User = Depends(get_current_user)
):
    try:
        rss_data = load_rss_json(json_path)
        entries = rss_data.get("entries", [])
        
        sources = {}
        for entry in entries:
            source = entry.get("source_title", "Unknown")
            sources[source] = sources.get(source, 0) + 1
        
        return {
            "file_info": {
                "generated_at": rss_data.get("generated_at"),
                "total_entries": rss_data.get("total_entries", 0),
                "time_range": rss_data.get("time_range", "unknown")
            },
            "sources": sources,
            "sample_entries": entries[:5]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка чтения файла: {str(e)}")

# ==================== GENERAL ====================

@app.get("/", tags=["General"])
async def root():
    return {
        "message": "News Processing & Recommendation API v2.0",
        "database": "PostgreSQL",
        "features": [
            "Automatic RSS JSON processing",
            "News clustering with DBSCAN",
            "Personalized recommendations",
            "User interest tracking",
            "Normalized hashtags (format: #tag_name)"
        ],
        "endpoints": {
            "docs": "/docs",
            "rss_processing": "/api/v1/articles/process-rss",
            "recommendations": "/api/v1/recommendations"
        }
    }


@app.get("/health", tags=["General"])
async def health_check(db: Session = Depends(get_db)):
    """Проверка здоровья сервиса для Docker"""
    try:
        db.execute("SELECT 1")
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    rss_file_exists = RSS_JSON_PATH.exists() if RSS_JSON_PATH else False
    
    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "database": db_status,
        "rss_file": "found" if rss_file_exists else "not found",
        "rss_file_path": str(RSS_JSON_PATH),
        "timestamp": datetime.utcnow().isoformat()
    }

# ==================== ЗАПУСК ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False,
        log_level=os.getenv("LOG_LEVEL", "info")
    )