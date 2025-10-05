import json
import re
from typing import List, Dict, Any, Set, Optional
from collections import defaultdict, Counter
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import networkx as nx
import warnings
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsNERTagger,
    Doc
)

warnings.filterwarnings('ignore')


class UnifiedNewsService:
    
    def __init__(self, model_name='deepvk/USER-bge-m3'):
        
        self.model = SentenceTransformer(model_name)

        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.ner_tagger = NewsNERTagger(self.emb)
        
        self.user_profile = {
            'read_articles': [],
            'interested_hashtags': Counter(),
            'interested_entities': Counter(),
            'interested_clusters': Counter()
        }
    
    # ==================== СУММАРИЗАЦИЯ ====================
    
    def positional_text_rank(self, text: str, max_sentences: int = 2) -> str:
        if not text or len(text.strip()) == 0:
            return text
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(sentences) <= max_sentences:
            return text
        
        embeddings = self.model.encode(sentences)
        similarity_matrix = cosine_similarity(embeddings)

        graph = nx.Graph()
        for i in range(len(sentences)):
            graph.add_node(i)
        
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                if similarity_matrix[i][j] > 0.1:
                    graph.add_edge(i, j, weight=similarity_matrix[i][j])
        
        try:
            textrank_scores = nx.pagerank(graph, weight='weight', max_iter=100)
        except:
            textrank_scores = {i: 1.0 / len(sentences) for i in range(len(sentences))}
        
        position_scores = {}
        for i in range(len(sentences)):
            if i == 0:
                position_scores[i] = 1.0
            elif i == 1:
                position_scores[i] = 0.8
            else:
                position_scores[i] = 1.0 / (i + 1)
        
        length_scores = {}
        for i, sent in enumerate(sentences):
            word_count = len(sent.split())
            if word_count < 5:
                length_scores[i] = 0.5
            elif word_count > 50:
                length_scores[i] = 0.7
            else:
                length_scores[i] = 1.0

        final_scores = {}
        for i in range(len(sentences)):
            final_scores[i] = (
                0.4 * textrank_scores[i] +
                0.5 * position_scores[i] +
                0.1 * length_scores[i]
            )
        
        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        top_indices = sorted([idx for idx, score in ranked[:max_sentences]])
        
        result = '. '.join([sentences[i] for i in top_indices])
        return result + '.' if not result.endswith('.') else result
    
    # ==================== ИЗВЛЕЧЕНИЕ ХЕШТЕГОВ ====================
    
    def extract_hashtags_from_text(self, text: str) -> List[str]:
        if not text:
            return []
        
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        doc.tag_ner(self.ner_tagger)
        
        for span in doc.spans:
            span.normalize(self.morph_vocab)
        
        hashtags = set()
        entities = {'PER': [], 'LOC': [], 'ORG': []}
        
        for span in doc.spans:
            if span.type in ['PER', 'LOC', 'ORG']:
                normalized_text = span.normal
                hashtag = '#' + normalized_text.replace(' ', '_')
                hashtags.add(hashtag)
                entities[span.type].append(normalized_text)
        
        return sorted(list(hashtags)), entities
    
    # ==================== КЛАСТЕРИЗАЦИЯ ====================
    
    def preprocess_summaries(self, articles: List[Dict]) -> List[str]:
        processed = []
        for article in articles:
            summary = article.get('summary', '')
            short_summary = self.positional_text_rank(summary, max_sentences=2)
            processed.append(short_summary)
            article['short_summary'] = short_summary
        
        return processed
    
    def cluster_articles(self, articles: List[Dict], eps: float = 0.4, 
                        min_samples: int = 1) -> Dict[int, List[Dict]]:
        if not articles:
            return {}
        
        short_summaries = self.preprocess_summaries(articles)
        embeddings = self.model.encode(short_summaries)
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = clustering.fit_predict(embeddings)
        
        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            articles[idx]['cluster_id'] = int(label)
            clusters[label].append(articles[idx])
        
        return dict(clusters)
    
    def generate_cluster_summary(self, articles: List[Dict]) -> str:
        if not articles:
            return "Новости без общей темы"
        
        if len(articles) == 1:
            return f"Единичная новость: {articles[0]['title']}"
        
        summaries = [a.get('short_summary', a.get('summary', '')) for a in articles]
        all_text = ' '.join(summaries)
        
        words = re.findall(r'\b[а-яёА-ЯЁa-zA-Z]{4,}\b', all_text.lower())
        word_freq = defaultdict(int)
        
        stop_words = {
            'этот', 'этого', 'который', 'которые', 'была', 'были', 'быть',
            'если', 'также', 'того', 'этом', 'более', 'всех', 'может',
            'сказал', 'заявил', 'стало', 'будет', 'году', 'года', 'годах',
            'news', 'said', 'that', 'this', 'will', 'have', 'been', 'with'
        }
        
        for word in words:
            if word not in stop_words:
                word_freq[word] += 1
        
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        keywords = [w[0] for w in top_words]
        
        if len(articles) > 3:
            cluster_type = "Серия новостей"
        else:
            cluster_type = "Группа связанных новостей"
        
        return f"{cluster_type} о {', '.join(keywords[:3])} ({len(articles)} статей)"
    
    # ==================== ПРОФИЛЬ ПОЛЬЗОВАТЕЛЯ ====================
    
    def update_user_profile(self, article: Dict):
        self.user_profile['read_articles'].append(article['link'])
        
        for hashtag in article.get('hashtags', []):
            self.user_profile['interested_hashtags'][hashtag] += 1
        
        entities = article.get('entities', {})
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                self.user_profile['interested_entities'][entity] += 1
        
        if 'cluster_id' in article:
            self.user_profile['interested_clusters'][article['cluster_id']] += 1
    
    def calculate_article_score(self, article: Dict) -> float:
        """
        Вычисляет релевантность статьи для пользователя
        На основе:
        - Совпадения хештегов
        - Совпадения сущностей
        - Принадлежности к интересным кластерам
        - Новизны (не читал ли пользователь)
        """
        score = 0.0
        
        if article['link'] in self.user_profile['read_articles']:
            return -1000.0
        
        hashtag_score = 0.0
        article_hashtags = set(article.get('hashtags', []))
        user_hashtags = set(self.user_profile['interested_hashtags'].keys())
        
        if article_hashtags and user_hashtags:
            common_hashtags = article_hashtags & user_hashtags
            for hashtag in common_hashtags:
                hashtag_score += self.user_profile['interested_hashtags'][hashtag]
        
        score += 0.4 * hashtag_score
        
        entity_score = 0.0
        article_entities = set()
        for entity_list in article.get('entities', {}).values():
            article_entities.update(entity_list)
        
        user_entities = set(self.user_profile['interested_entities'].keys())
        
        if article_entities and user_entities:
            common_entities = article_entities & user_entities
            for entity in common_entities:
                entity_score += self.user_profile['interested_entities'][entity]
        
        score += 0.3 * entity_score
        
        cluster_score = 0.0
        if 'cluster_id' in article:
            cluster_id = article['cluster_id']
            if cluster_id in self.user_profile['interested_clusters']:
                cluster_score = self.user_profile['interested_clusters'][cluster_id]
        
        score += 0.2 * cluster_score
        
        freshness_score = 10.0
        score += 0.1 * freshness_score
        
        return score
    
    def recommend_articles(self, articles: List[Dict], top_n: int = 10) -> List[Dict]:
        if not self.user_profile['read_articles']:
            return sorted(articles, 
                         key=lambda x: x.get('cluster_size', 0), 
                         reverse=True)[:top_n]
        
        scored_articles = []
        for article in articles:
            score = self.calculate_article_score(article)
            if score > -1000.0:
                scored_articles.append({
                    'article': article,
                    'score': score
                })
        
        scored_articles.sort(key=lambda x: x['score'], reverse=True)

        recommendations = []
        for item in scored_articles[:top_n]:
            article = item['article'].copy()
            article['recommendation_score'] = round(item['score'], 2)
            article['recommendation_reason'] = self._generate_recommendation_reason(article)
            recommendations.append(article)
        
        return recommendations
    
    def _generate_recommendation_reason(self, article: Dict) -> str:
        reasons = []
        
        article_hashtags = set(article.get('hashtags', []))
        user_hashtags = set(self.user_profile['interested_hashtags'].keys())
        common_hashtags = article_hashtags & user_hashtags
        
        if common_hashtags:
            top_hashtags = sorted(common_hashtags, 
                                 key=lambda h: self.user_profile['interested_hashtags'][h], 
                                 reverse=True)[:2]
            reasons.append(f"Темы: {', '.join(top_hashtags)}")

        article_entities = set()
        for entity_list in article.get('entities', {}).values():
            article_entities.update(entity_list)
        
        user_entities = set(self.user_profile['interested_entities'].keys())
        common_entities = article_entities & user_entities
        
        if common_entities:
            top_entities = sorted(common_entities,
                                 key=lambda e: self.user_profile['interested_entities'][e],
                                 reverse=True)[:2]
            reasons.append(f"Упоминаются: {', '.join(top_entities)}")
        
        if 'cluster_id' in article:
            cluster_id = article['cluster_id']
            if cluster_id in self.user_profile['interested_clusters']:
                reasons.append(f"Похожие новости интересовали вас ранее")
        
        return " | ".join(reasons) if reasons else "Новая тема"
    
    # ==================== ПОЛНЫЙ PIPELINE ====================
    
    def process_articles(self, articles: List[Dict], 
                        eps: float = 0.4, 
                        min_samples: int = 1) -> Dict[str, Any]:
        for article in articles:
            text = article.get('summary', '') + ' ' + article.get('title', '')
            hashtags, entities = self.extract_hashtags_from_text(text)
            article['hashtags'] = hashtags[:10]
            article['entities'] = entities

        clusters = self.cluster_articles(articles, eps=eps, min_samples=min_samples)
        
        for cluster_id, cluster_articles in clusters.items():
            for article in cluster_articles:
                article['cluster_size'] = len(cluster_articles)

        result = self._format_output(clusters)
        
        return result
    
    def _format_output(self, clusters: Dict[int, List[Dict]]) -> Dict[str, Any]:
        result = {
            "clusters": [],
            "total_articles": sum(len(articles) for articles in clusters.values()),
            "total_clusters": len([k for k in clusters.keys() if k != -1])
        }
        
        for cluster_id, articles in clusters.items():
            if cluster_id == -1:
                cluster_name = "Разное (без общей темы)"
            else:
                cluster_name = f"Кластер {cluster_id + 1}"
            
            cluster_summary = self.generate_cluster_summary(articles)
            
            formatted_articles = []
            for article in articles:
                other_articles = [
                    {
                        "title": a['title'],
                        "link": a['link']
                    }
                    for a in articles if a['link'] != article['link']
                ]
                
                formatted_article = {
                    "title": article['title'],
                    "summary": article.get('summary', ''),
                    "short_summary": article.get('short_summary', ''),
                    "link": article['link'],
                    "source": article.get('source_title', 'Unknown'),
                    "published": article.get('published', ''),
                    "hashtags": article.get('hashtags', []),
                    "entities": article.get('entities', {}),
                    "cluster_size": article.get('cluster_size', 1),
                    "related_articles_in_cluster": other_articles if other_articles else None
                }
                formatted_articles.append(formatted_article)
            
            cluster_obj = {
                "cluster_id": cluster_id if cluster_id != -1 else "mixed",
                "cluster_name": cluster_name,
                "cluster_summary": cluster_summary,
                "articles_count": len(articles),
                "articles": formatted_articles
            }
            
            result["clusters"].append(cluster_obj)
        
        result["clusters"].sort(key=lambda x: x["articles_count"], reverse=True)
        
        return result
    
    def get_all_articles_flat(self, processed_data: Dict) -> List[Dict]:
        all_articles = []
        for cluster in processed_data['clusters']:
            all_articles.extend(cluster['articles'])
        return all_articles


def main():
    with open('output/rss_feed_24h.json', "r", encoding="utf-8") as f:
        sample_data = json.loads(f.read())
    
    service = UnifiedNewsService()
    
    result = service.process_articles(
        sample_data['entries'],
        eps=0.4,
        min_samples=1
    )
    
    output_file = 'clustered_news_with_hashtags.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    all_articles = service.get_all_articles_flat(result)

    for i in range(min(3, len(all_articles))):
        article = all_articles[i]
        service.update_user_profile(article)

    recommendations = service.recommend_articles(all_articles, top_n=5)
    
    print("\n📋 ТОП-5 РЕКОМЕНДАЦИЙ:")
    for idx, rec in enumerate(recommendations, 1):
        print(f"\n{idx}. {rec['title']}")
        print(f"   Скор: {rec['recommendation_score']}")
        print(f"   Причина: {rec['recommendation_reason']}")
        print(f"   Хештеги: {', '.join(rec['hashtags'][:5])}")
    
    recommendations_file = 'output/user_recommendations.json'
    with open(recommendations_file, 'w', encoding='utf-8') as f:
        json.dump({
            'user_profile': {
                'read_count': len(service.user_profile['read_articles']),
                'top_hashtags': dict(service.user_profile['interested_hashtags'].most_common(5)),
                'top_entities': dict(service.user_profile['interested_entities'].most_common(5))
            },
            'recommendations': recommendations
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Рекомендации сохранены в {recommendations_file}")
    
    return result, recommendations


if __name__ == "__main__":
    result, recommendations = main()