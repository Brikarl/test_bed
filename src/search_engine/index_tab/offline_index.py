#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线模块 - 索引构建+样本收集+RAG功能
负责倒排索引构建、文档管理、样本收集、RAG检索增强等任务"""

import json
import math
import os
import re
from collections import defaultdict, Counter
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import jieba
import pandas as pd


@dataclass
class RAGResult:
    """RAG结果数据类"""
    query: str
    retrieved_docs: List[Tuple[str, float, str]]
    context: str
    answer: str
    metadata: Dict


class InvertedIndex:
    """倒排索引类"""

    def __init__(self):
        self.index = defaultdict(set)  # 词项 -> 文档ID集合
        self.doc_lengths = {}  # 文档ID -> 文档长度
        self.documents = {}  # 文档ID -> 文档内容
        self.term_freq = defaultdict(dict)  # 词项 -> {文档ID: 词频}
        self.doc_freq = defaultdict(int)  # 词项 -> 文档频率
        self.doc_vectors = {}  # 文档ID -> TF-IDF向量（稀疏表示）
        # 停用词
        self.stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到',
            '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'
            }

    def preprocess_text(self, text: str) -> List[str]:
        """文本预处理"""
        # 分词
        words = jieba.lcut(text.lower())
        # 过滤停用词和短词
        words = [word for word in words if len(word) > 1 and word not in self.stop_words]
        return words

    def add_document(self, doc_id: str, content: str):
        """添加文档到索引"""
        # 保存原始文档
        self.documents[doc_id] = content
        # 预处理文本
        words = self.preprocess_text(content)
        # 计算文档长度
        self.doc_lengths[doc_id] = len(words)
        # 统计词频
        word_freq = Counter(words)
        # 更新倒排索引
        for word, freq in word_freq.items():
            self.index[word].add(doc_id)
            self.term_freq[word][doc_id] = freq
        # 更新文档频率
        for word in word_freq:
            self.doc_freq[word] = len(self.index[word])

    def compute_doc_vectors(self):
        """计算所有文档的TF-IDF向量"""
        total_docs = len(self.documents)
        for doc_id in self.documents:
            doc_vector = {}
            for word in self.index:
                if doc_id in self.index[word]:
                    tf = self.term_freq[word][doc_id] / self.doc_lengths[doc_id]
                    idf = math.log(total_docs / self.doc_freq[word])
                    doc_vector[word] = tf * idf
            self.doc_vectors[doc_id] = doc_vector

    def delete_document(self, doc_id: str) -> bool:
        """删除文档从索引"""
        if doc_id not in self.documents:
            return False
        # 获取文档的词频信息
        content = self.documents[doc_id]
        words = self.preprocess_text(content)
        word_freq = Counter(words)
        # 从倒排索引中移除文档
        for word in word_freq:
            if word in self.index:
                self.index[word].discard(doc_id)
                # 如果词项没有文档了，删除该词项
                if not self.index[word]:
                    del self.index[word]
                    if word in self.term_freq:
                        del self.term_freq[word]
                    if word in self.doc_freq:
                        del self.doc_freq[word]
                else:
                    # 更新词频信息
                    if word in self.term_freq and doc_id in self.term_freq[word]:
                        del self.term_freq[word][doc_id]
                    # 更新文档频率
                    if word in self.doc_freq:
                        self.doc_freq[word] = len(self.index[word])
        # 删除文档相关数据
        del self.documents[doc_id]
        if doc_id in self.doc_lengths:
            del self.doc_lengths[doc_id]
        if doc_id in self.doc_vectors:
            del self.doc_vectors[doc_id]
        return True

    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> List[Tuple[str, float, str]]:
        """搜索文档"""
        # 预处理查询
        query_words = self.preprocess_text(query)
        if not query_words:
            return []
        # 计算TF-IDF分数
        scores = {}
        total_docs = len(self.documents)
        for doc_id in self.documents:
            score = 0
            for word in query_words:
                if word in self.index and doc_id in self.index[word]:
                    # TF
                    tf = self.term_freq[word][doc_id] / self.doc_lengths[doc_id]
                    # IDF
                    idf = math.log(total_docs / self.doc_freq[word])
                    # TF-IDF
                    score += tf * idf
            if score > min_score:
                scores[doc_id] = score
        # 排序并返回结果
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        # 生成摘要
        results = []
        for doc_id, score in sorted_results[:top_k]:
            summary = self.generate_summary(doc_id, query_words)
            results.append((doc_id, score, summary))
        return results

    def generate_summary(self, doc_id: str, query_words: List[str], max_length: int = 200) -> str:
        """生成文档摘要"""
        content = self.documents[doc_id]
        # 找到包含最多查询词的文本窗口
        best_window = ""
        best_score = 0
        # 简单的滑动窗口方法
        words = content.split()
        for i in range(len(words)):
            for j in range(i + 1, min(i + 50, len(words) + 1)):  # 最多50个词
                window = " ".join(words[i:j])
                window_words = self.preprocess_text(window)
                # 计算窗口包含的查询词数量
                score = sum(1 for word in query_words if word in window_words)
                if score > best_score and len(window) <= max_length:
                    best_score = score
                    best_window = window
        if not best_window:
            # 如果没有找到好的窗口，使用文档开头
            best_window = content[:max_length]
        # 高亮查询词
        highlighted_summary = self.highlight_keywords(best_window, query_words)
        return highlighted_summary

    def highlight_keywords(self, text: str, keywords: List[str]) -> str:
        """高亮关键词"""
        highlighted_text = text
        for keyword in keywords:
            if keyword in highlighted_text:
                highlighted_text = highlighted_text.replace(
                    keyword,
                    f'<span style="background-color: yellow; font-weight: bold;">{keyword}</span>'
                    )
        return highlighted_text

    def get_document(self, doc_id: str) -> str:
        """获取文档内容"""
        return self.documents.get(doc_id, "")

    def get_all_documents(self) -> Dict[str, str]:
        """获取所有文档"""
        return self.documents.copy()

    def get_index_stats(self) -> Dict:
        """获取索引统计信息"""
        total_documents = len(self.documents)
        total_terms = len(self.index)
        if total_documents > 0:
            average_doc_length = sum(self.doc_lengths.values()) / total_documents
        else:
            average_doc_length = 0
        return {
            'total_documents': total_documents,
            'total_terms': total_terms,
            'average_doc_length': average_doc_length
            }

    def save_to_file(self, filename: str):
        """保存索引到文件"""
        data = {
            'index': {k: list(v) for k, v in self.index.items()},
            'doc_lengths': self.doc_lengths,
            'documents': self.documents,
            'term_freq': {k: dict(v) for k, v in self.term_freq.items()},
            'doc_freq': dict(self.doc_freq)
            }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ 索引已保存到: {filename}")

    def load_from_file(self, filename: str):
        """从文件加载索引"""
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.index = defaultdict(set)
        for k, v in data['index'].items():
            self.index[k] = set(v)
        self.doc_lengths = data['doc_lengths']
        self.documents = data['documents']
        self.term_freq = defaultdict(dict)
        for k, v in data['term_freq'].items():
            self.term_freq[k] = v
        self.doc_freq = defaultdict(int)
        for k, v in data['doc_freq'].items():
            self.doc_freq[k] = v
        print(f"✅ 索引已从文件加载: {filename}")


class RAGEngine:
    """RAG引擎 - 检索增强生成"""

    def __init__(self, index: InvertedIndex):
        self.index = index
        self.templates = {
            'default': """基于以下相关文档回答问题：

{context}

问题：{query}

答案：""",
            'summary': """请根据以下文档内容总结关于"{query}"的信息：

{context}

总结：""",
            'comparison': """请比较以下文档中关于"{query}"的不同观点或方面：

{context}

比较分析："""
            }

    def retrieve_context(self, query: str, top_k: int = 3, max_context_length: int = 1000) -> Tuple[
        str, List[Tuple[str, float, str]]]:
        """检索相关上下文"""
        # 使用TF-IDF搜索相关文档
        results = self.index.search(query, top_k=top_k)

        # 构建上下文
        context_parts = []
        for doc_id, score, summary in results:
            doc_content = self.index.get_document(doc_id)
            context_parts.append(f"[文档 {doc_id} (相关度: {score:.3f})]:\n{doc_content.strip()}\n")

        context = "\n---\n".join(context_parts)

        # 限制上下文长度
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."

        return context, results

    def generate_answer(self, query: str, context: str, template_type: str = 'default') -> str:
        """生成答案（这里使用简单的模板和规则）"""
        # 选择模板
        template = self.templates.get(template_type, self.templates['default'])
        prompt = template.format(query=query, context=context)

        # 这里可以集成真实的LLM，现在使用简单的规则生成
        answer = self._simple_answer_generation(query, context)

        return answer

    def _simple_answer_generation(self, query: str, context: str) -> str:
        """简单的答案生成（基于规则和模板）"""
        query_words = self.index.preprocess_text(query)

        # 提取相关句子
        sentences = [s.strip() for s in re.split(r'[。！？]', context) if s.strip()]
        relevant_sentences = []

        for sent in sentences:
            sent_words = self.index.preprocess_text(sent)
            # 计算句子与查询的相关度
            relevance = sum(1 for word in query_words if word in sent_words)
            if relevance > 0:
                relevant_sentences.append((relevance, sent))

        # 按相关度排序
        relevant_sentences.sort(key=lambda x: x[0], reverse=True)

        # 生成答案
        if relevant_sentences:
            # 取最相关的2-3个句子
            top_sentences = [sent for _, sent in relevant_sentences[:3]]
            answer = "根据检索到的文档，" + "".join(top_sentences)

            # 添加总结
            if len(relevant_sentences) > 3:
                answer += f"\n\n此外，文档中还包含了其他{len(relevant_sentences) - 3}条相关信息。"
        else:
            answer = f"抱歉，在现有文档中没有找到与'{query}'直接相关的信息。请尝试使用不同的关键词进行搜索。"

        return answer

    def query(
            self,
            query: str,
            top_k: int = 3,
            template_type: str = 'default',
            include_sources: bool = True
            ) -> RAGResult:
        """执行RAG查询"""
        # 1. 检索相关文档
        context, retrieved_docs = self.retrieve_context(query, top_k=top_k)

        # 2. 生成答案
        answer = self.generate_answer(query, context, template_type)

        # 3. 添加来源信息
        if include_sources and retrieved_docs:
            sources = "\n\n信息来源：\n"
            for doc_id, score, _ in retrieved_docs:
                sources += f"- {doc_id} (相关度: {score:.3f})\n"
            answer += sources

        # 4. 构建结果
        result = RAGResult(
            query=query,
            retrieved_docs=retrieved_docs,
            context=context,
            answer=answer,
            metadata={
                'template_type': template_type,
                'top_k': top_k,
                'timestamp': datetime.now().isoformat()
                }
            )

        return result

    def batch_query(self, queries: List[str], **kwargs) -> List[RAGResult]:
        """批量查询"""
        results = []
        for query in queries:
            result = self.query(query, **kwargs)
            results.append(result)
        return results


class SampleCollector:
    """样本收集器"""

    def __init__(self):
        self.samples = []
        self.rag_samples = []  # RAG查询样本

    def add_sample(self, sample: Dict):
        """添加样本"""
        sample['timestamp'] = datetime.now().isoformat()
        self.samples.append(sample)

    def add_rag_sample(self, rag_result: RAGResult, user_feedback: Optional[Dict] = None):
        """添加RAG样本"""
        rag_sample = {
            'query': rag_result.query,
            'retrieved_docs': [(doc_id, score) for doc_id, score, _ in rag_result.retrieved_docs],
            'answer': rag_result.answer,
            'metadata': rag_result.metadata,
            'timestamp': datetime.now().isoformat()
            }
        if user_feedback:
            rag_sample['feedback'] = user_feedback
        self.rag_samples.append(rag_sample)

    def get_samples(self) -> List[Dict]:
        """获取所有样本"""
        return self.samples

    def get_rag_samples(self) -> List[Dict]:
        """获取RAG样本"""
        return self.rag_samples

    def export_samples(self, filename: str):
        """导出样本到文件"""
        # 导出普通样本
        if self.samples:
            df = pd.DataFrame(self.samples)
            df.to_csv(filename, index=False, encoding='utf-8')
            print(f"✅ 样本已导出到: {filename}")

        # 导出RAG样本
        if self.rag_samples:
            rag_filename = filename.replace('.csv', '_rag.csv')
            df_rag = pd.DataFrame(self.rag_samples)
            df_rag.to_csv(rag_filename, index=False, encoding='utf-8')
            print(f"✅ RAG样本已导出到: {rag_filename}")

    def get_stats(self) -> Dict:
        """获取样本统计"""
        stats = {
            'total_samples': len(self.samples),
            'total_rag_samples': len(self.rag_samples),
            }

        if self.samples:
            total_clicks = sum(sample.get('clicked', 0) for sample in self.samples)
            stats['click_rate'] = total_clicks / len(self.samples) if len(self.samples) > 0 else 0

        if self.rag_samples:
            # RAG样本统计
            total_positive_feedback = sum(
                1 for sample in self.rag_samples
                if sample.get('feedback', {}).get('helpful', False)
                )
            stats['rag_helpful_rate'] = total_positive_feedback / len(self.rag_samples) if len(
                self.rag_samples
                ) > 0 else 0

        return stats


def create_sample_documents():
    """创建示例文档"""
    documents = {
        "doc1": """
        人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
        该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
        """,
        "doc2": """
        机器学习是人工智能的一个子集，它使用统计学方法让计算机系统能够"学习"（即，逐步提高特定任务的性能），而无需明确编程。
        机器学习算法通过分析数据来识别模式，并使用这些模式来做出预测或决策。
        """,
        "doc3": """
        深度学习是机器学习的一个分支，它基于人工神经网络，特别是深层神经网络。
        深度学习模型可以自动学习数据的层次表示，这使得它们在图像识别、语音识别和自然语言处理等任务中表现出色。
        """,
        "doc4": """
        自然语言处理是人工智能和语言学的一个交叉领域，它研究计算机与人类语言之间的交互。
        NLP技术被广泛应用于机器翻译、情感分析、问答系统和聊天机器人等应用。
        """,
        "doc5": """
        计算机视觉是人工智能的一个分支，它使计算机能够从数字图像或视频中获得高层次的理解。
        计算机视觉技术包括图像识别、目标检测、图像分割和视频分析等。
        """,
        "doc6": """
        神经网络是一种模仿生物神经系统的计算模型，由大量相互连接的神经元组成。
        神经网络能够学习复杂的非线性关系，在模式识别和预测任务中表现出色。
        """,
        "doc7": """
        强化学习是机器学习的一种方法，通过让智能体与环境交互来学习最优策略。
        强化学习在游戏、机器人控制和自动驾驶等领域有重要应用。
        """,
        "doc8": """
        知识图谱是一种结构化的知识表示方法，将实体和关系组织成图结构。
        知识图谱在搜索引擎、推荐系统和问答系统中发挥重要作用。
        """,
        "doc9": """
        大数据是指无法用传统数据处理软件在合理时间内处理的数据集。
        大数据技术包括数据存储、数据处理、数据分析和数据可视化等方面。
        """,
        "doc10": """
        云计算是一种通过互联网提供计算资源的服务模式。
        云计算包括基础设施即服务、平台即服务和软件即服务等不同层次。
        """
        }
    return documents


def build_index_from_documents(documents: Dict[str, str], save_path: str = ""):
    """从文档构建索引"""
    print("🔨 构建倒排索引...")
    index = InvertedIndex()
    for doc_id, content in documents.items():
        index.add_document(doc_id, content)
        print(f"   添加文档: {doc_id}")

    # 计算文档向量
    index.compute_doc_vectors()

    stats = index.get_index_stats()
    print(f"✅ 索引构建完成:")
    print(f"   总文档数: {stats['total_documents']}")
    print(f"   总词项数: {stats['total_terms']}")
    print(f"   平均文档长度: {stats['average_doc_length']:.2f}")
    if save_path:
        index.save_to_file(save_path)
    return index


def test_rag_functionality(index: InvertedIndex):
    """测试RAG功能"""
    print("\n🤖 测试RAG功能:")
    print("=" * 50)

    # 创建RAG引擎
    rag_engine = RAGEngine(index)

    # 创建样本收集器
    collector = SampleCollector()

    # 测试查询
    test_queries = [
        "什么是人工智能？",
        "机器学习和深度学习有什么区别？",
        "自然语言处理有哪些应用？",
        "计算机视觉包括哪些技术？",
        "大数据和云计算的关系是什么？"
        ]

    for query in test_queries:
        print(f"\n📝 查询: {query}")
        print("-" * 40)

        # 执行RAG查询
        result = rag_engine.query(query, top_k=3)

        print(f"检索到 {len(result.retrieved_docs)} 个相关文档:")
        for doc_id, score, _ in result.retrieved_docs:
            print(f"  - {doc_id}: 相关度 {score:.3f}")

        print(f"\n💡 生成的答案:")
        print(result.answer)

        # 收集样本
        collector.add_rag_sample(result, user_feedback={'helpful': True})

    # 测试不同的模板类型
    print("\n\n📊 测试不同模板类型:")
    print("=" * 50)

    query = "人工智能的各个分支"
    for template_type in ['summary', 'comparison']:
        print(f"\n使用 {template_type} 模板:")
        result = rag_engine.query(query, template_type=template_type)
        print(result.answer)

    # 打印统计信息
    stats = collector.get_stats()
    print("\n\n📈 样本统计:")
    print(f"  - RAG查询样本数: {stats['total_rag_samples']}")
    print(f"  - 有用率: {stats.get('rag_helpful_rate', 0):.2%}")

    return collector


def main():
    """主函数 - 构建示例索引并测试RAG功能"""
    print("🏗️  离线索引构建模块 + RAG功能")
    print("=" * 50)

    # 创建输出目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('samples', exist_ok=True)

    # 创建示例文档
    documents = create_sample_documents()

    # 构建索引
    index = build_index_from_documents(documents, 'models/index_data.json')

    # 测试基础搜索
    print("\n\n🔍 测试基础TF-IDF搜索:")
    print("=" * 50)
    test_queries = ["人工智能", "机器学习", "深度学习"]
    for query in test_queries:
        print(f"\n查询: {query}")
        results = index.search(query, top_k=3)
        for doc_id, score, summary in results:
            print(f"  - {doc_id}: {score:.4f}")

    # 测试RAG功能
    collector = test_rag_functionality(index)

    # 导出样本
    if collector.rag_samples:
        collector.export_samples('samples/search_samples.csv')

    print("\n\n✅ 离线索引构建和RAG功能测试完成!")
    print("💡 现在可以启动在线服务: python online_service.py")
    print("📚 RAG功能已集成，可以提供更智能的搜索答案")


if __name__ == "__main__":
    main()
