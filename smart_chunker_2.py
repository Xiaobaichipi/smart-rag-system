import requests
import json
import pickle
import os
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
import hashlib


class SmartRAGSystem:
    """使用本地Ollama模型进行智能文档切分和RAG检索的完整系统"""

    def __init__(self, model_name: str = "deepseek-r1:14b",
                 ollama_base_url: str = "http://localhost:11434",
                 knowledge_base_path: str = "./knowledge_base"):
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url
        self.knowledge_base_path = knowledge_base_path
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

        # 创建知识库目录
        os.makedirs(knowledge_base_path, exist_ok=True)

        # 验证Ollama连接
        self._check_ollama_connection()

        # 加载现有知识库
        self.knowledge_base = self._load_knowledge_base()
        self.chunk_embeddings = self._load_embeddings()

    def _load_knowledge_base(self) -> List[Dict]:
        """加载本地知识库"""
        kb_file = os.path.join(self.knowledge_base_path, "knowledge_base.pkl")
        if os.path.exists(kb_file):
            with open(kb_file, 'rb') as f:
                return pickle.load(f)
        return []

    def _save_knowledge_base(self):
        """保存知识库到本地"""
        kb_file = os.path.join(self.knowledge_base_path, "knowledge_base.pkl")
        with open(kb_file, 'wb') as f:
            pickle.dump(self.knowledge_base, f)

    def _load_embeddings(self) -> Optional[np.ndarray]:
        """加载预计算的嵌入向量"""
        emb_file = os.path.join(self.knowledge_base_path, "embeddings.pkl")
        if os.path.exists(emb_file):
            with open(emb_file, 'rb') as f:
                return pickle.load(f)
        return None

    def _save_embeddings(self, embeddings: np.ndarray):
        """保存嵌入向量到本地"""
        emb_file = os.path.join(self.knowledge_base_path, "embeddings.pkl")
        with open(emb_file, 'wb') as f:
            pickle.dump(embeddings, f)

    def _check_ollama_connection(self) -> bool:
        """检查Ollama服务连接状态"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model["name"] for model in models]
                if self.model_name in model_names:
                    print(f"✓ 成功连接到Ollama，模型 {self.model_name} 可用")
                    return True
                else:
                    print(f"⚠ Ollama连接成功，但模型 {self.model_name} 不可用")
                    print(f"可用模型: {model_names}")
                    return False
            else:
                print(f"✗ 无法连接到Ollama服务: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"✗ Ollama连接失败: {e}")
            print("请确保Ollama服务正在运行: ollama serve")
            return False

    def _call_llm(self, messages: List[Dict], temperature: float = 0.1) -> str:
        """调用本地Ollama模型"""
        try:
            # 将messages格式转换为Ollama格式
            if len(messages) == 1:
                prompt = messages[0]["content"]
            else:
                # 合并系统提示和用户输入
                system_msg = ""
                user_msg = ""
                for msg in messages:
                    if msg["role"] == "system":
                        system_msg = msg["content"]
                    elif msg["role"] == "user":
                        user_msg = msg["content"]

                if system_msg:
                    prompt = f"系统提示: {system_msg}\n\n用户输入: {user_msg}"
                else:
                    prompt = user_msg

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": 0.9,
                    "max_tokens": 2048
                }
            }

            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                print(f"Ollama API调用失败: {response.status_code}, {response.text}")
                return ""

        except requests.exceptions.Timeout:
            print("Ollama调用超时，请检查模型是否正常运行")
            return ""
        except Exception as e:
            print(f"Ollama调用异常: {e}")
            return ""

    def semantic_chunking(self, document: str, max_chunk_size: int = 1000) -> List[Dict]:
        """基于语义的智能文档切分"""

        # 生成文档指纹，避免重复处理
        doc_hash = hashlib.md5(document.encode()).hexdigest()

        # 使用大模型识别文档结构
        structure_prompt = f"""
        请分析以下文档的结构并识别自然断点（如章节、主题转换、段落分界等）。
        返回断点位置的字符索引列表。

        文档内容：
        {document[:2000]}...  # 限制输入长度

        请严格按照以下JSON格式返回：
        {{"breakpoints": [100, 250, 500, 800]}}
        """

        messages = [
            {"role": "system", "content": "你是一个专业的文档结构分析助手，擅长识别文档的自然分界点。"},
            {"role": "user", "content": structure_prompt}
        ]

        llm_response = self._call_llm(messages)

        try:
            # 尝试解析JSON响应
            if llm_response.strip():
                # 提取JSON部分
                json_start = llm_response.find('{')
                json_end = llm_response.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = llm_response[json_start:json_end]
                    structure = json.loads(json_str)
                    breakpoints = structure.get("breakpoints", [])
                else:
                    breakpoints = []
            else:
                breakpoints = []
        except (json.JSONDecodeError, ValueError) as e:
            print(f"解析断点JSON失败: {e}")
            breakpoints = []

        # 如果没有识别到断点，使用基于长度的切分
        if not breakpoints:
            return self.fallback_chunking(document, max_chunk_size)

        # 根据断点切分文档
        chunks = []
        start = 0

        # 确保断点是有序的且在文档范围内
        breakpoints = [bp for bp in sorted(breakpoints) if 0 <= bp <= len(document)]

        for i, end in enumerate(breakpoints):
            if end <= start:
                continue

            chunk_content = document[start:end].strip()
            if len(chunk_content) > 50:  # 忽略过短的块
                metadata = self.extract_chunk_metadata(chunk_content)
                chunks.append({
                    "id": f"{doc_hash}_{i}",
                    "content": chunk_content,
                    "start": start,
                    "end": end,
                    "metadata": metadata,
                    "doc_hash": doc_hash
                })
            start = end

        # 添加最后一个块
        if start < len(document):
            final_content = document[start:].strip()
            if len(final_content) > 50:
                metadata = self.extract_chunk_metadata(final_content)
                chunks.append({
                    "id": f"{doc_hash}_final",
                    "content": final_content,
                    "start": start,
                    "end": len(document),
                    "metadata": metadata,
                    "doc_hash": doc_hash
                })

        return chunks

    def extract_chunk_metadata(self, chunk: str) -> Dict:
        """使用大模型提取块元数据"""
        metadata_prompt = f"""
        请分析以下文本片段并提取关键信息：

        文本：{chunk[:800]}

        请严格按照以下JSON格式返回：
        {{
            "key_topics": ["主题1", "主题2"],
            "entities": ["实体1", "实体2"],
            "summary": "简短摘要(不超过100字)",
            "keywords": ["关键词1", "关键词2", "关键词3"]
        }}
        """

        messages = [
            {"role": "system", "content": "你是一个专业的文本分析助手，擅长提取文本的关键信息。"},
            {"role": "user", "content": metadata_prompt}
        ]

        llm_response = self._call_llm(messages)

        try:
            # 提取JSON部分
            json_start = llm_response.find('{')
            json_end = llm_response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = llm_response[json_start:json_end]
                metadata = json.loads(json_str)

                # 验证并设置默认值
                return {
                    "key_topics": metadata.get("key_topics", []),
                    "entities": metadata.get("entities", []),
                    "summary": metadata.get("summary", "")[:200],  # 限制摘要长度
                    "keywords": metadata.get("keywords", [])
                }
            else:
                return self._get_default_metadata()
        except (json.JSONDecodeError, ValueError) as e:
            print(f"解析元数据JSON失败: {e}")
            return self._get_default_metadata()

    def _get_default_metadata(self) -> Dict:
        """返回默认元数据"""
        return {
            "key_topics": [],
            "entities": [],
            "summary": "",
            "keywords": []
        }

    def fallback_chunking(self, document: str, max_chunk_size: int = 1000) -> List[Dict]:
        """降级方案：基于长度的文档切分"""
        chunks = []
        doc_hash = hashlib.md5(document.encode()).hexdigest()

        for i in range(0, len(document), max_chunk_size):
            chunk_content = document[i:i + max_chunk_size]
            if len(chunk_content.strip()) > 50:
                chunks.append({
                    "id": f"{doc_hash}_fallback_{i}",
                    "content": chunk_content,
                    "start": i,
                    "end": min(i + max_chunk_size, len(document)),
                    "metadata": self._get_default_metadata(),
                    "doc_hash": doc_hash
                })

        return chunks

    def build_knowledge_base(self, documents: List[str]) -> None:
        """构建知识库"""
        print("开始构建知识库...")

        all_chunks = []
        for doc_idx, document in enumerate(documents):
            print(f"处理文档 {doc_idx + 1}/{len(documents)}")
            chunks = self.semantic_chunking(document)
            all_chunks.extend(chunks)

        # 更新知识库
        self.knowledge_base = all_chunks

        # 计算所有块的TF-IDF嵌入
        chunk_texts = [chunk["content"] for chunk in all_chunks]
        if chunk_texts:
            self.chunk_embeddings = self.vectorizer.fit_transform(chunk_texts).toarray()
        else:
            self.chunk_embeddings = np.array([])

        # 保存到本地
        self._save_knowledge_base()
        self._save_embeddings(self.chunk_embeddings)

        print(f"知识库构建完成，共 {len(all_chunks)} 个文档块")

    def extract_key_information(self, user_query: str) -> str:
        """使用推理模型提取关键信息"""
        extraction_prompt = f"""
        请从用户问题中提取关键信息和检索词，这些信息将用于在知识库中进行检索。

        用户问题：{user_query}

        请提取：
        1. 核心关键词
        2. 主要概念
        3. 相关术语

        请将提取的关键信息整理成一个简洁的检索查询，保持与原问题的高度相关性。

        输出格式：直接返回检索查询文本，不需要其他格式。
        """

        messages = [
            {"role": "system", "content": "你是一个专业的信息提取助手，擅长从用户查询中提取关键检索信息。"},
            {"role": "user", "content": extraction_prompt}
        ]

        key_info = self._call_llm(messages, temperature=0.1)
        return key_info.strip() if key_info else user_query

    def calculate_query_relevance(self, user_query: str, key_info: str) -> float:
        """计算关键信息与用户查询的相关性"""
        if not key_info or not user_query:
            return 0.0

        try:
            # 使用TF-IDF计算相似度
            texts = [user_query, key_info]
            temp_vectorizer = TfidfVectorizer()
            vectors = temp_vectorizer.fit_transform(texts)
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"计算查询相关性失败: {e}")
            return 0.5  # 返回中等相关性作为默认值

    def retrieve_relevant_chunks(self, key_info: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """在知识库中检索相关块"""
        if not self.knowledge_base or self.chunk_embeddings is None or len(self.chunk_embeddings) == 0:
            return []

        try:
            # 将关键信息转换为向量
            query_vector = self.vectorizer.transform([key_info]).toarray()

            # 计算与所有块的相似度
            similarities = cosine_similarity(query_vector, self.chunk_embeddings)[0]

            # 获取top_k个最相似的块
            top_indices = np.argsort(similarities)[::-1][:top_k]

            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # 过滤低相关性结果
                    chunk = self.knowledge_base[idx]
                    results.append((chunk, float(similarities[idx])))

            return results

        except Exception as e:
            print(f"检索相关块失败: {e}")
            return []

    def calculate_retrieval_relevance(self, key_info: str, retrieved_chunks: List[Tuple[Dict, float]]) -> List[
        Tuple[Dict, float, float]]:
        """计算检索结果与关键信息的相关性"""
        results = []

        for chunk, initial_score in retrieved_chunks:
            try:
                # 使用TF-IDF计算关键信息与块内容的相似度
                texts = [key_info, chunk["content"]]
                temp_vectorizer = TfidfVectorizer()
                vectors = temp_vectorizer.fit_transform(texts)
                content_similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

                # 综合评分（初始检索分数 + 内容相似度）
                final_score = (initial_score + content_similarity) / 2

                results.append((chunk, initial_score, float(final_score)))

            except Exception as e:
                print(f"计算检索相关性失败: {e}")
                results.append((chunk, initial_score, initial_score))

        # 按最终得分排序
        results.sort(key=lambda x: x[2], reverse=True)
        return results

    def query_knowledge_base(self, user_query: str, top_k: int = 5,
                             relevance_threshold: float = 0.3) -> Dict:
        """完整的RAG查询流程"""

        # 步骤1：提取关键信息
        key_info = self.extract_key_information(user_query)

        # 步骤2：计算关键信息与原查询的相关性
        query_relevance = self.calculate_query_relevance(user_query, key_info)

        # 步骤3：如果相关性太低，使用原查询进行检索
        if query_relevance < relevance_threshold:
            print(f"关键信息提取相关性较低({query_relevance:.3f})，使用原查询进行检索")
            search_query = user_query
        else:
            search_query = key_info

        # 步骤4：在知识库中检索
        retrieved_chunks = self.retrieve_relevant_chunks(search_query, top_k)

        # 步骤5：计算检索结果的相关性
        final_results = self.calculate_retrieval_relevance(key_info, retrieved_chunks)

        return {
            "user_query": user_query,
            "extracted_key_info": key_info,
            "query_relevance": query_relevance,
            "search_query_used": search_query,
            "retrieved_chunks": final_results[:top_k],
            "total_chunks_in_kb": len(self.knowledge_base)
        }

    def generate_answer(self, user_query: str, context_chunks: List[Dict]) -> str:
        """基于检索到的上下文生成答案"""
        if not context_chunks:
            return "抱歉，在知识库中没有找到相关信息来回答您的问题。"

        # 构建上下文
        context = "\n\n".join([
            f"文档片段{i + 1}:\n{chunk['content'][:500]}..."
            for i, (chunk, _, _) in enumerate(context_chunks[:3])
        ])

        answer_prompt = f"""
        基于以下文档片段回答用户问题，请确保答案准确、相关且有依据。

        用户问题：{user_query}

        相关文档片段：
        {context}

        请提供一个准确、详细的答案，并指出信息来源。如果文档片段中没有足够的信息回答问题，请如实说明。
        """

        messages = [
            {"role": "system", "content": "你是一个专业的知识问答助手，基于提供的文档内容准确回答用户问题。"},
            {"role": "user", "content": answer_prompt}
        ]

        return self._call_llm(messages, temperature=0.2)

    def complete_rag_pipeline(self, user_query: str) -> Dict:
        """完整的RAG管道：查询->检索->生成答案"""

        # 查询知识库
        retrieval_results = self.query_knowledge_base(user_query)

        # 生成答案
        answer = self.generate_answer(user_query, retrieval_results["retrieved_chunks"])

        return {
            **retrieval_results,
            "generated_answer": answer
        }


# 使用示例
if __name__ == "__main__":
    # 初始化系统 (不再需要API密钥)
    rag_system = SmartRAGSystem(
        model_name="deepseek-r1:14b",
        ollama_base_url="http://localhost:11434"  # 默认Ollama服务地址
    )

    # 示例文档
    sample_documents = [
        "人工智能是计算机科学的一个分支，致力于创造能够执行通常需要人类智能的任务的系统。机器学习是人工智能的一个重要子领域。",
        "深度学习是机器学习的一个分支，使用人工神经网络来模拟人脑的学习过程。它在图像识别、自然语言处理等领域取得了巨大成功。",
        "自然语言处理（NLP）是人工智能的一个重要应用领域，专注于让计算机理解和生成人类语言。"
    ]

    # 构建知识库（首次运行时）
    if not rag_system.knowledge_base:
        rag_system.build_knowledge_base(sample_documents)

    # 查询示例
    user_question = "什么是深度学习？"
    result = rag_system.complete_rag_pipeline(user_question)

    print("=" * 60)
    print(f"用户问题: {result['user_query']}")
    print(f"提取的关键信息: {result['extracted_key_info']}")
    print(f"查询相关性: {result['query_relevance']:.3f}")
    print(f"使用的搜索查询: {result['search_query_used']}")
    print(f"检索到的相关块数量: {len(result['retrieved_chunks'])}")
    print("\n生成的答案:")
    print(result['generated_answer'])
    print("=" * 60)