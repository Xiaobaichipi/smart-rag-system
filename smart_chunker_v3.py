import hashlib
import json
import os
from typing import List, Dict, Tuple, Optional

import numpy as np
# 【关键修改1】: 引入真正的语义嵌入模型和向量数据库
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import hub


class SmartRAGSystemV2:
    """使用Ollama模型和语义嵌入的RAG系统"""

    def __init__(self,
                 llm_model: str = "deepseek-coder:6.7b",  # 用于生成和分析的模型
                 embedding_model: str = "nomic-embed-text",  # 专门用于嵌入的模型
                 knowledge_base_path: str = "./knowledge_base_chroma"):

        print("正在初始化RAG系统...")
        self.llm_model_name = llm_model
        self.embedding_model_name = embedding_model
        self.knowledge_base_path = knowledge_base_path

        # 【关键修改2】: 初始化LangChain组件，替换手动requests调用
        # 用于生成答案、分析查询等
        self.llm = ChatOllama(model=llm_model, temperature=0.2)
        # 用于将文本转换为语义向量
        self.embeddings = OllamaEmbeddings(model=embedding_model)

        # 【关键修改3】: 初始化向量数据库
        # Chroma会在指定路径下创建文件来持久化存储数据
        self.vector_store = Chroma(
            persist_directory=self.knowledge_base_path,
            embedding_function=self.embeddings
        )

        # 将向量数据库包装成一个检索器
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",  # 使用相似度搜索
            search_kwargs={'k': 5}  # 默认返回5个最相关的结果
        )

        print("✓ RAG系统初始化完成。")

    def build_knowledge_base(self, documents: List[str], chunk_size: int = 800, chunk_overlap: int = 100):
        """
        构建知识库。
        """
        print("开始构建知识库...")
        if not documents:
            print("没有提供任何文档，跳过构建。")
            return

        # 使用基于字符的递归分割器，稳定可靠
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        all_splits = []
        for i, doc in enumerate(documents):
            splits = text_splitter.create_documents([doc], metadatas=[{"source": f"doc_{i}"}])
            all_splits.extend(splits)

        if not all_splits:
            print("文档分割后没有产生任何文本块。")
            return

        print(f"文档被分割成 {len(all_splits)} 个文本块。")

        # 【关键修改4】: 将文本块和其语义向量存入向量数据库
        # add_documents 会自动处理文本的嵌入和存储
        self.vector_store.add_documents(all_splits)
        # 持久化数据到磁盘
        self.vector_store.persist()

        print(f"✓ 知识库构建完成并已持久化到: {self.knowledge_base_path}")

    def query_rewrite(self, user_query: str) -> str:
        """
        使用LLM对用户问题进行重写，使其更适合检索。
        这保留了v2中`extract_key_information`的核心思想。
        """
        rewrite_prompt_template = """你是一个专业的查询优化助手。请分析以下用户问题，并将其改写成一个更清晰、更适合在知识库中进行语义检索的查询。
        保持核心意图不变，但可以补充关键词、纠正错别字或转换为更正式的表述。

        用户原始问题: "{question}"

        优化后的查询:"""

        prompt = rewrite_prompt_template.format(question=user_query)

        try:
            # 直接调用LangChain的LLM组件
            rewritten_query = self.llm.invoke(prompt).content.strip()
            # 如果模型返回空，则使用原问题
            return rewritten_query if rewritten_query else user_query
        except Exception as e:
            print(f"查询重写失败: {e}，将使用原始查询。")
            return user_query

    def complete_rag_pipeline(self, user_query: str):
        """
        完整的RAG管道：查询重写 -> 检索 -> 生成答案
        这里使用了LangChain Expression Language (LCEL) 来构建一个清晰的链条。
        """
        print("\n--- 开始执行RAG流程 ---")

        # 步骤1: 对用户查询进行重写 (可选但推荐)
        rewritten_query = self.query_rewrite(user_query)
        print(f"原始查询: {user_query}")
        print(f"重写查询: {rewritten_query}")

        # 定义RAG的提示模板
        # hub.pull可以从网上拉取一个经过优化的标准RAG提示
        prompt_template = hub.pull("rlm/rag-prompt")

        # 格式化检索到的文档
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # 【关键修改5】: 构建一个标准的、高效的RAG链
        rag_chain = (
            # 这里的 "context" 会接收检索器的输出, "question" 接收原始问题
                {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
                | prompt_template
                | self.llm
                | StrOutputParser()
        )

        # 步骤2 & 3: 使用重写后的查询进行检索，并结合上下文生成答案
        print("正在检索相关信息并生成答案...")
        answer = rag_chain.invoke(rewritten_query)

        # 额外：我们可以看看检索到了哪些内容
        retrieved_docs = self.retriever.invoke(rewritten_query)

        print("--- RAG流程结束 ---")

        return {
            "user_query": user_query,
            "rewritten_query": rewritten_query,
            "generated_answer": answer,
            "retrieved_docs": [doc.page_content for doc in retrieved_docs]
        }


# --- 使用示例 ---
if __name__ == "__main__":
    # 确保你的Ollama服务正在运行，并且已经下载了模型:
    # ollama pull deepseek-coder:6.7b
    # ollama pull nomic-embed-text
    rag_system = SmartRAGSystemV2()

    # 检查知识库是否需要构建
    # 这是一个简化的检查，实际应用可能需要更复杂的版本管理
    if not os.path.exists(rag_system.knowledge_base_path) or not os.listdir(rag_system.knowledge_base_path):
        print("未找到本地知识库，开始构建...")
        sample_documents = [
            "人工智能是计算机科学的一个分支，其目标是创造能够执行通常需要人类智能的任务的机器。机器学习是实现人工智能的一种流行方法。",
            "深度学习是机器学习的一个特定子领域，它利用深度神经网络（具有许多层的网络）来处理数据。它在图像识别、自然语言处理和语音识别等领域取得了革命性的成功。",
            "自然语言处理（NLP）是人工智能和语言学的交叉领域，专注于让计算机能够理解、解释和生成人类语言。常见的NLP任务包括情感分析、机器翻译和问答系统。"
        ]
        rag_system.build_knowledge_base(sample_documents)
    else:
        print("检测到已存在的本地知识库，直接加载。")

    # 查询示例
    user_question = "深度学习在哪些方面很成功？"
    result = rag_system.complete_rag_pipeline(user_question)

    print("\n" + "=" * 60)
    print(f"用户问题: {result['user_query']}")
    print(f"使用的检索查询: {result['rewritten_query']}")
    print("\n生成的答案:")
    print(result['generated_answer'])
    print("\n" + "-" * 20)
    print("检索到的相关上下文片段:")
    for i, doc in enumerate(result['retrieved_docs']):
        print(f"片段 {i + 1}:\n{doc}\n")
    print("=" * 60)
