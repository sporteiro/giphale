import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGProvider(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def load_documents(self, source_path: str) -> List[Any]:
        pass

    @abstractmethod
    def create_retriever(self, documents: List[Any]) -> Any:
        pass

    @abstractmethod
    def generate_response(self, query: str, retriever: Any) -> str:
        pass


class OllamaRAGProvider(RAGProvider):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = config.get("model", "qwen2.5-coder:7b")
        self.ollama_url = config.get("ollama_url", "http://localhost:11434")
        logger.info("Ollama RAG provider initialized successfully")

    def load_documents(self, source_path: str) -> List[Any]:
        logger.info(f"Loading documents from: {source_path}")
        documents = []
        source_dir = Path(source_path)

        if not source_dir.exists():
            logger.error(f"Source directory does not exist: {source_path}")
            return documents

        for file_path in source_dir.glob("*.txt"):
            try:
                loader = TextLoader(str(file_path))
                docs = loader.load()
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} documents from {file_path.name}")
            except Exception as e:
                logger.error(f"Error loading {file_path.name}: {e}")

        logger.info(f"Total documents loaded: {len(documents)}")
        return documents

    def create_retriever(self, documents: List[Any]) -> Any:
        if not documents:
            logger.warning("No documents to create retriever")
            return None

        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)

            embeddings = OllamaEmbeddings(base_url=self.ollama_url, model=self.model)

            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

            retriever = vectorstore.as_retriever()
            logger.info("Retriever created successfully")
            return retriever

        except Exception as e:
            logger.error(f"Error creating retriever: {e}")
            return None

    def generate_response(self, query: str, retriever: Any) -> str:
        if not retriever:
            logger.warning("No retriever available, falling back to direct query")
            return self._direct_query(query)

        try:
            llm = Ollama(base_url=self.ollama_url, model=self.model)

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
            )

            result = qa_chain.invoke({"query": query})
            response = result.get("result", "")

            logger.info("RAG response generated successfully")
            return response

        except Exception as e:
            logger.error(f"Error in RAG generation: {e}")
            return self._direct_query(query)

    def _direct_query(self, query: str) -> str:
        try:
            llm = Ollama(base_url=self.ollama_url, model=self.model)
            response = llm.invoke(query)
            return response
        except Exception as e:
            logger.error(f"Error in direct query: {e}")
            return f"Error: {str(e)}"


class RAGService:
    def __init__(self):
        self.providers = {}
        self.rag_sources_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "rag_sources"
        )
        self._initialize_providers()

    def _initialize_providers(self):
        ollama_config = {
            "model": os.getenv("LLM_MODEL", "qwen2.5-coder:7b"),
            "ollama_url": os.getenv("OLLAMA_URL", "http://localhost:11434"),
        }

        try:
            self.providers["ollama"] = OllamaRAGProvider(ollama_config)
            logger.info("Ollama RAG provider registered")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama RAG provider: {e}")

    def get_provider(self, provider_name: str) -> RAGProvider:
        return self.providers.get(provider_name)

    def query_with_rag(self, query: str, provider: str, rag_source: str) -> str:
        logger.info(f"RAG query - Provider: {provider}, Source: {rag_source}")

        rag_provider = self.get_provider(provider)
        if not rag_provider:
            raise ValueError(f"RAG provider not available: {provider}")

        source_path = os.path.join(self.rag_sources_dir, rag_source)
        documents = rag_provider.load_documents(source_path)

        if not documents:
            logger.warning(f"No documents found in {rag_source}")
            return rag_provider._direct_query(query)

        retriever = rag_provider.create_retriever(documents)
        if not retriever:
            logger.warning("Failed to create retriever, using direct query")
            return rag_provider._direct_query(query)

        return rag_provider.generate_response(query, retriever)
