import logging
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

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

    def _convert_pdf_to_text(self, pdf_path: Path, output_dir: Path) -> Optional[Path]:
        try:
            reader = PdfReader(str(pdf_path))
            text_content = []
            for page in reader.pages:
                text_content.append(page.extract_text())

            text = "\n".join(text_content)
            output_path = output_dir / f"{pdf_path.stem}.txt"

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)

            logger.info(f"Converted {pdf_path.name} to {output_path.name}")
            return output_path
        except Exception as e:
            logger.error(f"Error converting PDF {pdf_path.name}: {e}")
            return None

    def _copy_markdown_as_text(self, md_path: Path, output_dir: Path) -> Optional[Path]:
        try:
            output_path = output_dir / f"{md_path.stem}.txt"
            shutil.copy2(md_path, output_path)
            logger.info(f"Copied {md_path.name} to {output_path.name}")
            return output_path
        except Exception as e:
            logger.error(f"Error copying markdown {md_path.name}: {e}")
            return None

    def _process_files_to_text(self, source_path: str) -> Optional[Path]:
        source_dir = Path(source_path)
        temp_dir = source_dir / ".temp_text"

        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(exist_ok=True)

        processed_files = []

        for file_path in source_dir.iterdir():
            if file_path.is_file() and not file_path.name.startswith("."):
                if file_path.suffix.lower() == ".pdf":
                    txt_path = self._convert_pdf_to_text(file_path, temp_dir)
                    if txt_path:
                        processed_files.append(txt_path)
                elif file_path.suffix.lower() in [".md", ".markdown"]:
                    txt_path = self._copy_markdown_as_text(file_path, temp_dir)
                    if txt_path:
                        processed_files.append(txt_path)
                elif file_path.suffix.lower() == ".txt":
                    txt_path = temp_dir / file_path.name
                    shutil.copy2(file_path, txt_path)
                    processed_files.append(txt_path)

        logger.info(f"Processed {len(processed_files)} files to text format")
        return temp_dir if processed_files else None

    def load_documents(self, source_path: str) -> List[Any]:
        logger.info(f"Loading documents from: {source_path}")
        documents = []
        source_dir = Path(source_path)

        if not source_dir.exists():
            logger.error(f"Source directory does not exist: {source_path}")
            return documents

        text_dir = self._process_files_to_text(source_path)

        if not text_dir:
            logger.warning("No text files to process")
            return documents

        try:
            for file_path in text_dir.glob("*.txt"):
                try:
                    loader = TextLoader(str(file_path))
                    docs = loader.load()
                    documents.extend(docs)
                    logger.info(f"Loaded {len(docs)} documents from {file_path.name}")
                except Exception as e:
                    logger.error(f"Error loading {file_path.name}: {e}")
        finally:
            if text_dir and text_dir.name == ".temp_text" and text_dir.exists():
                shutil.rmtree(text_dir)
                logger.info("Cleaned up temporary text files")

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
            retrieved_docs = retriever.invoke(query)

            if not retrieved_docs:
                logger.warning("No documents retrieved, falling back to direct query")
                return self._direct_query(query)

            context = "\n\n".join([doc.page_content for doc in retrieved_docs])

            prompt = f"Answer the question based \
            on the following context:\n\n{context}\n\nQuestion: {query}"

            llm = Ollama(base_url=self.ollama_url, model=self.model)
            response = llm.invoke(prompt)

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

    def query_with_rag(
        self, query: str, provider: str, rag_source: str, model: str = None
    ) -> str:
        logger.info(
            f"RAG query - Provider: {provider}, Source: {rag_source}, Model: {model}"
        )

        if model:
            from .rag import OllamaRAGProvider

            ollama_config = {
                "model": model,
                "ollama_url": os.getenv("OLLAMA_URL", "http://localhost:11434"),
            }
            rag_provider = OllamaRAGProvider(ollama_config)
        else:
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
