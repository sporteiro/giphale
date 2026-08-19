import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.rag import RAGService


def test_rag_initialization():
    print("Testing RAG service initialization...")
    try:
        rag_service = RAGService()
        print("✓ RAG service initialized successfully")
        print(f"Available providers: {list(rag_service.providers.keys())}")
        return rag_service
    except Exception as e:
        print(f"✗ RAG service initialization failed: {e}")
        return None


def test_file_conversion(rag_service):
    print("\nTesting file conversion...")
    rag_sources_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "rag_sources"
    )
    
    sources = []
    if os.path.exists(rag_sources_dir):
        for item in os.listdir(rag_sources_dir):
            item_path = os.path.join(rag_sources_dir, item)
            if os.path.isdir(item_path):
                sources.append(item)
    
    print(f"Available RAG sources: {sources}")
    
    if not sources:
        print("✗ No RAG sources found")
        return None
    
    test_source = sources[0]
    print(f"Testing with source: {test_source}")
    
    provider = rag_service.get_provider("ollama")
    if not provider:
        print("✗ Ollama provider not available")
        return None
    
    source_path = os.path.join(rag_sources_dir, test_source)
    print(f"Source path: {source_path}")
    
    try:
        documents = provider.load_documents(source_path)
        print(f"✓ Loaded {len(documents)} documents")
        if documents:
            print(f"First document preview: {documents[0].page_content[:200]}...")
        return documents
    except Exception as e:
        print(f"✗ File conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_retriever_creation(rag_service, documents):
    print("\nTesting retriever creation...")
    if not documents:
        print("✗ No documents to create retriever")
        return None
    
    provider = rag_service.get_provider("ollama")
    try:
        retriever = provider.create_retriever(documents)
        if retriever:
            print("✓ Retriever created successfully")
            return retriever
        else:
            print("✗ Retriever creation returned None")
            return None
    except Exception as e:
        print(f"✗ Retriever creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_rag_query(rag_service, test_query="What is this document about?"):
    print(f"\nTesting RAG query: '{test_query}'")
    rag_sources_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "rag_sources"
    )
    
    sources = []
    if os.path.exists(rag_sources_dir):
        for item in os.listdir(rag_sources_dir):
            item_path = os.path.join(rag_sources_dir, item)
            if os.path.isdir(item_path):
                sources.append(item)
    
    if not sources:
        print("✗ No RAG sources found")
        return None
    
    test_source = sources[0]
    
    try:
        response = rag_service.query_with_rag(test_query, "ollama", test_source)
        print(f"✓ RAG query successful")
        print(f"Response: {response[:300]}...")
        return response
    except Exception as e:
        print(f"✗ RAG query failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_direct_llm():
    print("\nTesting direct LLM query (without RAG)...")
    try:
        from services.ai import AI
        ai = AI()
        response = ai.ask_ai("Say hello", provider="local", model="qwen2.5-coder:7b")
        print(f"✓ Direct LLM query successful")
        print(f"Response: {response}")
        return response
    except Exception as e:
        print(f"✗ Direct LLM query failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("=" * 50)
    print("RAG System Test")
    print("=" * 50)
    
    rag_service = test_rag_initialization()
    
    if rag_service:
        documents = test_file_conversion(rag_service)
        
        if documents:
            retriever = test_retriever_creation(rag_service, documents)
        
        test_rag_query(rag_service)
    
    test_direct_llm()
    
    print("\n" + "=" * 50)
    print("Test completed")
    print("=" * 50)