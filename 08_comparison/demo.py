"""
Comparison Demo
Compares all FastEmbed methods and provides guidance on when to use each approach.

Based on comprehensive FastEmbed documentation from Qdrant.
"""

import os
from typing import List, Dict, Tuple
import numpy as np


def run_comparison_demo():
    """Compare all embedding methods."""
    print("\n🔹 Comparison of All Methods")
    print("-" * 40)
    
    # Load Qdrant configuration from environment variables
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY", None)
    
    print(f"🔧 Qdrant Configuration: {qdrant_url}")
    if qdrant_api_key:
        print(f"🔑 API Key: {'*' * (len(qdrant_api_key) - 4) + qdrant_api_key[-4:]}")
    print()
    
    # Check Qdrant connectivity
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        collections = client.get_collections()
        print("✅ Qdrant connection verified - all methods can be tested")
        print(f"📊 Found {len(collections.collections)} existing collections")
    except Exception as e:
        print(f"⚠️  Qdrant connection issue: {e}")
        print("💡 Some demos require Qdrant to be running for full functionality")
    print()
    
    try:
        query = "vector search technology"
        print(f"🔍 Query: '{query}'")
        print()
        
        # Method comparison table
        print("📊 Method Comparison:")
        print("┌" + "─" * 80 + "┐")
        print("│ Method        │ Type      │ Dimensions │ Use Case                    │")
        print("├" + "─" * 80 + "┤")
        print("│ Dense (BGE)   │ Dense     │ 384        │ General semantic search     │")
        print("│ miniCOIL      │ Sparse    │ Variable   │ Keyword + semantic hybrid   │")
        print("│ SPLADE        │ Sparse    │ Variable   │ Lexical + semantic hybrid   │")
        print("│ ColBERT       │ Multi     │ 384×N      │ Fine-grained token matching │")
        print("│ Reranking     │ Post-proc │ N/A        │ Improve initial results     │")
        print("└" + "─" * 80 + "┘")
        print()
        
        # Performance characteristics
        print("⚡ Performance Characteristics:")
        print("┌" + "─" * 80 + "┐")
        print("│ Method        │ Speed     │ Memory     │ Storage     │ Accuracy    │")
        print("├" + "─" * 80 + "┤")
        print("│ Dense (BGE)   │ Fast      │ Low        │ Medium      │ High        │")
        print("│ miniCOIL      │ Medium    │ Medium     │ Low         │ High        │")
        print("│ SPLADE        │ Medium    │ Medium     │ Low         │ High        │")
        print("│ ColBERT       │ Slow      │ High       │ High        │ Very High   │")
        print("│ Reranking     │ Slow      │ High       │ N/A         │ Very High   │")
        print("└" + "─" * 80 + "┘")
        print()
        
        # When to use each method
        print("🎯 When to use each method:")
        print()
        
        print("🔸 Dense Embeddings (BGE):")
        print("   ✅ General purpose semantic search")
        print("   ✅ Similarity search and clustering")
        print("   ✅ Most NLP tasks")
        print("   ✅ When you need good semantic understanding")
        print("   ❌ When exact keyword matches are required")
        print("   ❌ When you need term-level control")
        print()
        
        print("🔸 miniCOIL:")
        print("   ✅ When exact keyword matches matter")
        print("   ✅ But context and meaning are important")
        print("   ✅ Domain-specific search (medical, legal)")
        print("   ✅ Hybrid search scenarios")
        print("   ❌ Pure semantic similarity tasks")
        print("   ❌ When you don't need keyword matching")
        print()
        
        print("🔸 SPLADE:")
        print("   ✅ When you need both lexical and semantic matching")
        print("   ✅ Domain-specific vocabularies")
        print("   ✅ Technical documentation search")
        print("   ✅ Term expansion capabilities")
        print("   ❌ When pure semantic understanding is sufficient")
        print("   ❌ Simple similarity tasks")
        print()
        
        print("🔸 ColBERT:")
        print("   ✅ Complex, multi-part queries")
        print("   ✅ When word order and position matter")
        print("   ✅ Fine-grained relevance requirements")
        print("   ✅ Question-answering systems")
        print("   ❌ Simple similarity search")
        print("   ❌ When storage/memory is limited")
        print()
        
        print("🔸 Reranking:")
        print("   ✅ As a second stage after initial retrieval")
        print("   ✅ When precision is more important than recall")
        print("   ✅ Domain-specific applications")
        print("   ✅ Limited result slots (top 5-10)")
        print("   ❌ As the only retrieval method")
        print("   ❌ When speed is critical")
        print()
        
        # Best practices
        print("💡 Best practices:")
        print()
        print("🚀 Getting started:")
        print("   1. Start with dense embeddings for most use cases")
        print("   2. Add sparse methods for keyword-heavy domains")
        print("   3. Use ColBERT for complex, multi-part queries")
        print("   4. Always consider reranking for production systems")
        print("   5. Combine methods for hybrid search when possible")
        print()
        
        print("🔧 Production recommendations:")
        print("   • Use dense embeddings as your baseline")
        print("   • Add miniCOIL for domain-specific search")
        print("   • Use SPLADE for technical documentation")
        print("   • Apply ColBERT for complex queries")
        print("   • Always use reranking for final results")
        print("   • Monitor performance and adjust accordingly")
        print()
        
        # Hybrid approaches
        print("🔄 Hybrid search approaches:")
        print()
        print("🔸 Dense + Sparse:")
        print("   • Combine dense embeddings with miniCOIL or SPLADE")
        print("   • Weight the results (e.g., 70% dense, 30% sparse)")
        print("   • Use for comprehensive search coverage")
        print()
        
        print("🔸 Multi-stage retrieval:")
        print("   1. Initial retrieval with dense embeddings")
        print("   2. Expand with sparse methods")
        print("   3. Rerank final candidates")
        print("   • Best of all worlds approach")
        print()
        
        print("🔸 Query-dependent selection:")
        print("   • Use dense for semantic queries")
        print("   • Use sparse for keyword queries")
        print("   • Use ColBERT for complex queries")
        print("   • Adaptive approach based on query type")
        print()
        
        # Implementation considerations
        print("⚙️ Implementation considerations:")
        print()
        print("📊 Resource requirements:")
        print("   • Dense: Low memory, medium storage")
        print("   • Sparse: Medium memory, low storage")
        print("   • ColBERT: High memory, high storage")
        print("   • Reranking: High memory, no storage")
        print()
        
        print("🔧 Integration complexity:")
        print("   • Dense: Simple integration")
        print("   • Sparse: Medium complexity")
        print("   • ColBERT: Complex integration")
        print("   • Reranking: Medium complexity")
        print()
        
        print("📈 Scalability:")
        print("   • Dense: Excellent scalability")
        print("   • Sparse: Good scalability")
        print("   • ColBERT: Limited scalability")
        print("   • Reranking: Limited scalability")
        print()
        
        # Decision tree
        print("🌳 Decision tree for method selection:")
        print()
        print("1. Do you need exact keyword matches?")
        print("   Yes → Consider miniCOIL or SPLADE")
        print("   No → Use dense embeddings")
        print()
        print("2. Do you have complex, multi-part queries?")
        print("   Yes → Consider ColBERT")
        print("   No → Continue with current method")
        print()
        print("3. Do you need high precision in top results?")
        print("   Yes → Add reranking")
        print("   No → Current method is sufficient")
        print()
        print("4. Do you have domain-specific requirements?")
        print("   Yes → Consider hybrid approach")
        print("   No → Single method is sufficient")
        print()
        
        print("🎯 Final recommendations:")
        print("   • Start simple with dense embeddings")
        print("   • Add complexity only when needed")
        print("   • Measure and optimize based on your data")
        print("   • Consider hybrid approaches for best results")
        print("   • Always test with your specific use case")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    run_comparison_demo()
