"""
Reranking Demo
Demonstrates reranking capabilities to improve search results by re-scoring initial candidates.

Based on: https://qdrant.tech/documentation/fastembed/fastembed-rerankers/
"""

import os
from typing import List, Tuple
import numpy as np

# Import FastEmbed reranker
try:
    from fastembed import Reranker
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False

# Import Qdrant client
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


def run_reranking_demo():
    """Demonstrate reranking capabilities."""
    print("\n🔹 Reranking Demo")
    print("-" * 40)
    
    # Load Qdrant configuration from environment variables
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY", None)
    
    print(f"🔧 Qdrant Configuration: {qdrant_url}")
    if qdrant_api_key:
        print(f"🔑 API Key: {'*' * (len(qdrant_api_key) - 4) + qdrant_api_key[-4:]}")
    print()
    
    if not RERANKER_AVAILABLE:
        print("❌ FastEmbed Reranker not available.")
        print("💡 This demo shows the concepts without actual reranking.")
        print()
    
    try:
        print("📝 Reranking improves search results by re-scoring initial candidates.")
        print("🎯 It's typically used as a second stage after initial retrieval.")
        print()
        
        # Sample search results that would be reranked
        initial_results = [
            ("FastEmbed is lighter than Transformers & Sentence-Transformers.", 0.85),
            ("FastEmbed is supported by and maintained by Qdrant.", 0.78),
            ("Vector Graphics in Modern Web Design", 0.65),
            ("The Art of Search and Self-Discovery", 0.52),
            ("Efficient Vector Search Algorithms for Large Datasets", 0.48),
            ("Searching the Soul: A Journey Through Mindfulness", 0.42),
            ("Vector-Based Animations for User Interface Design", 0.38),
            ("Search Engines: A Technical and Social Overview", 0.35)
        ]
        
        query = "FastEmbed and Qdrant integration"
        print(f"🔍 Query: '{query}'")
        print()
        
        print("📊 Initial retrieval results (before reranking):")
        for i, (doc, score) in enumerate(initial_results, 1):
            relevance_indicator = "🟢" if "FastEmbed" in doc or "Qdrant" in doc else "🟡" if "vector" in doc.lower() or "search" in doc.lower() else "🔴"
            print(f"   {i}. [{score:.2f}] {relevance_indicator} {doc}")
        print()
        
        print("🔄 After reranking (expected improvement):")
        # Simulate reranking - documents more relevant to query get higher scores
        reranked_results = [
            ("FastEmbed is supported by and maintained by Qdrant.", 0.95),  # Perfect match
            ("FastEmbed is lighter than Transformers & Sentence-Transformers.", 0.88),  # FastEmbed mention
            ("Efficient Vector Search Algorithms for Large Datasets", 0.72),  # Search algorithms
            ("Vector Graphics in Modern Web Design", 0.45),  # Less relevant
            ("The Art of Search and Self-Discovery", 0.38),  # Less relevant
            ("Searching the Soul: A Journey Through Mindfulness", 0.32),  # Less relevant
            ("Vector-Based Animations for User Interface Design", 0.28),  # Less relevant
            ("Search Engines: A Technical and Social Overview", 0.25)  # Less relevant
        ]
        
        for i, (doc, score) in enumerate(reranked_results, 1):
            relevance_indicator = "🟢" if "FastEmbed" in doc or "Qdrant" in doc else "🟡" if "vector" in doc.lower() or "search" in doc.lower() else "🔴"
            improvement = "📈" if score > initial_results[i-1][1] else "📉" if score < initial_results[i-1][1] else "➡️"
            print(f"   {i}. [{score:.2f}] {relevance_indicator} {improvement} {doc}")
        print()
        
        # Show improvement metrics
        print("📈 Reranking improvement analysis:")
        perfect_matches = [doc for doc, score in reranked_results if "FastEmbed" in doc or "Qdrant" in doc]
        print(f"   • Perfect matches moved to top: {len(perfect_matches)}")
        print(f"   • Most relevant result score: {reranked_results[0][1]:.2f} (was {initial_results[0][1]:.2f})")
        print(f"   • Score improvement: +{reranked_results[0][1] - initial_results[0][1]:.2f}")
        print()
        
        print("✨ Reranking benefits:")
        print("   • More relevant results move to top")
        print("   • Better precision for specific queries")
        print("   • Can combine multiple signals (semantic + lexical)")
        print("   • Improves user experience significantly")
        print("   • Reduces false positives in top results")
        print("   • Enables fine-tuning for specific domains")
        print()
        
        print("🎯 When to use reranking:")
        print("   • As a second stage after initial retrieval")
        print("   • When precision is more important than recall")
        print("   • For domain-specific search applications")
        print("   • When you have limited result slots (e.g., top 5)")
        print("   • For production search systems")
        print("   • When combining multiple retrieval methods")
        print()
        
        # Show reranking pipeline
        print("🔄 Typical reranking pipeline:")
        print("   1. Initial retrieval (dense/sparse/hybrid)")
        print("   2. Get top-K candidates (e.g., top 100)")
        print("   3. Rerank candidates with specialized model")
        print("   4. Return top-N final results (e.g., top 10)")
        print()
        
        # Show different reranking approaches
        print("🔧 Reranking approaches:")
        print("   • Cross-encoder models: Query-document pairs")
        print("   • Point-wise: Score each document independently")
        print("   • Pair-wise: Compare document pairs")
        print("   • List-wise: Optimize entire result list")
        print()
        
        if RERANKER_AVAILABLE:
            print("\n🔧 Qdrant Integration Demo:")
            try:
                # Connect to Qdrant
                print(f"   Connecting to Qdrant at {qdrant_url}...")
                client = QdrantClient(
                    url=qdrant_url,
                    api_key=qdrant_api_key
                )
                
                # Check if Qdrant is accessible
                collections = client.get_collections()
                print("   ✅ Connected to Qdrant successfully!")
                
                # Create collection for reranking demo
                collection_name = "reranking_demo"
                print(f"   Creating collection: {collection_name}")
                
                try:
                    client.create_collection(
                        collection_name=collection_name,
                        vectors_config={
                            "dense": models.VectorParams(
                                size=384,  # BGE model size
                                distance=models.Distance.COSINE
                            )
                        }
                    )
                    print("   ✅ Collection created successfully!")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        print("   ℹ️  Collection already exists, using existing one")
                    else:
                        raise e
                
                # Upload documents with dense embeddings
                print("   Uploading documents with dense embeddings...")
                from fastembed import TextEmbedding
                embedding_model = TextEmbedding()
                
                points = []
                for i, (doc, score) in enumerate(initial_results):
                    # Generate embedding for the document
                    doc_embedding = list(embedding_model.embed([doc]))[0]
                    points.append(
                        models.PointStruct(
                            id=i + 1,
                            payload={"text": doc, "original_score": score, "doc_id": i + 1},
                            vector={"dense": doc_embedding.tolist()}
                        )
                    )
                
                client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                print(f"   ✅ Uploaded {len(points)} documents!")
                
                # Perform initial search
                print(f"\n🔍 Initial search for: '{query}'")
                query_embedding = list(embedding_model.embed([query]))[0]
                
                initial_search_results = client.query_points(
                    collection_name=collection_name,
                    query=query_embedding.tolist(),
                    limit=5
                ).points
                
                print("   📊 Initial search results:")
                for i, result in enumerate(initial_search_results, 1):
                    print(f"      {i}. Score: {result.score:.4f} - {result.payload['text']}")
                
                # Apply reranking
                print(f"\n🔄 Applying reranking...")
                reranker = Reranker()
                
                # Prepare documents for reranking
                documents_to_rerank = [result.payload['text'] for result in initial_search_results]
                
                # Rerank the documents
                rerank_scores = reranker.rank(query, documents_to_rerank)
                
                # Combine results with reranking scores
                reranked_results = []
                for i, (result, rerank_score) in enumerate(zip(initial_search_results, rerank_scores)):
                    reranked_results.append({
                        'text': result.payload['text'],
                        'original_score': result.score,
                        'rerank_score': rerank_score,
                        'doc_id': result.payload['doc_id']
                    })
                
                # Sort by reranking scores
                reranked_results.sort(key=lambda x: x['rerank_score'], reverse=True)
                
                print("   📊 Reranked results:")
                for i, result in enumerate(reranked_results, 1):
                    improvement = "📈" if result['rerank_score'] > result['original_score'] else "📉"
                    print(f"      {i}. Rerank: {result['rerank_score']:.4f} {improvement} - {result['text']}")
                
                # Clean up - delete the demo collection
                print(f"\n🧹 Cleaning up demo collection...")
                client.delete_collection(collection_name)
                print("   ✅ Demo collection deleted")
                
            except Exception as e:
                print(f"   ❌ Qdrant integration error: {e}")
                print("   💡 Make sure Qdrant is running at the configured URL")
        else:
            print("💡 To use FastEmbed Reranker:")
            print("   1. Install: pip install fastembed")
            print("   2. Import: from fastembed import Reranker")
            print("   3. Initialize reranker model")
            print("   4. Rerank query-document pairs")
            print("   5. Sort by reranking scores")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    run_reranking_demo()
