"""
Qdrant Integration Demo
Demonstrates Qdrant vector database integration with FastEmbed.

Based on: https://qdrant.tech/documentation/fastembed/
"""

import os
from typing import List, Dict, Any
import numpy as np

# Import Qdrant client
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

# Import FastEmbed
try:
    from fastembed import TextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False


def run_qdrant_integration_demo():
    """Demonstrate Qdrant integration."""
    print("\n🔹 Qdrant Integration Demo")
    print("-" * 40)
    
    # Load Qdrant configuration from environment variables
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY", None)
    qdrant_timeout = int(os.getenv("QDRANT_TIMEOUT", "60"))
    qdrant_prefer_grpc = os.getenv("QDRANT_PREFER_GRPC", "false").lower() == "true"
    
    print(f"🔧 Qdrant Configuration:")
    print(f"   URL: {qdrant_url}")
    if qdrant_api_key:
        print(f"   API Key: {'*' * (len(qdrant_api_key) - 4) + qdrant_api_key[-4:]}")
    print(f"   Timeout: {qdrant_timeout}s")
    print(f"   Prefer gRPC: {qdrant_prefer_grpc}")
    print()
    
    if not QDRANT_AVAILABLE:
        print("❌ Qdrant client not available. Please install: pip install qdrant-client[fastembed]")
        print("💡 This demo shows the concepts without actual Qdrant integration.")
        print()
    
    if not FASTEMBED_AVAILABLE:
        print("❌ FastEmbed not available. Please install: pip install fastembed")
        print()
    
    try:
        print("📝 Qdrant is a vector database that works seamlessly with FastEmbed.")
        print("🎯 It provides efficient storage and retrieval of vector embeddings.")
        print()
        
        # Sample documents
        documents = [
            "FastEmbed is lighter than Transformers & Sentence-Transformers.",
            "FastEmbed is supported by and maintained by Qdrant.",
            "Vector Graphics in Modern Web Design",
            "The Art of Search and Self-Discovery",
            "Efficient Vector Search Algorithms for Large Datasets",
            "Searching the Soul: A Journey Through Mindfulness",
            "Vector-Based Animations for User Interface Design",
            "Search Engines: A Technical and Social Overview",
            "The Rise of Vector Databases in AI Systems",
            "Search Patterns in Human Behavior"
        ]
        
        print("📄 Sample documents for Qdrant integration:")
        for i, doc in enumerate(documents[:5], 1):
            print(f"   {i}. {doc}")
        print("   ... and 5 more documents")
        print()
        
        print("🔧 Qdrant + FastEmbed integration features:")
        print("   • Automatic model downloading and inference")
        print("   • Efficient vector storage and indexing")
        print("   • Support for dense, sparse, and hybrid search")
        print("   • Built-in similarity search algorithms")
        print("   • Filtering and payload support")
        print("   • Horizontal scaling and clustering")
        print("   • Real-time updates and synchronization")
        print()
        
        print("📊 Supported vector types:")
        print("   • Dense vectors (from TextEmbedding)")
        print("   • Sparse vectors (from SPLADE, miniCOIL)")
        print("   • Multi-vectors (from ColBERT)")
        print("   • Hybrid combinations")
        print("   • Custom vector dimensions")
        print()
        
        print("🚀 Typical Qdrant + FastEmbed workflow:")
        print("   1. Create collection with vector configuration")
        print("   2. Upload documents with FastEmbed inference")
        print("   3. Query with semantic search")
        print("   4. Retrieve relevant results with scores")
        print("   5. Apply filters and aggregations")
        print()
        
        # Show collection configuration example
        print("🔧 Collection configuration example:")
        print("```python")
        print("client.create_collection(")
        print("    collection_name='documents',")
        print("    vectors_config={")
        print("        'dense': models.VectorParams(size=384, distance=models.Distance.COSINE),")
        print("        'sparse': models.SparseVectorParams()")
        print("    }")
        print(")")
        print("```")
        print()
        
        # Show document upload example
        print("📤 Document upload example:")
        print("```python")
        print("client.upsert(")
        print("    collection_name='documents',")
        print("    points=[")
        print("        models.PointStruct(")
        print("            id=1,")
        print("            payload={'text': 'FastEmbed is lightweight'},")
        print("            vector={")
        print("                'dense': models.Document(")
        print("                    text='FastEmbed is lightweight',")
        print("                    model='BAAI/bge-small-en-v1.5'")
        print("                )")
        print("            }")
        print("        )")
        print("    ]")
        print(")")
        print("```")
        print()
        
        # Show query example
        print("🔍 Query example:")
        print("```python")
        print("results = client.query_points(")
        print("    collection_name='documents',")
        print("    query=models.Document(")
        print("        text='vector search technology',")
        print("        model='BAAI/bge-small-en-v1.5'")
        print("    ),")
        print("    using='dense',")
        print("    limit=5")
        print(")")
        print("```")
        print()
        
        print("✨ Qdrant advantages:")
        print("   • High-performance vector search")
        print("   • Multiple distance metrics (cosine, dot, euclidean)")
        print("   • Advanced filtering capabilities")
        print("   • Payload storage and retrieval")
        print("   • Horizontal scaling")
        print("   • Real-time updates")
        print("   • REST and gRPC APIs")
        print("   • Web UI for management")
        print()
        
        print("🎯 Use cases:")
        print("   • Semantic search applications")
        print("   • Recommendation systems")
        print("   • Question-answering systems")
        print("   • Document retrieval")
        print("   • Image and multimedia search")
        print("   • Chatbots and conversational AI")
        print("   • Knowledge graphs and RAG")
        print()
        
        if QDRANT_AVAILABLE and FASTEMBED_AVAILABLE:
            print("\n🔧 Live Qdrant Integration Demo:")
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
                print(f"   📊 Found {len(collections.collections)} existing collections")
                
                # Create a comprehensive demo collection
                collection_name = "fastembed_demo_integration"
                print(f"\n   Creating comprehensive demo collection: {collection_name}")
                
                try:
                    client.create_collection(
                        collection_name=collection_name,
                        vectors_config={
                            "dense": models.VectorParams(
                                size=384,  # BGE model size
                                distance=models.Distance.COSINE
                            )
                        },
                        sparse_vectors_config={
                            "sparse": models.SparseVectorParams()
                        }
                    )
                    print("   ✅ Collection created with dense and sparse vector support!")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        print("   ℹ️  Collection already exists, using existing one")
                    else:
                        raise e
                
                # Upload documents with FastEmbed inference
                print("\n   📤 Uploading documents with FastEmbed inference...")
                embedding_model = TextEmbedding()
                
                points = []
                for i, doc in enumerate(documents):
                    # Generate dense embedding
                    doc_embedding = list(embedding_model.embed([doc]))[0]
                    
                    points.append(
                        models.PointStruct(
                            id=i + 1,
                            payload={
                                "text": doc, 
                                "doc_id": i + 1,
                                "category": "demo",
                                "timestamp": "2024-01-01"
                            },
                            vector={
                                "dense": doc_embedding.tolist()
                            }
                        )
                    )
                
                client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                print(f"   ✅ Uploaded {len(points)} documents with FastEmbed inference!")
                
                # Demonstrate various search capabilities
                print("\n🔍 Search Demonstrations:")
                
                # 1. Basic similarity search
                print("\n   1. Basic Similarity Search:")
                query_text = "FastEmbed and Qdrant integration"
                query_embedding = list(embedding_model.embed([query_text]))[0]
                
                search_results = client.query_points(
                    collection_name=collection_name,
                    query=query_embedding.tolist(),
                    using="dense",
                    limit=3
                ).points
                
                print(f"      Query: '{query_text}'")
                for i, result in enumerate(search_results, 1):
                    print(f"      {i}. Score: {result.score:.4f} - {result.payload['text']}")
                
                # 2. Search with filters
                print("\n   2. Search with Filters:")
                filtered_results = client.query_points(
                    collection_name=collection_name,
                    query=query_embedding.tolist(),
                    using="dense",
                    query_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="category",
                                match=models.MatchValue(value="demo")
                            )
                        ]
                    ),
                    limit=3
                ).points
                
                print(f"      Query with category filter:")
                for i, result in enumerate(filtered_results, 1):
                    print(f"      {i}. Score: {result.score:.4f} - {result.payload['text']}")
                
                # 3. Batch search
                print("\n   3. Batch Search:")
                batch_queries = [
                    "vector search technology",
                    "machine learning algorithms",
                    "web design graphics"
                ]
                
                batch_embeddings = [list(embedding_model.embed([q]))[0] for q in batch_queries]
                
                batch_results = client.query_batch_points(
                    collection_name=collection_name,
                    requests=[
                        models.QueryRequest(
                            query=emb.tolist(),
                            using="dense",
                            limit=2
                        ) for emb in batch_embeddings
                    ]
                )
                
                for i, (query, results) in enumerate(zip(batch_queries, batch_results)):
                    print(f"      Query {i+1}: '{query}'")
                    if results and hasattr(results, 'points'):
                        for j, result in enumerate(results.points, 1):
                            print(f"         {j}. Score: {result.score:.4f} - {result.payload['text']}")
                    else:
                        print("         No results found")
                
                # 4. Collection info
                print("\n   4. Collection Information:")
                collection_info = client.get_collection(collection_name)
                print(f"      Points count: {collection_info.points_count}")
                print(f"      Vector size: {collection_info.config.params.vectors['dense'].size}")
                print(f"      Distance metric: {collection_info.config.params.vectors['dense'].distance}")
                
                # Clean up - delete the demo collection
                print(f"\n🧹 Cleaning up demo collection...")
                client.delete_collection(collection_name)
                print("   ✅ Demo collection deleted")
                
                print("\n✨ Qdrant + FastEmbed Integration Features Demonstrated:")
                print("   • Automatic model downloading and inference")
                print("   • Efficient vector storage and indexing")
                print("   • Support for dense and sparse vectors")
                print("   • Advanced filtering and search capabilities")
                print("   • Batch operations for efficiency")
                print("   • Real-time updates and synchronization")
                
            except Exception as e:
                print(f"   ❌ Qdrant integration error: {e}")
                print("   💡 Make sure Qdrant is running at the configured URL")
                print("   💡 Check the Qdrant Web UI at http://localhost:6333/dashboard")
        else:
            print("💡 To get started:")
            print("   1. Install: pip install qdrant-client[fastembed]")
            print("   2. Set up Qdrant instance:")
            print("      • Local: docker run -p 6333:6333 qdrant/qdrant")
            print("      • Cloud: https://cloud.qdrant.io/")
            print("   3. Configure .env file with your Qdrant URL")
            print("   4. Run this demo with actual Qdrant connection")
            print("   5. Explore the Qdrant Web UI at http://localhost:6333/dashboard")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    run_qdrant_integration_demo()
