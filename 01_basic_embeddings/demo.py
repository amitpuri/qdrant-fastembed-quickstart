"""
Basic Text Embeddings Demo
Demonstrates standard dense text embeddings using FastEmbed with Qdrant integration.

Based on: https://qdrant.tech/documentation/fastembed/fastembed-quickstart/
"""

import os
from typing import List
import numpy as np
from fastembed import TextEmbedding

# Import Qdrant client
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


def run_basic_embeddings_demo():
    """Demonstrate basic text embeddings with Qdrant integration."""
    print("\n🔹 Basic Text Embeddings Demo")
    print("-" * 40)
    
    # Load Qdrant configuration from environment variables
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY", None)
    
    print(f"🔧 Qdrant Configuration: {qdrant_url}")
    if qdrant_api_key:
        print(f"🔑 API Key: {'*' * (len(qdrant_api_key) - 4) + qdrant_api_key[-4:]}")
    print()
    
    if not QDRANT_AVAILABLE:
        print("❌ Qdrant client not available. Please install: pip install qdrant-client[fastembed]")
        print("💡 This demo shows the concepts without actual Qdrant integration.")
        print()
    
    # Sample documents
    documents: List[str] = [
        "FastEmbed is lighter than Transformers & Sentence-Transformers.",
        "FastEmbed is supported by and maintained by Qdrant.",
        "Vector Graphics in Modern Web Design",
        "The Art of Search and Self-Discovery",
        "Efficient Vector Search Algorithms for Large Datasets"
    ]
    
    print("📄 Sample documents:")
    for i, doc in enumerate(documents, 1):
        print(f"   {i}. {doc}")
    print()
    
    try:
        # Load model
        print("Loading FastEmbed model (BAAI/bge-small-en-v1.5)...")
        embedding_model = TextEmbedding()
        print("✅ Model loaded successfully!")
        
        # Generate embeddings
        print("\nGenerating embeddings...")
        embeddings_generator = embedding_model.embed(documents)
        embeddings_list = list(embeddings_generator)
        
        print(f"📊 Generated {len(embeddings_list)} embeddings")
        print(f"📏 Vector dimensions: {len(embeddings_list[0])}")
        
        # Show similarity between documents
        print("\n🔗 Cosine similarities between documents:")
        for i in range(len(embeddings_list)):
            for j in range(i + 1, len(embeddings_list)):
                similarity = np.dot(embeddings_list[i], embeddings_list[j])
                print(f"   Doc {i+1} ↔ Doc {j+1}: {similarity:.4f}")
        
        # Show sample embedding
        print(f"\n📋 Sample embedding (first 10 dimensions):")
        print(f"   {embeddings_list[0][:10]}")
        
        # Show embedding statistics
        print(f"\n📈 Embedding statistics:")
        all_embeddings = np.array(embeddings_list)
        print(f"   Mean: {np.mean(all_embeddings):.6f}")
        print(f"   Std:  {np.std(all_embeddings):.6f}")
        print(f"   Min:  {np.min(all_embeddings):.6f}")
        print(f"   Max:  {np.max(all_embeddings):.6f}")
        
        # Qdrant Integration
        if QDRANT_AVAILABLE:
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
                
                # Create collection for this demo
                collection_name = "basic_embeddings_demo"
                print(f"   Creating collection: {collection_name}")
                
                try:
                    client.create_collection(
                        collection_name=collection_name,
                        vectors_config=models.VectorParams(
                            size=len(embeddings_list[0]),
                            distance=models.Distance.COSINE
                        )
                    )
                    print("   ✅ Collection created successfully!")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        print("   ℹ️  Collection already exists, using existing one")
                    else:
                        raise e
                
                # Upload documents with embeddings
                print("   Uploading documents with embeddings...")
                points = []
                for i, (doc, embedding) in enumerate(zip(documents, embeddings_list)):
                    points.append(
                        models.PointStruct(
                            id=i + 1,
                            payload={"text": doc, "doc_id": i + 1},
                            vector=embedding.tolist()
                        )
                    )
                
                client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                print(f"   ✅ Uploaded {len(points)} documents to Qdrant!")
                
                # Perform a search query
                query_text = "FastEmbed and Qdrant integration"
                print(f"\n🔍 Searching for: '{query_text}'")
                
                # Generate embedding for query
                query_embedding = list(embedding_model.embed([query_text]))[0]
                
                # Search in Qdrant
                search_results = client.query_points(
                    collection_name=collection_name,
                    query=query_embedding.tolist(),
                    limit=3
                ).points
                
                print("   📊 Search results from Qdrant:")
                for i, result in enumerate(search_results, 1):
                    print(f"      {i}. Score: {result.score:.4f} - {result.payload['text']}")
                
                # Clean up - delete the demo collection
                print(f"\n🧹 Cleaning up demo collection...")
                client.delete_collection(collection_name)
                print("   ✅ Demo collection deleted")
                
            except Exception as e:
                print(f"   ❌ Qdrant integration error: {e}")
                print("   💡 Make sure Qdrant is running at the configured URL")
        
        print("\n✨ Key features of dense embeddings:")
        print("   • General purpose semantic understanding")
        print("   • Good for similarity search and clustering")
        print("   • Works well with vector databases like Qdrant")
        print("   • Suitable for most NLP tasks")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    run_basic_embeddings_demo()
