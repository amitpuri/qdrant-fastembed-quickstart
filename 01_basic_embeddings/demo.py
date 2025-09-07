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
    
    print("📚 What are Dense Text Embeddings?")
    print("   • Dense embeddings convert text into high-dimensional vectors")
    print("   • Each dimension captures some semantic meaning")
    print("   • Similar texts have similar vector representations")
    print("   • Perfect for semantic search and similarity matching")
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
    
    print("📄 Sample documents we'll work with:")
    for i, doc in enumerate(documents, 1):
        print(f"   {i}. {doc}")
    print()
    
    print("🎯 What we'll demonstrate:")
    print("   1. Convert text to dense embeddings")
    print("   2. Calculate similarities between documents")
    print("   3. Store embeddings in Qdrant vector database")
    print("   4. Perform semantic search")
    print("   5. Show how similar texts cluster together")
    print()
    
    input("Press Enter to start the embedding process...")
    
    try:
        # Load model
        print("🔄 Step 1: Loading FastEmbed model...")
        print("   Model: BAAI/bge-small-en-v1.5 (optimized for speed and quality)")
        print("   This model converts text into 384-dimensional vectors")
        embedding_model = TextEmbedding()
        print("   ✅ Model loaded successfully!")
        print()
        
        # Generate embeddings
        print("🔄 Step 2: Converting text to embeddings...")
        print("   Processing each document through the neural network...")
        embeddings_generator = embedding_model.embed(documents)
        embeddings_list = list(embeddings_generator)
        
        print(f"   ✅ Generated {len(embeddings_list)} embeddings")
        print(f"   📏 Each vector has {len(embeddings_list[0])} dimensions")
        print(f"   💾 Total storage: {len(embeddings_list) * len(embeddings_list[0]) * 4} bytes")
        print()
        
        # Show similarity between documents
        print("🔄 Step 3: Analyzing document similarities...")
        print("   Calculating cosine similarity between all document pairs...")
        print("   (Higher values = more similar content)")
        print()
        
        similarities = []
        for i in range(len(embeddings_list)):
            for j in range(i + 1, len(embeddings_list)):
                similarity = np.dot(embeddings_list[i], embeddings_list[j])
                similarities.append((i, j, similarity))
                print(f"   📊 Doc {i+1} ↔ Doc {j+1}: {similarity:.4f}")
                print(f"      '{documents[i][:50]}...' ↔ '{documents[j][:50]}...'")
        
        # Find most and least similar pairs
        most_similar = max(similarities, key=lambda x: x[2])
        least_similar = min(similarities, key=lambda x: x[2])
        
        print(f"\n   🎯 Most similar pair: Doc {most_similar[0]+1} ↔ Doc {most_similar[1]+1} ({most_similar[2]:.4f})")
        print(f"   🎯 Least similar pair: Doc {least_similar[0]+1} ↔ Doc {least_similar[1]+1} ({least_similar[2]:.4f})")
        print()
        
        input("Press Enter to see embedding details...")
        
        # Show sample embedding
        print("🔄 Step 4: Examining embedding structure...")
        print(f"   📋 Sample embedding (first 10 dimensions):")
        print(f"   {embeddings_list[0][:10]}")
        print(f"   📋 Sample embedding (last 10 dimensions):")
        print(f"   {embeddings_list[0][-10:]}")
        
        # Show embedding statistics
        print(f"\n   📈 Embedding statistics:")
        all_embeddings = np.array(embeddings_list)
        print(f"      Mean: {np.mean(all_embeddings):.6f}")
        print(f"      Std:  {np.std(all_embeddings):.6f}")
        print(f"      Min:  {np.min(all_embeddings):.6f}")
        print(f"      Max:  {np.max(all_embeddings):.6f}")
        print()
        
        # Qdrant Integration
        if QDRANT_AVAILABLE:
            print("🔄 Step 5: Qdrant Vector Database Integration...")
            input("Press Enter to connect to Qdrant and store embeddings...")
            
            try:
                # Connect to Qdrant
                print(f"   🔌 Connecting to Qdrant at {qdrant_url}...")
                client = QdrantClient(
                    url=qdrant_url,
                    api_key=qdrant_api_key
                )
                
                # Check if Qdrant is accessible
                collections = client.get_collections()
                print("   ✅ Connected to Qdrant successfully!")
                print(f"   📊 Found {len(collections.collections)} existing collections")
                
                # Create collection for this demo
                collection_name = "fastembed_demo_basic_embeddings"
                print(f"   🗂️  Creating collection: {collection_name}")
                print(f"   📏 Vector size: {len(embeddings_list[0])} dimensions")
                print(f"   📐 Distance metric: Cosine similarity")
                
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
                print(f"\n   📤 Uploading {len(documents)} documents with embeddings...")
                print("   Each document becomes a point with:")
                print("   • ID: Unique identifier")
                print("   • Vector: 384-dimensional embedding")
                print("   • Payload: Original text and metadata")
                
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
                print(f"   ✅ Successfully uploaded {len(points)} documents to Qdrant!")
                print(f"   💾 Total vectors stored: {len(points)}")
                print(f"   🗄️  Collection ready for semantic search!")
                
                # Perform a search query
                print(f"\n🔍 Step 6: Performing Semantic Search...")
                query_text = "FastEmbed and Qdrant integration"
                print(f"   🔎 Query: '{query_text}'")
                print(f"   🧠 Converting query to embedding...")
                
                # Generate embedding for query
                query_embedding = list(embedding_model.embed([query_text]))[0]
                print(f"   ✅ Query embedding generated ({len(query_embedding)} dimensions)")
                
                print(f"   🔍 Searching for most similar documents...")
                print(f"   📊 Using cosine similarity to find best matches...")
                
                # Search in Qdrant
                search_results = client.query_points(
                    collection_name=collection_name,
                    query=query_embedding.tolist(),
                    limit=3
                ).points
                
                print(f"\n   🎯 Top {len(search_results)} most relevant results:")
                for i, result in enumerate(search_results, 1):
                    relevance = "🟢" if result.score > 0.8 else "🟡" if result.score > 0.6 else "🔴"
                    print(f"      {i}. {relevance} Score: {result.score:.4f}")
                    print(f"         📄 {result.payload['text']}")
                    print()
                
                # Note: Collection will be cleaned up by main menu option 9
                print(f"\n💡 Demo collection '{collection_name}' created successfully!")
                print(f"   🧹 Use main menu option 9 to clean up all demo resources")
                
            except Exception as e:
                print(f"   ❌ Qdrant integration error: {e}")
                print("   💡 Make sure Qdrant is running at the configured URL")
                print("   💡 Check the Qdrant Web UI at http://localhost:6333/dashboard")
        
        print(f"\n🎉 Demo Complete! Here's what we accomplished:")
        print("   ✅ Loaded FastEmbed model (BAAI/bge-small-en-v1.5)")
        print("   ✅ Converted 5 documents to 384-dimensional embeddings")
        print("   ✅ Analyzed document similarities using cosine similarity")
        print("   ✅ Stored embeddings in Qdrant vector database")
        print("   ✅ Performed semantic search with query embedding")
        print("   ✅ Retrieved most relevant documents based on meaning")
        print("   💡 Demo collection created for further experimentation")
        
        print(f"\n✨ Key features of dense embeddings:")
        print("   • General purpose semantic understanding")
        print("   • Good for similarity search and clustering")
        print("   • Works well with vector databases like Qdrant")
        print("   • Suitable for most NLP tasks")
        print("   • Fast inference with FastEmbed optimization")
        print("   • High-quality semantic representations")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    run_basic_embeddings_demo()
