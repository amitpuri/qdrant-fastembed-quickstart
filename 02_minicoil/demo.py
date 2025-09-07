"""
miniCOIL Sparse Retrieval Demo
Demonstrates miniCOIL sparse neural retrieval that combines BM25 with semantic understanding.

Based on: https://qdrant.tech/documentation/fastembed/fastembed-minicoil/
"""

import os
from typing import List
import numpy as np

# Import Qdrant client
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


def run_minicoil_demo():
    """Demonstrate miniCOIL sparse retrieval."""
    print("\n🔹 miniCOIL Sparse Retrieval Demo")
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
    
    try:
        print("📝 miniCOIL is a sparse neural retrieval model that combines BM25 with semantic understanding.")
        print("🎯 It's designed for cases where exact keyword matches are important but should be ranked by meaning.")
        print()
        
        # Sample documents for miniCOIL demonstration
        minicoil_docs = [
            "Vector Graphics in Modern Web Design",
            "The Art of Search and Self-Discovery", 
            "Efficient Vector Search Algorithms for Large Datasets",
            "Searching the Soul: A Journey Through Mindfulness",
            "Vector-Based Animations for User Interface Design",
            "Search Engines: A Technical and Social Overview",
            "The Rise of Vector Databases in AI Systems",
            "Search Patterns in Human Behavior",
            "Vector Illustrations: A Guide for Creatives",
            "Search and Rescue: Technologies in Emergency Response",
            "Vectors in Physics: From Arrows to Equations",
            "Searching for Lost Time in the Digital Age",
            "Vector Spaces and Linear Transformations",
            "The Endless Search for Truth in Philosophy",
            "3D Modeling with Vectors in Blender",
            "Search Optimization Strategies for E-commerce",
            "Vector Drawing Techniques with Open-Source Tools",
            "In Search of Meaning: A Psychological Perspective",
            "Advanced Vector Calculus for Engineers",
            "Search Interfaces: UX Principles and Case Studies",
            "The Use of Vector Fields in Meteorology",
            "Search and Destroy: Cybersecurity in the 21st Century",
            "From Bitmap to Vector: A Designer's Guide",
            "Search Engines and the Democratization of Knowledge",
            "Vector Geometry in Game Development",
            "The Human Search for Connection in a Digital World",
            "AI-Powered Vector Search in Recommendation Systems",
            "Searchable Archives: The History of Digital Retrieval",
            "Vector Control Strategies in Public Health",
            "The Search for Extraterrestrial Intelligence"
        ]
        
        query = "Vectors in Medicine"
        print(f"🔍 Query: '{query}'")
        print()
        
        print("📊 Expected behavior comparison:")
        print("   🔸 BM25 would match: 'Advanced Vector Calculus for Engineers'")
        print("      (exact 'vector' keyword match, but engineering context)")
        print("   🔸 miniCOIL would match: 'Vector Control Strategies in Public Health'")
        print("      (semantic 'medicine' context understanding)")
        print()
        
        print("📚 Sample documents containing 'vector':")
        vector_docs = [doc for doc in minicoil_docs if 'vector' in doc.lower()]
        for i, doc in enumerate(vector_docs[:8], 1):
            print(f"   {i:2d}. {doc}")
        print()
        
        print("🔍 miniCOIL scoring formula:")
        print("   miniCOIL(D,Q) = Σ IDF(qi) × Importance_D^qi × Meaning^qi×dj")
        print("   where keyword dj ∈ D equals qi")
        print()
        
        print("✨ Key miniCOIL advantages:")
        print("   • Combines BM25's keyword matching with neural semantic understanding")
        print("   • Understands context and meaning of keywords")
        print("   • Better ranking for domain-specific queries")
        print("   • Maintains exact keyword match requirements")
        print("   • Scales well with Qdrant's IDF calculations")
        print()
        
        print("🎯 When to use miniCOIL:")
        print("   • When exact keyword matches are required")
        print("   • But context and meaning matter for ranking")
        print("   • Domain-specific search (medical, legal, technical)")
        print("   • Hybrid search scenarios")
        print()
        
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
                
                # Create collection for miniCOIL demo
                collection_name = "minicoil_demo"
                print(f"   Creating collection: {collection_name}")
                
                try:
                    client.create_collection(
                        collection_name=collection_name,
                        sparse_vectors_config={
                            "minicoil": models.SparseVectorParams(
                                modifier=models.Modifier.IDF
                            )
                        }
                    )
                    print("   ✅ Collection created with sparse vector support!")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        print("   ℹ️  Collection already exists, using existing one")
                    else:
                        raise e
                
                # Upload documents with miniCOIL inference
                print("   Uploading documents with miniCOIL inference...")
                points = []
                for i, doc in enumerate(minicoil_docs[:5]):  # Use first 5 docs for demo
                    points.append(
                        models.PointStruct(
                            id=i + 1,
                            payload={"text": doc, "doc_id": i + 1},
                            vector={
                                "minicoil": models.Document(
                                    text=doc,
                                    model="Qdrant/minicoil-v1",
                                    options={"avg_len": 10}  # Approximate average document length
                                )
                            }
                        )
                    )
                
                client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                print(f"   ✅ Uploaded {len(points)} documents with miniCOIL embeddings!")
                
                # Perform a search query
                print(f"\n🔍 Searching for: '{query}'")
                
                # Search in Qdrant with miniCOIL
                search_results = client.query_points(
                    collection_name=collection_name,
                    query=models.Document(
                        text=query,
                        model="Qdrant/minicoil-v1",
                        options={"avg_len": 10}
                    ),
                    using="minicoil",
                    limit=3
                )
                
                print("   📊 miniCOIL search results from Qdrant:")
                for i, result in enumerate(search_results.points, 1):
                    print(f"      {i}. Score: {result.score:.4f} - {result.payload['text']}")
                
                # Clean up - delete the demo collection
                print(f"\n🧹 Cleaning up demo collection...")
                client.delete_collection(collection_name)
                print("   ✅ Demo collection deleted")
                
            except Exception as e:
                print(f"   ❌ Qdrant integration error: {e}")
                print("   💡 Make sure Qdrant is running at the configured URL")
                print("   💡 Note: miniCOIL requires Qdrant with sparse vector support")
        else:
            print("💡 To use miniCOIL with Qdrant:")
            print("   1. Install: pip install qdrant-client[fastembed]")
            print("   2. Set up Qdrant instance (local or cloud)")
            print("   3. Create collection with sparse vector configuration")
            print("   4. Upload documents with miniCOIL inference")
            print("   5. Query with semantic keyword understanding")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    run_minicoil_demo()
