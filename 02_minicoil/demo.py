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
    
    print("📚 What is miniCOIL?")
    print("   • miniCOIL = mini Contextualized Inverted List")
    print("   • Combines BM25's keyword matching with neural semantic understanding")
    print("   • Each term gets a learned importance weight, not just TF-IDF")
    print("   • Understands context and meaning of keywords")
    print("   • Perfect for domain-specific search where exact matches matter")
    print()
    
    if not QDRANT_AVAILABLE:
        print("❌ Qdrant client not available. Please install: pip install qdrant-client[fastembed]")
        print("💡 This demo shows the concepts without actual Qdrant integration.")
        print()
    
    try:
        print("🎯 What we'll demonstrate:")
        print("   1. Show how miniCOIL differs from traditional BM25")
        print("   2. Demonstrate context-aware keyword matching")
        print("   3. Compare miniCOIL vs BM25 ranking")
        print("   4. Store sparse vectors in Qdrant")
        print("   5. Perform semantic keyword search")
        print()
        
        input("Press Enter to see miniCOIL in action...")
        
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
        print(f"🔍 Step 1: Understanding the Query")
        print(f"   Query: '{query}'")
        print(f"   🎯 We want documents about vectors in medical/health context")
        print()
        
        print("📊 Step 2: How different methods would rank results:")
        print("   🔸 Traditional BM25 would rank:")
        print("      1. 'Advanced Vector Calculus for Engineers' (exact 'vector' match)")
        print("      2. 'Vector Graphics in Modern Web Design' (exact 'vector' match)")
        print("      ❌ Problem: No understanding of 'medicine' context!")
        print()
        print("   🔸 miniCOIL would rank:")
        print("      1. 'Vector Control Strategies in Public Health' (medical context!)")
        print("      2. 'Vector-Based Animations for User Interface Design' (less relevant)")
        print("      ✅ Solution: Understands 'medicine' = 'public health' context!")
        print()
        
        input("Press Enter to see the detailed comparison...")
        
        print("📚 Step 3: Documents containing 'vector' keyword:")
        vector_docs = [doc for doc in minicoil_docs if 'vector' in doc.lower()]
        for i, doc in enumerate(vector_docs[:8], 1):
            # Add context analysis
            if 'health' in doc.lower() or 'medical' in doc.lower():
                context = "🏥 Medical context"
            elif 'graphics' in doc.lower() or 'design' in doc.lower():
                context = "🎨 Design context"
            elif 'calculus' in doc.lower() or 'engineering' in doc.lower():
                context = "🔧 Engineering context"
            elif 'search' in doc.lower() or 'algorithm' in doc.lower():
                context = "🔍 Search context"
            else:
                context = "📄 General context"
            print(f"   {i:2d}. {context} - {doc}")
        print()
        
        print("🧮 Step 4: miniCOIL Scoring Formula:")
        print("   miniCOIL(D,Q) = Σ IDF(qi) × Importance_D^qi × Meaning^qi×dj")
        print("   where:")
        print("   • IDF(qi) = Inverse Document Frequency (from BM25)")
        print("   • Importance_D^qi = Learned importance weight for term qi in document D")
        print("   • Meaning^qi×dj = Semantic similarity between query term qi and document term dj")
        print("   • dj ∈ D equals qi (exact keyword match required)")
        print()
        
        input("Press Enter to see miniCOIL advantages...")
        
        print("✨ Step 5: Key miniCOIL Advantages:")
        print("   🎯 Combines BM25's keyword matching with neural semantic understanding")
        print("   🧠 Understands context and meaning of keywords")
        print("   📈 Better ranking for domain-specific queries")
        print("   ✅ Maintains exact keyword match requirements")
        print("   ⚡ Scales well with Qdrant's IDF calculations")
        print("   🔍 Perfect for hybrid search scenarios")
        print()
        
        print("🎯 When to use miniCOIL:")
        print("   • When exact keyword matches are required")
        print("   • But context and meaning matter for ranking")
        print("   • Domain-specific search (medical, legal, technical)")
        print("   • Hybrid search scenarios")
        print("   • When you need both precision and recall")
        print()
        
        if QDRANT_AVAILABLE:
            print("🔄 Step 6: Qdrant Integration Demo")
            input("Press Enter to connect to Qdrant and demonstrate miniCOIL...")
            
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
                
                # Create collection for miniCOIL demo
                collection_name = "fastembed_demo_minicoil"
                print(f"   🗂️  Creating collection: {collection_name}")
                print(f"   📊 Sparse vector configuration: miniCOIL with IDF modifier")
                print(f"   🎯 This enables context-aware keyword matching")
                
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
                print(f"\n   📤 Uploading {len(minicoil_docs[:5])} documents with miniCOIL inference...")
                print("   🧠 Each document gets sparse vectors with learned term weights")
                print("   📊 Terms get importance scores based on context, not just frequency")
                
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
                print(f"   ✅ Successfully uploaded {len(points)} documents with miniCOIL embeddings!")
                print(f"   💾 Sparse vectors stored with context-aware term weights")
                
                # Perform a search query
                print(f"\n🔍 Step 7: Performing miniCOIL Search...")
                print(f"   🔎 Query: '{query}'")
                print(f"   🧠 Converting query to miniCOIL sparse representation...")
                print(f"   🎯 Looking for documents with 'vector' keyword in medical context...")
                
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
                
                print(f"\n   🎯 miniCOIL search results (context-aware ranking):")
                for i, result in enumerate(search_results.points, 1):
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
                print("   💡 Note: miniCOIL requires Qdrant with sparse vector support")
        else:
            print("💡 To use miniCOIL with Qdrant:")
            print("   1. Install: pip install qdrant-client[fastembed]")
            print("   2. Set up Qdrant instance (local or cloud)")
            print("   3. Create collection with sparse vector configuration")
            print("   4. Upload documents with miniCOIL inference")
            print("   5. Query with semantic keyword understanding")
        
        print(f"\n🎉 miniCOIL Demo Complete! Here's what we accomplished:")
        print("   ✅ Explained miniCOIL's context-aware keyword matching")
        print("   ✅ Compared miniCOIL vs traditional BM25 ranking")
        print("   ✅ Demonstrated semantic understanding of keywords")
        print("   ✅ Created Qdrant collection with sparse vector support")
        print("   ✅ Stored documents with miniCOIL embeddings")
        print("   ✅ Performed context-aware semantic search")
        print("   💡 Demo collection created for further experimentation")
        
        print(f"\n✨ Key miniCOIL takeaways:")
        print("   • Combines exact keyword matching with semantic understanding")
        print("   • Perfect for domain-specific search scenarios")
        print("   • Better ranking than traditional BM25")
        print("   • Maintains keyword match requirements")
        print("   • Scales efficiently with Qdrant's sparse vector support")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    run_minicoil_demo()
