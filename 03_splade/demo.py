"""
SPLADE Sparse Embeddings Demo
Demonstrates SPLADE (Sparse Lexical and Dense) sparse embeddings with learned weights.

Based on: https://qdrant.tech/documentation/fastembed/fastembed-splade/
"""

from typing import List, Dict, Tuple
import numpy as np

# Import Qdrant client
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


def run_splade_demo():
    """Demonstrate SPLADE sparse embeddings."""
    print("\n🔹 SPLADE Sparse Embeddings Demo")
    print("-" * 40)
    
    if not QDRANT_AVAILABLE:
        print("❌ Qdrant client not available. Please install: pip install qdrant-client[fastembed]")
        print("💡 This demo shows the concepts without actual Qdrant integration.")
        print()
    
    try:
        print("📝 SPLADE (Sparse Lexical and Dense) generates sparse embeddings with learned weights.")
        print("🎯 It's excellent for capturing both lexical and semantic information.")
        print()
        
        # Sample texts for SPLADE demonstration
        sample_texts = [
            "The third planet from the sun, Earth, is the only known planet with life.",
            "Machine learning algorithms for natural language processing",
            "Vector databases enable efficient similarity search and retrieval",
            "Artificial intelligence transforms healthcare and medical diagnosis"
        ]
        
        print("📄 Sample texts for SPLADE analysis:")
        for i, text in enumerate(sample_texts, 1):
            print(f"   {i}. {text}")
        print()
        
        # Demonstrate SPLADE concept with the first text
        sample_text = sample_texts[0]
        print(f"🔍 SPLADE analysis for: '{sample_text}'")
        print()
        
        # Simulate SPLADE token weights (in real implementation, these would be learned)
        splade_weights = {
            'planet': 0.85,    # High weight - key concept
            'earth': 0.80,     # High weight - main subject
            'life': 0.70,      # High weight - important concept
            'sun': 0.60,       # Medium weight - context
            'third': 0.40,     # Lower weight - positional
            'known': 0.35,     # Lower weight - modifier
            'only': 0.30,      # Lower weight - qualifier
            'from': 0.20,      # Low weight - preposition
            'the': 0.10,       # Very low weight - stop word
            'is': 0.10,        # Very low weight - stop word
        }
        
        print("📊 SPLADE would generate sparse embeddings with learned weights:")
        for token, weight in sorted(splade_weights.items(), key=lambda x: x[1], reverse=True):
            importance = "🔴 High" if weight > 0.6 else "🟡 Medium" if weight > 0.3 else "🟢 Low"
            print(f"   {importance} '{token}': {weight:.2f}")
        print()
        
        # Show term expansion concept
        print("🔍 SPLADE term expansion capabilities:")
        expanded_terms = {
            'planet': ['world', 'globe', 'celestial body', 'orb'],
            'earth': ['world', 'globe', 'terra', 'planet earth'],
            'life': ['living', 'existence', 'biology', 'organisms'],
            'sun': ['star', 'solar', 'sunlight', 'daylight']
        }
        
        for term, expansions in expanded_terms.items():
            if term in splade_weights and splade_weights[term] > 0.5:
                print(f"   '{term}' (weight: {splade_weights[term]:.2f}) → expands to: {', '.join(expansions)}")
        print()
        
        print("✨ Key SPLADE features:")
        print("   • Term expansion: Generates weights for related terms not in text")
        print("   • Learned weights: Weights are learned during training, not just TF-IDF")
        print("   • Sparse representation: Only non-zero weights are stored")
        print("   • Typo resilience: Handles out-of-vocabulary tokens gracefully")
        print("   • Lexical + semantic: Combines exact matching with meaning")
        print()
        
        print("🎯 When to use SPLADE:")
        print("   • When you need both lexical and semantic matching")
        print("   • Domain-specific vocabularies and terminology")
        print("   • Search scenarios requiring exact term matching")
        print("   • Hybrid search with dense embeddings")
        print("   • Technical documentation and scientific papers")
        print()
        
        # Show comparison with other methods
        print("📊 SPLADE vs other methods:")
        print("   🔸 vs BM25: Learned weights vs statistical TF-IDF")
        print("   🔸 vs Dense: Explicit term matching vs pure semantic")
        print("   🔸 vs miniCOIL: Term expansion vs context-aware ranking")
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
                
                # Create collection for SPLADE demo
                collection_name = "splade_demo"
                print(f"   Creating collection: {collection_name}")
                
                try:
                    client.create_collection(
                        collection_name=collection_name,
                        sparse_vectors_config={
                            "splade": models.SparseVectorParams()
                        }
                    )
                    print("   ✅ Collection created with sparse vector support!")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        print("   ℹ️  Collection already exists, using existing one")
                    else:
                        raise e
                
                # Upload documents with SPLADE inference
                print("   Uploading documents with SPLADE inference...")
                points = []
                for i, text in enumerate(sample_texts):
                    points.append(
                        models.PointStruct(
                            id=i + 1,
                            payload={"text": text, "doc_id": i + 1},
                            vector={
                                "splade": models.Document(
                                    text=text,
                                    model="splade-model"  # Use appropriate SPLADE model
                                )
                            }
                        )
                    )
                
                client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                print(f"   ✅ Uploaded {len(points)} documents with SPLADE embeddings!")
                
                # Perform a search query
                query_text = "machine learning algorithms"
                print(f"\n🔍 Searching for: '{query_text}'")
                
                # Search in Qdrant with SPLADE
                search_results = client.query_points(
                    collection_name=collection_name,
                    query=models.Document(
                        text=query_text,
                        model="splade-model"
                    ),
                    using="splade",
                    limit=3
                )
                
                print("   📊 SPLADE search results from Qdrant:")
                for i, result in enumerate(search_results.points, 1):
                    print(f"      {i}. Score: {result.score:.4f} - {result.payload['text']}")
                
                # Clean up - delete the demo collection
                print(f"\n🧹 Cleaning up demo collection...")
                client.delete_collection(collection_name)
                print("   ✅ Demo collection deleted")
                
            except Exception as e:
                print(f"   ❌ Qdrant integration error: {e}")
                print("   💡 Make sure Qdrant is running at the configured URL")
                print("   💡 Note: SPLADE requires Qdrant with sparse vector support")
        else:
            print("💡 To use SPLADE with Qdrant:")
            print("   1. Install: pip install qdrant-client[fastembed]")
            print("   2. Set up Qdrant instance")
            print("   3. Create collection with sparse vector configuration")
            print("   4. Upload documents with SPLADE inference")
            print("   5. Query with learned sparse representations")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    run_splade_demo()
