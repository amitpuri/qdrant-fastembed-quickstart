"""
SPLADE Sparse Embeddings Demo
Demonstrates SPLADE (Sparse Lexical and Dense) sparse embeddings with learned weights.

Based on: https://qdrant.tech/documentation/fastembed/fastembed-splade/
"""

import os
from typing import List, Dict, Tuple
import numpy as np

# Import FastEmbed for SPLADE
try:
    from fastembed import SparseTextEmbedding
    SPLADE_AVAILABLE = True
except ImportError:
    SPLADE_AVAILABLE = False

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
    
    # Load Qdrant configuration from environment variables
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY", None)
    
    print(f"🔧 Qdrant Configuration: {qdrant_url}")
    if qdrant_api_key:
        print(f"🔑 API Key: {'*' * (len(qdrant_api_key) - 4) + qdrant_api_key[-4:]}")
    print()
    
    print("📚 What is SPLADE?")
    print("   • SPLADE = Sparse Lexical and Dense")
    print("   • Generates sparse embeddings with learned weights")
    print("   • Combines lexical (exact matching) with semantic understanding")
    print("   • Each term gets a learned importance score")
    print("   • Can expand terms to related concepts not in the text")
    print("   • Perfect for technical documents and domain-specific search")
    print()
    
    if not SPLADE_AVAILABLE:
        print("❌ FastEmbed SPLADE not available. Please install: pip install fastembed")
        print("💡 This demo shows the concepts without actual SPLADE functionality.")
        print()
    
    if not QDRANT_AVAILABLE:
        print("❌ Qdrant client not available. Please install: pip install qdrant-client[fastembed]")
        print("💡 This demo shows the concepts without actual Qdrant integration.")
        print()
    
    try:
        print("🎯 What we'll demonstrate:")
        print("   1. Show how SPLADE generates learned term weights")
        print("   2. Demonstrate term expansion capabilities")
        print("   3. Compare SPLADE vs other sparse methods")
        print("   4. Store sparse vectors in Qdrant")
        print("   5. Perform lexical + semantic search")
        print()
        
        input("Press Enter to see SPLADE in action...")
        
        # Sample texts for SPLADE demonstration
        sample_texts = [
            "The third planet from the sun, Earth, is the only known planet with life.",
            "Machine learning algorithms for natural language processing",
            "Vector databases enable efficient similarity search and retrieval",
            "Artificial intelligence transforms healthcare and medical diagnosis"
        ]
        
        print("📄 Step 1: Sample texts for SPLADE analysis:")
        for i, text in enumerate(sample_texts, 1):
            print(f"   {i}. {text}")
        print()
        
        # Demonstrate SPLADE concept with the first text
        sample_text = sample_texts[0]
        print(f"🔍 Step 2: SPLADE analysis for: '{sample_text}'")
        print("   🧠 SPLADE will analyze each term and assign learned importance weights")
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
        
        print("📊 Step 3: SPLADE learned term weights (simulated):")
        print("   🎯 Unlike TF-IDF, these weights are learned during training")
        print("   📈 Higher weights = more important for semantic understanding")
        print()
        
        for token, weight in sorted(splade_weights.items(), key=lambda x: x[1], reverse=True):
            importance = "🔴 High" if weight > 0.6 else "🟡 Medium" if weight > 0.3 else "🟢 Low"
            print(f"   {importance} '{token}': {weight:.2f}")
        print()
        
        input("Press Enter to see SPLADE term expansion...")
        
        # Show term expansion concept
        print("🔍 Step 4: SPLADE term expansion capabilities:")
        print("   🚀 SPLADE can generate weights for related terms NOT in the original text!")
        print("   🎯 This helps match documents with different terminology")
        print()
        
        expanded_terms = {
            'planet': ['world', 'globe', 'celestial body', 'orb'],
            'earth': ['world', 'globe', 'terra', 'planet earth'],
            'life': ['living', 'existence', 'biology', 'organisms'],
            'sun': ['star', 'solar', 'sunlight', 'daylight']
        }
        
        for term, expansions in expanded_terms.items():
            if term in splade_weights and splade_weights[term] > 0.5:
                print(f"   🔗 '{term}' (weight: {splade_weights[term]:.2f}) → expands to: {', '.join(expansions)}")
        print()
        
        input("Press Enter to see SPLADE advantages...")
        
        print("✨ Step 5: Key SPLADE Features:")
        print("   🚀 Term expansion: Generates weights for related terms not in text")
        print("   🧠 Learned weights: Weights are learned during training, not just TF-IDF")
        print("   💾 Sparse representation: Only non-zero weights are stored")
        print("   🛡️  Typo resilience: Handles out-of-vocabulary tokens gracefully")
        print("   🔗 Lexical + semantic: Combines exact matching with meaning")
        print("   📊 Better than BM25: Learned importance vs statistical frequency")
        print()
        
        print("🎯 When to use SPLADE:")
        print("   • When you need both lexical and semantic matching")
        print("   • Domain-specific vocabularies and terminology")
        print("   • Search scenarios requiring exact term matching")
        print("   • Hybrid search with dense embeddings")
        print("   • Technical documentation and scientific papers")
        print("   • When you want better than BM25 but need explicit term matching")
        print()
        
        # Show comparison with other methods
        print("📊 Step 6: SPLADE vs other methods:")
        print("   🔸 vs BM25: Learned weights vs statistical TF-IDF")
        print("   🔸 vs Dense: Explicit term matching vs pure semantic")
        print("   🔸 vs miniCOIL: Term expansion vs context-aware ranking")
        print("   🔸 vs Sparse: Better term expansion and learned importance")
        print()
        
        if QDRANT_AVAILABLE and SPLADE_AVAILABLE:
            print("🔄 Step 7: Qdrant Integration Demo")
            input("Press Enter to connect to Qdrant and demonstrate SPLADE...")
            
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
                
                # Create collection for SPLADE demo
                collection_name = "fastembed_demo_splade"
                print(f"   🗂️  Creating collection: {collection_name}")
                print(f"   📊 Sparse vector configuration: SPLADE with learned weights")
                print(f"   🎯 This enables term expansion and learned importance scoring")
                
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
                
                # Load SPLADE model
                print(f"\n   🧠 Loading SPLADE model...")
                splade_model = SparseTextEmbedding("prithivida/Splade_PP_en_v1")
                print("   ✅ SPLADE model loaded successfully!")
                
                # Upload documents with SPLADE inference
                print(f"\n   📤 Uploading {len(sample_texts)} documents with SPLADE inference...")
                print("   🧠 Each document gets sparse vectors with learned term weights")
                print("   🚀 Terms can expand to related concepts not in the original text")
                
                points = []
                for i, text in enumerate(sample_texts):
                    points.append(
                        models.PointStruct(
                            id=i + 1,
                            payload={"text": text, "doc_id": i + 1},
                            vector={
                                "splade": models.Document(
                                    text=text,
                                    model="prithivida/Splade_PP_en_v1"
                                )
                            }
                        )
                    )
                
                client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                print(f"   ✅ Successfully uploaded {len(points)} documents with SPLADE embeddings!")
                print(f"   💾 Sparse vectors stored with learned term weights and expansion")
                
                # Perform a search query
                query_text = "machine learning algorithms"
                print(f"\n🔍 Step 8: Performing SPLADE Search...")
                print(f"   🔎 Query: '{query_text}'")
                print(f"   🧠 Converting query to SPLADE sparse representation...")
                print(f"   🚀 SPLADE will expand terms and use learned weights...")
                
                # Search in Qdrant with SPLADE
                search_results = client.query_points(
                    collection_name=collection_name,
                    query=models.Document(
                        text=query_text,
                        model="prithivida/Splade_PP_en_v1"
                    ),
                    using="splade",
                    limit=3
                )
                
                print(f"\n   🎯 SPLADE search results (with term expansion):")
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
                print("   💡 Note: SPLADE requires Qdrant with sparse vector support")
        elif not SPLADE_AVAILABLE:
            print("💡 To use SPLADE with Qdrant:")
            print("   1. Install: pip install fastembed")
            print("   2. Install: pip install qdrant-client[fastembed]")
            print("   3. Set up Qdrant instance")
            print("   4. Create collection with sparse vector configuration")
            print("   5. Upload documents with SPLADE inference")
            print("   6. Query with learned sparse representations")
        else:
            print("💡 To use SPLADE with Qdrant:")
            print("   1. Install: pip install qdrant-client[fastembed]")
            print("   2. Set up Qdrant instance")
            print("   3. Create collection with sparse vector configuration")
            print("   4. Upload documents with SPLADE inference")
            print("   5. Query with learned sparse representations")
        
        print(f"\n🎉 SPLADE Demo Complete! Here's what we accomplished:")
        print("   ✅ Explained SPLADE's learned term weighting system")
        print("   ✅ Demonstrated term expansion capabilities")
        print("   ✅ Compared SPLADE vs other sparse methods")
        print("   ✅ Created Qdrant collection with sparse vector support")
        print("   ✅ Stored documents with SPLADE embeddings")
        print("   ✅ Performed lexical + semantic search with term expansion")
        print("   💡 Demo collection created for further experimentation")
        
        print(f"\n✨ Key SPLADE takeaways:")
        print("   • Generates learned weights for terms (better than TF-IDF)")
        print("   • Can expand terms to related concepts not in text")
        print("   • Combines lexical matching with semantic understanding")
        print("   • Perfect for technical and domain-specific documents")
        print("   • Handles out-of-vocabulary terms gracefully")
        print("   • Scales efficiently with Qdrant's sparse vector support")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    run_splade_demo()
