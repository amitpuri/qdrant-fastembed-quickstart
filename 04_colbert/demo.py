"""
ColBERT Multi-Vector Search Demo
Demonstrates ColBERT (Contextualized Late Interaction over BERT) multi-vector search.

Based on: https://qdrant.tech/documentation/fastembed/fastembed-colbert/
"""

from typing import List, Tuple
import numpy as np

# Import Qdrant client
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


def run_colbert_demo():
    """Demonstrate ColBERT multi-vector search."""
    print("\n🔹 ColBERT Multi-Vector Search Demo")
    print("-" * 40)
    
    if not QDRANT_AVAILABLE:
        print("❌ Qdrant client not available. Please install: pip install qdrant-client[fastembed]")
        print("💡 This demo shows the concepts without actual Qdrant integration.")
        print()
    
    try:
        print("📝 ColBERT (Contextualized Late Interaction over BERT) uses multiple vectors per document.")
        print("🎯 Each token gets its own embedding, enabling fine-grained matching.")
        print()
        
        # Sample texts for ColBERT demonstration
        sample_texts = [
            "Machine learning algorithms for natural language processing",
            "Deep learning models for computer vision applications",
            "Neural networks and artificial intelligence research",
            "Advanced algorithms for data science and analytics"
        ]
        
        print("📄 Sample texts for ColBERT analysis:")
        for i, text in enumerate(sample_texts, 1):
            print(f"   {i}. {text}")
        print()
        
        # Demonstrate ColBERT concept with the first text
        sample_text = sample_texts[0]
        print(f"🔍 ColBERT analysis for: '{sample_text}'")
        print()
        
        # Tokenize the text
        tokens = sample_text.split()
        print(f"📊 ColBERT would create separate embeddings for each token:")
        print(f"   Total tokens: {len(tokens)}")
        print(f"   Each token → [384-dimensional vector]")
        print()
        
        for i, token in enumerate(tokens):
            print(f"   Token {i+1:2d}: '{token}' → [384-dimensional vector]")
        print()
        
        # Show query matching process
        query = "machine learning algorithms"
        query_tokens = query.split()
        print(f"🔍 Query: '{query}'")
        print(f"   Query tokens: {query_tokens}")
        print()
        
        print("🔗 ColBERT query matching process:")
        print("   1. Query tokens get their own embeddings")
        print("   2. Each query token is matched against all document tokens")
        print("   3. Maximum similarity per query token is taken")
        print("   4. Final score is sum of max similarities")
        print()
        
        # Simulate the matching process
        print("📊 Simulated matching process:")
        for i, query_token in enumerate(query_tokens):
            print(f"   Query token '{query_token}':")
            # Find best matches in document
            matches = []
            for j, doc_token in enumerate(tokens):
                # Simulate similarity (in real implementation, this would be cosine similarity)
                if query_token.lower() in doc_token.lower() or doc_token.lower() in query_token.lower():
                    similarity = 0.9
                elif any(char in doc_token.lower() for char in query_token.lower()):
                    similarity = 0.6
                else:
                    similarity = 0.3
                matches.append((doc_token, similarity))
            
            # Sort by similarity and show top matches
            matches.sort(key=lambda x: x[1], reverse=True)
            print(f"      Best matches:")
            for doc_token, sim in matches[:3]:
                print(f"         '{doc_token}': {sim:.2f}")
        print()
        
        print("✨ Key ColBERT advantages:")
        print("   • Fine-grained token-level matching")
        print("   • Better handling of word order and position")
        print("   • More precise relevance scoring")
        print("   • Improved performance on complex queries")
        print("   • Contextual understanding per token")
        print("   • Late interaction between query and document")
        print()
        
        print("🎯 When to use ColBERT:")
        print("   • Complex, multi-part queries")
        print("   • When word order and position matter")
        print("   • Fine-grained relevance requirements")
        print("   • Long documents with specific information")
        print("   • Question-answering systems")
        print("   • Information retrieval with high precision needs")
        print()
        
        # Show comparison with other methods
        print("📊 ColBERT vs other methods:")
        print("   🔸 vs Dense: Token-level vs document-level matching")
        print("   🔸 vs Sparse: Contextual vs lexical matching")
        print("   🔸 vs miniCOIL: Multi-vector vs single sparse vector")
        print()
        
        # Show storage requirements
        print("💾 Storage considerations:")
        print(f"   • Dense embedding: 1 vector per document ({384} dimensions)")
        print(f"   • ColBERT: {len(tokens)} vectors per document ({384} dimensions each)")
        print(f"   • Storage ratio: {len(tokens)}x more storage than dense")
        print("   • Trade-off: More storage for better precision")
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
                
                # Create collection for ColBERT demo
                collection_name = "colbert_demo"
                print(f"   Creating collection: {collection_name}")
                
                try:
                    client.create_collection(
                        collection_name=collection_name,
                        vectors_config={
                            "colbert": models.VectorParams(
                                size=384,  # ColBERT vector size
                                distance=models.Distance.COSINE
                            )
                        }
                    )
                    print("   ✅ Collection created with multi-vector support!")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        print("   ℹ️  Collection already exists, using existing one")
                    else:
                        raise e
                
                # Upload documents with ColBERT inference
                print("   Uploading documents with ColBERT inference...")
                points = []
                for i, text in enumerate(sample_texts):
                    points.append(
                        models.PointStruct(
                            id=i + 1,
                            payload={"text": text, "doc_id": i + 1},
                            vector={
                                "colbert": models.Document(
                                    text=text,
                                    model="colbert-model"  # Use appropriate ColBERT model
                                )
                            }
                        )
                    )
                
                client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                print(f"   ✅ Uploaded {len(points)} documents with ColBERT embeddings!")
                
                # Perform a search query
                print(f"\n🔍 Searching for: '{query}'")
                
                # Search in Qdrant with ColBERT
                search_results = client.query_points(
                    collection_name=collection_name,
                    query=models.Document(
                        text=query,
                        model="colbert-model"
                    ),
                    using="colbert",
                    limit=3
                )
                
                print("   📊 ColBERT search results from Qdrant:")
                for i, result in enumerate(search_results.points, 1):
                    print(f"      {i}. Score: {result.score:.4f} - {result.payload['text']}")
                
                # Clean up - delete the demo collection
                print(f"\n🧹 Cleaning up demo collection...")
                client.delete_collection(collection_name)
                print("   ✅ Demo collection deleted")
                
            except Exception as e:
                print(f"   ❌ Qdrant integration error: {e}")
                print("   💡 Make sure Qdrant is running at the configured URL")
                print("   💡 Note: ColBERT requires Qdrant with multi-vector support")
        else:
            print("💡 To use ColBERT with Qdrant:")
            print("   1. Install: pip install qdrant-client[fastembed]")
            print("   2. Set up Qdrant instance")
            print("   3. Create collection with multi-vector configuration")
            print("   4. Upload documents with ColBERT inference")
            print("   5. Query with fine-grained token matching")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    run_colbert_demo()
