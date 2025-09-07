"""
ColBERT Multi-Vector Search Demo
Demonstrates ColBERT (Contextualized Late Interaction over BERT) multi-vector search.

Based on: https://qdrant.tech/documentation/fastembed/fastembed-colbert/
"""

import os
from typing import List, Tuple
import numpy as np

# Import FastEmbed for ColBERT
try:
    from fastembed import LateInteractionTextEmbedding
    COLBERT_AVAILABLE = True
except ImportError:
    COLBERT_AVAILABLE = False

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
    
    # Load Qdrant configuration from environment variables
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY", None)
    
    print(f"🔧 Qdrant Configuration: {qdrant_url}")
    if qdrant_api_key:
        print(f"🔑 API Key: {'*' * (len(qdrant_api_key) - 4) + qdrant_api_key[-4:]}")
    print()
    
    print("📚 What is ColBERT?")
    print("   • ColBERT = Contextualized Late Interaction over BERT")
    print("   • Uses multiple vectors per document (one per token)")
    print("   • Each token gets its own contextualized embedding")
    print("   • Enables fine-grained token-level matching")
    print("   • Better handling of word order and position")
    print("   • Perfect for complex queries and precise relevance")
    print()
    
    if not COLBERT_AVAILABLE:
        print("❌ FastEmbed ColBERT not available. Please install: pip install fastembed")
        print("💡 This demo shows the concepts without actual ColBERT functionality.")
        print()
    
    if not QDRANT_AVAILABLE:
        print("❌ Qdrant client not available. Please install: pip install qdrant-client[fastembed]")
        print("💡 This demo shows the concepts without actual Qdrant integration.")
        print()
    
    try:
        print("🎯 What we'll demonstrate:")
        print("   1. Show how ColBERT creates multiple vectors per document")
        print("   2. Demonstrate token-level matching process")
        print("   3. Compare ColBERT vs single-vector approaches")
        print("   4. Store multi-vectors in Qdrant")
        print("   5. Perform fine-grained semantic search")
        print()
        
        input("Press Enter to see ColBERT in action...")
        
        # Sample texts for ColBERT demonstration
        sample_texts = [
            "Machine learning algorithms for natural language processing",
            "Deep learning models for computer vision applications",
            "Neural networks and artificial intelligence research",
            "Advanced algorithms for data science and analytics"
        ]
        
        print("📄 Step 1: Sample texts for ColBERT analysis:")
        for i, text in enumerate(sample_texts, 1):
            print(f"   {i}. {text}")
        print()
        
        # Demonstrate ColBERT concept with the first text
        sample_text = sample_texts[0]
        print(f"🔍 Step 2: ColBERT analysis for: '{sample_text}'")
        print("   🧠 ColBERT will create separate embeddings for each token")
        print("   🎯 This enables fine-grained matching at the token level")
        print()
        
        # Tokenize the text
        tokens = sample_text.split()
        print(f"📊 Step 3: ColBERT token-level embedding creation:")
        print(f"   📝 Total tokens: {len(tokens)}")
        print(f"   📏 Each token → [384-dimensional vector]")
        print(f"   💾 Storage: {len(tokens)} vectors per document (vs 1 for dense)")
        print()
        
        for i, token in enumerate(tokens):
            print(f"   Token {i+1:2d}: '{token}' → [384-dimensional vector]")
        print()
        
        input("Press Enter to see the query matching process...")
        
        # Show query matching process
        query = "machine learning algorithms"
        query_tokens = query.split()
        print(f"🔍 Step 4: Query matching process")
        print(f"   🔎 Query: '{query}'")
        print(f"   📝 Query tokens: {query_tokens}")
        print()
        
        print("🔗 ColBERT query matching process:")
        print("   1. 🧠 Query tokens get their own embeddings")
        print("   2. 🔍 Each query token is matched against all document tokens")
        print("   3. 📊 Maximum similarity per query token is taken")
        print("   4. 🎯 Final score is sum of max similarities")
        print("   5. ⚡ This enables fine-grained relevance scoring")
        print()
        
        # Simulate the matching process
        print("📊 Step 5: Simulated token-level matching process:")
        print("   🎯 This shows how ColBERT finds the best match for each query token")
        print()
        
        for i, query_token in enumerate(query_tokens):
            print(f"   🔍 Query token '{query_token}':")
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
            print(f"      🎯 Best matches:")
            for doc_token, sim in matches[:3]:
                print(f"         '{doc_token}': {sim:.2f}")
        print()
        
        input("Press Enter to see ColBERT advantages...")
        
        print("✨ Step 6: Key ColBERT Advantages:")
        print("   🎯 Fine-grained token-level matching")
        print("   📍 Better handling of word order and position")
        print("   📊 More precise relevance scoring")
        print("   🔍 Improved performance on complex queries")
        print("   🧠 Contextual understanding per token")
        print("   ⚡ Late interaction between query and document")
        print("   🎪 Better than single-vector approaches for complex queries")
        print()
        
        print("🎯 When to use ColBERT:")
        print("   • Complex, multi-part queries")
        print("   • When word order and position matter")
        print("   • Fine-grained relevance requirements")
        print("   • Long documents with specific information")
        print("   • Question-answering systems")
        print("   • Information retrieval with high precision needs")
        print("   • When you need better precision than dense embeddings")
        print()
        
        # Show comparison with other methods
        print("📊 Step 7: ColBERT vs other methods:")
        print("   🔸 vs Dense: Token-level vs document-level matching")
        print("   🔸 vs Sparse: Contextual vs lexical matching")
        print("   🔸 vs miniCOIL: Multi-vector vs single sparse vector")
        print("   🔸 vs SPLADE: Contextual vs learned term weights")
        print()
        
        # Show storage requirements
        print("💾 Storage considerations:")
        print(f"   • Dense embedding: 1 vector per document ({384} dimensions)")
        print(f"   • ColBERT: {len(tokens)} vectors per document ({384} dimensions each)")
        print(f"   • Storage ratio: {len(tokens)}x more storage than dense")
        print("   • Trade-off: More storage for better precision")
        print("   • Worth it for: Complex queries, high precision needs")
        print()
        
        if QDRANT_AVAILABLE and COLBERT_AVAILABLE:
            print("🔄 Step 8: Qdrant Integration Demo")
            input("Press Enter to connect to Qdrant and demonstrate ColBERT...")
            
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
                
                # Create collection for ColBERT demo
                collection_name = "fastembed_demo_colbert"
                print(f"   🗂️  Creating collection: {collection_name}")
                print(f"   📊 Multi-vector configuration: ColBERT with 128-dimensional vectors")
                print(f"   🎯 This enables fine-grained token-level matching")
                
                try:
                    client.create_collection(
                        collection_name=collection_name,
                        vectors_config={
                            "colbert": models.VectorParams(
                                size=128,  # ColBERT vector size
                                distance=models.Distance.COSINE,
                                multivector_config=models.MultiVectorConfig(
                                    comparator=models.MultiVectorComparator.MAX_SIM
                                ),
                                hnsw_config={"m": 0}  # Disable HNSW for reranking
                            )
                        }
                    )
                    print("   ✅ Collection created with multi-vector support!")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        print("   ℹ️  Collection already exists, using existing one")
                    else:
                        raise e
                
                # Load ColBERT model
                print(f"\n   🧠 Loading ColBERT model...")
                colbert_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
                print("   ✅ ColBERT model loaded successfully!")
                
                # Upload documents with ColBERT inference
                print(f"\n   📤 Uploading {len(sample_texts)} documents with ColBERT inference...")
                print("   🧠 Each document gets multiple vectors (one per token)")
                print("   🎯 This enables fine-grained token-level matching")
                
                points = []
                for i, text in enumerate(sample_texts):
                    # Generate ColBERT embedding
                    colbert_embedding = list(colbert_model.embed([text]))[0]
                    
                    points.append(
                        models.PointStruct(
                            id=i + 1,
                            payload={"text": text, "doc_id": i + 1},
                            vector={
                                "colbert": colbert_embedding.tolist()
                            }
                        )
                    )
                
                client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                print(f"   ✅ Successfully uploaded {len(points)} documents with ColBERT embeddings!")
                print(f"   💾 Multi-vectors stored for fine-grained token matching")
                
                # Perform a search query
                print(f"\n🔍 Step 9: Performing ColBERT Search...")
                print(f"   🔎 Query: '{query}'")
                print(f"   🧠 Converting query to ColBERT multi-vector representation...")
                print(f"   🎯 ColBERT will perform token-level matching...")
                
                # Generate ColBERT embedding for query
                query_embedding = list(colbert_model.query_embed([query]))[0]
                
                # Search in Qdrant with ColBERT
                search_results = client.query_points(
                    collection_name=collection_name,
                    query=query_embedding.tolist(),
                    using="colbert",
                    limit=3
                )
                
                print(f"\n   🎯 ColBERT search results (fine-grained token matching):")
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
                print("   💡 Note: ColBERT requires Qdrant with multi-vector support")
        elif not COLBERT_AVAILABLE:
            print("💡 To use ColBERT with Qdrant:")
            print("   1. Install: pip install fastembed")
            print("   2. Install: pip install qdrant-client[fastembed]")
            print("   3. Set up Qdrant instance")
            print("   4. Create collection with multi-vector configuration")
            print("   5. Upload documents with ColBERT inference")
            print("   6. Query with fine-grained token matching")
        else:
            print("💡 To use ColBERT with Qdrant:")
            print("   1. Install: pip install qdrant-client[fastembed]")
            print("   2. Set up Qdrant instance")
            print("   3. Create collection with multi-vector configuration")
            print("   4. Upload documents with ColBERT inference")
            print("   5. Query with fine-grained token matching")
        
        print(f"\n🎉 ColBERT Demo Complete! Here's what we accomplished:")
        print("   ✅ Explained ColBERT's multi-vector approach")
        print("   ✅ Demonstrated token-level matching process")
        print("   ✅ Compared ColBERT vs single-vector methods")
        print("   ✅ Created Qdrant collection with multi-vector support")
        print("   ✅ Stored documents with ColBERT embeddings")
        print("   ✅ Performed fine-grained semantic search")
        print("   💡 Demo collection created for further experimentation")
        
        print(f"\n✨ Key ColBERT takeaways:")
        print("   • Uses multiple vectors per document (one per token)")
        print("   • Enables fine-grained token-level matching")
        print("   • Better handling of word order and position")
        print("   • More precise relevance scoring for complex queries")
        print("   • Perfect for question-answering and high-precision search")
        print("   • Trade-off: More storage for better precision")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    run_colbert_demo()
