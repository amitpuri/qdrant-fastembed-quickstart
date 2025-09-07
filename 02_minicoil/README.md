# miniCOIL Sparse Retrieval

This folder demonstrates miniCOIL sparse neural retrieval that combines BM25 with semantic understanding.

## Overview

miniCOIL is a sparse neural retrieval model that acts as if a BM25-based retriever understood the contextual meaning of keywords and ranked results accordingly. It combines the exact keyword matching of BM25 with the semantic understanding of neural models.

## Features

- **Model**: Qdrant/minicoil-v1
- **Type**: Sparse embeddings
- **Use Case**: Keyword-aware semantic search
- **Performance**: Medium speed, high accuracy for keyword queries

## Files

- `demo.py` - Interactive demonstration of miniCOIL concepts
- `README.md` - This documentation

## Usage

```bash
python demo.py
```

## What You'll Learn

1. How miniCOIL combines BM25 with semantic understanding
2. When to use sparse vs dense embeddings
3. Understanding keyword-aware ranking
4. Context-aware search capabilities
5. Integration with Qdrant vector database

## Demo Overview: Working with LLMs

This demo showcases miniCOIL's unique approach to semantic search that's particularly valuable for LLM applications requiring both keyword precision and semantic understanding. Here's what makes this demo essential for advanced LLM workflows:

### What This Demo Does

The demo demonstrates miniCOIL's hybrid approach that:
- **Combines keyword matching with semantic understanding**: Unlike pure semantic search, miniCOIL requires exact keyword matches while understanding their context
- **Shows context-aware ranking**: Demonstrates how "vectors in medicine" ranks medical documents higher than engineering documents, even with the same keyword
- **Illustrates the miniCOIL formula**: Shows how learned importance weights and semantic similarity work together
- **Compares with traditional BM25**: Highlights why context understanding matters for LLM applications

### How This is Useful for LLMs

**1. Domain-Specific LLM Applications**
- **Medical/Healthcare LLMs**: Ensures medical terms are matched in correct medical context
- **Legal LLMs**: Maintains precision for legal terminology while understanding legal concepts
- **Technical LLMs**: Balances exact technical term matching with semantic understanding
- **Scientific LLMs**: Handles domain-specific vocabulary with contextual relevance

**2. Hybrid Search for RAG Systems**
- **Keyword + Semantic RAG**: Combines the precision of keyword matching with semantic understanding
- **Context-Aware Retrieval**: Ensures retrieved documents match both keywords and intended meaning
- **Domain-Specific Knowledge Bases**: Perfect for specialized LLM applications with technical vocabularies
- **Multi-Modal Search**: Supports both exact term requirements and semantic similarity

**3. LLM Response Quality**
- **Reduces False Positives**: Prevents irrelevant documents from being retrieved due to keyword matches
- **Improves Context Relevance**: Ensures LLMs receive contextually appropriate information
- **Maintains Term Precision**: Preserves exact matching requirements while adding semantic intelligence
- **Better Source Attribution**: Helps LLMs cite more relevant and accurate sources

**4. Production LLM Systems**
- **Enterprise Knowledge Management**: Ideal for corporate LLMs working with technical documentation
- **Regulatory Compliance**: Ensures precise term matching for compliance-focused LLM applications
- **Customer Support LLMs**: Balances exact product name matching with semantic understanding
- **Research Assistant LLMs**: Perfect for academic and research applications requiring term precision

**5. Advanced RAG Architectures**
- **Multi-Stage Retrieval**: Can be combined with dense embeddings for comprehensive search
- **Query Understanding**: Helps LLMs better understand user intent when keywords are ambiguous
- **Contextual Ranking**: Improves the quality of retrieved context for LLM generation
- **Domain Adaptation**: Easily adapts to different domains while maintaining keyword precision

## Key Concepts

### miniCOIL Formula
```
miniCOIL(D,Q) = Σ IDF(qi) × Importance_D^qi × Meaning^qi×dj
where keyword dj ∈ D equals qi
```

### Sparse Embeddings
- Only non-zero weights are stored
- Efficient storage and retrieval
- Explicit keyword matching
- Learned importance weights

### Context Understanding
- Understands meaning of keywords in context
- Better ranking than pure BM25
- Maintains exact keyword match requirements
- Semantic similarity between matched terms

## Example Scenario

**Query**: "Vectors in Medicine"

**BM25 Result**: "Advanced Vector Calculus for Engineers"
- Exact "vector" match, but engineering context

**miniCOIL Result**: "Vector Control Strategies in Public Health"
- Semantic "medicine" context understanding

## When to Use

✅ **Good for:**
- When exact keyword matches are required
- But context and meaning are important
- Domain-specific search (medical, legal, technical)
- Hybrid search scenarios
- Keyword-heavy domains

❌ **Not ideal for:**
- Pure semantic similarity tasks
- When you don't need keyword matching
- General similarity search
- Cross-lingual applications

## Qdrant Integration

miniCOIL is designed to work seamlessly with Qdrant:

```python
# Create collection with sparse vectors
client.create_collection(
    collection_name="minicoil_collection",
    sparse_vectors_config={
        "minicoil": models.SparseVectorParams(
            modifier=models.Modifier.IDF
        )
    }
)

# Upload documents with miniCOIL inference
client.upsert(
    collection_name="minicoil_collection",
    points=[
        models.PointStruct(
            id=1,
            payload={"text": "Vector Control Strategies in Public Health"},
            vector={
                "minicoil": models.Document(
                    text="Vector Control Strategies in Public Health",
                    model="Qdrant/minicoil-v1",
                    options={"avg_len": avg_documents_length}
                )
            }
        )
    ]
)
```

## Advantages

- **Keyword Precision**: Maintains exact keyword matching
- **Context Awareness**: Understands semantic context
- **Efficient Storage**: Sparse representation saves space
- **Qdrant Optimized**: Built for Qdrant's IDF calculations
- **Domain Flexibility**: Works well across different domains

## Next Steps

- Try the [SPLADE demo](../splade/) for learned sparse weights
- Explore [ColBERT](../colbert/) for fine-grained matching
- See [reranking](../reranking/) for improving results
- Learn about [Qdrant integration](../qdrant_integration/)

## Resources

- [miniCOIL Documentation](https://qdrant.tech/documentation/fastembed/fastembed-minicoil/)
- [miniCOIL Paper](https://arxiv.org/abs/2308.11440)
- [Sparse Neural Retrieval](https://qdrant.tech/blog/sparse-neural-retrieval/)
- [Qdrant Sparse Vectors](https://qdrant.tech/documentation/concepts/sparse-vectors/)
