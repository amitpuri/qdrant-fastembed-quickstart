# ColBERT Multi-Vector Search

This folder demonstrates ColBERT (Contextualized Late Interaction over BERT) multi-vector search.

## Overview

ColBERT uses multiple vectors per document, with each token getting its own embedding. This enables fine-grained token-level matching and better handling of word order and position, making it ideal for complex queries and precise relevance scoring.

## Features

- **Model**: ColBERT variants
- **Type**: Multi-vector embeddings (one per token)
- **Use Case**: Fine-grained token matching
- **Performance**: Slower than dense, higher accuracy for complex queries

## Files

- `demo.py` - Interactive demonstration of ColBERT concepts
- `README.md` - This documentation

## Usage

```bash
python demo.py
```

## What You'll Learn

1. How ColBERT creates multiple vectors per document
2. Token-level matching process
3. When to use multi-vector vs single-vector approaches
4. Fine-grained relevance scoring
5. Storage and performance considerations

## Key Concepts

### Multi-Vector Representation
- Each token gets its own embedding
- Enables fine-grained matching
- Better handling of word order
- More precise relevance scoring

### Late Interaction
- Query and document tokens interact late in the process
- Each query token matched against all document tokens
- Maximum similarity per query token is taken
- Final score is sum of max similarities

### Token-Level Matching
- Query: "machine learning algorithms"
- Document: "Machine learning algorithms for natural language processing"
- Each query token matched against all document tokens
- Best matches contribute to final score

## Example Process

**Query**: "machine learning algorithms"
**Document**: "Machine learning algorithms for natural language processing"

**Token Matching**:
1. Query token "machine" → Best match: "Machine" (similarity: 0.95)
2. Query token "learning" → Best match: "learning" (similarity: 0.98)
3. Query token "algorithms" → Best match: "algorithms" (similarity: 0.97)

**Final Score**: 0.95 + 0.98 + 0.97 = 2.90

## When to Use

✅ **Good for:**
- Complex, multi-part queries
- When word order and position matter
- Fine-grained relevance requirements
- Long documents with specific information
- Question-answering systems
- Information retrieval with high precision needs
- When you need token-level control

❌ **Not ideal for:**
- Simple similarity search
- When storage/memory is limited
- Fast retrieval requirements
- General semantic similarity
- Cross-lingual applications

## Storage Considerations

| Method | Vectors per Document | Storage Ratio |
|--------|---------------------|---------------|
| Dense | 1 | 1x |
| ColBERT | N (tokens) | Nx |

**Example**: 10-token document
- Dense: 1 vector (384 dimensions)
- ColBERT: 10 vectors (384 dimensions each)
- Storage: 10x more than dense

## Advantages

- **Fine-Grained Matching**: Token-level precision
- **Word Order Awareness**: Better handling of position
- **Complex Query Support**: Multi-part queries work well
- **High Precision**: Excellent for specific information retrieval
- **Contextual Understanding**: Each token in context
- **Flexible Scoring**: Late interaction allows complex scoring

## Qdrant Integration

ColBERT works with Qdrant's multi-vector support:

```python
# Create collection with multi-vector support
client.create_collection(
    collection_name="colbert_collection",
    vectors_config={
        "colbert": models.VectorParams(
            size=384,
            distance=models.Distance.COSINE
        )
    }
)

# Upload documents with ColBERT inference
client.upsert(
    collection_name="colbert_collection",
    points=[
        models.PointStruct(
            id=1,
            payload={"text": "Machine learning algorithms"},
            vector={
                "colbert": models.Document(
                    text="Machine learning algorithms",
                    model="colbert-model"
                )
            }
        )
    ]
)
```

## Performance Trade-offs

| Aspect | Dense | ColBERT |
|--------|-------|---------|
| Speed | Fast | Slower |
| Memory | Low | High |
| Storage | Medium | High |
| Accuracy | Good | Very High |
| Complexity | Simple | Complex |

## Next Steps

- Try the [reranking demo](../reranking/) for improving results
- Explore [Qdrant integration](../qdrant_integration/) for storage
- See [comparison demo](../comparison/) for method selection
- Learn about [hybrid approaches](../comparison/)

## Resources

- [ColBERT Documentation](https://qdrant.tech/documentation/fastembed/fastembed-colbert/)
- [ColBERT Paper](https://arxiv.org/abs/2004.12832)
- [Multi-Vector Search](https://qdrant.tech/documentation/concepts/multi-vector/)
- [Late Interaction Models](https://qdrant.tech/blog/late-interaction/)
