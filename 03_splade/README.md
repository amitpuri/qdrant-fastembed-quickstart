# SPLADE Sparse Embeddings

This folder demonstrates SPLADE (Sparse Lexical and Dense) sparse embeddings with learned weights.

## Overview

SPLADE generates sparse embeddings with learned weights that capture both lexical and semantic information. Unlike traditional TF-IDF, SPLADE learns optimal weights during training and can expand terms to related concepts not present in the original text.

## Features

- **Model**: SPLADE variants
- **Type**: Sparse embeddings with learned weights
- **Use Case**: Lexical + semantic hybrid search
- **Performance**: Medium speed, high accuracy for term matching

## Files

- `demo.py` - Interactive demonstration of SPLADE concepts
- `README.md` - This documentation

## Usage

```bash
python demo.py
```

## What You'll Learn

1. How SPLADE generates learned sparse weights
2. Term expansion capabilities
3. Difference between SPLADE and traditional TF-IDF
4. When to use sparse vs dense embeddings
5. Integration with vector databases

## Key Concepts

### Learned Weights
- Weights are learned during training, not just statistical
- Better than TF-IDF for semantic understanding
- Captures term importance in context
- Optimized for retrieval tasks

### Term Expansion
- Can generate weights for related terms not in text
- Handles synonyms and related concepts
- Improves recall for semantic queries
- Reduces vocabulary mismatch

### Sparse Representation
- Only non-zero weights are stored
- Efficient storage and retrieval
- Fast similarity computation
- Scalable to large vocabularies

## Example Analysis

**Text**: "The third planet from the sun, Earth, is the only known planet with life."

**SPLADE Weights**:
- 'planet': 0.85 (high weight - key concept)
- 'earth': 0.80 (high weight - main subject)
- 'life': 0.70 (high weight - important concept)
- 'sun': 0.60 (medium weight - context)
- 'third': 0.40 (lower weight - positional)
- 'known': 0.35 (lower weight - modifier)

**Term Expansion**:
- 'planet' → ['world', 'globe', 'celestial body', 'orb']
- 'earth' → ['world', 'globe', 'terra', 'planet earth']
- 'life' → ['living', 'existence', 'biology', 'organisms']

## When to Use

✅ **Good for:**
- When you need both lexical and semantic matching
- Domain-specific vocabularies and terminology
- Search scenarios requiring exact term matching
- Hybrid search with dense embeddings
- Technical documentation and scientific papers
- When term expansion is beneficial

❌ **Not ideal for:**
- When pure semantic understanding is sufficient
- Simple similarity tasks
- Cross-lingual applications
- When storage efficiency is critical

## Advantages

- **Learned Weights**: Better than statistical TF-IDF
- **Term Expansion**: Handles vocabulary mismatch
- **Sparse Efficiency**: Fast retrieval and storage
- **Semantic + Lexical**: Best of both worlds
- **Typo Resilience**: Handles out-of-vocabulary tokens
- **Domain Adaptable**: Works across different domains

## Comparison with Other Methods

| Method | Weights | Term Expansion | Semantic | Lexical |
|--------|---------|----------------|----------|---------|
| TF-IDF | Statistical | No | No | Yes |
| SPLADE | Learned | Yes | Yes | Yes |
| Dense | N/A | N/A | Yes | No |
| miniCOIL | Learned | No | Yes | Yes |

## Qdrant Integration

SPLADE works seamlessly with Qdrant's sparse vector support:

```python
# Create collection with sparse vectors
client.create_collection(
    collection_name="splade_collection",
    sparse_vectors_config={
        "splade": models.SparseVectorParams()
    }
)

# Upload documents with SPLADE inference
client.upsert(
    collection_name="splade_collection",
    points=[
        models.PointStruct(
            id=1,
            payload={"text": "Machine learning algorithms"},
            vector={
                "splade": models.Document(
                    text="Machine learning algorithms",
                    model="splade-model"
                )
            }
        )
    ]
)
```

## Next Steps

- Try the [miniCOIL demo](../minicoil/) for context-aware ranking
- Explore [ColBERT](../colbert/) for fine-grained matching
- See [reranking](../reranking/) for improving results
- Learn about [Qdrant integration](../qdrant_integration/)

## Resources

- [SPLADE Documentation](https://qdrant.tech/documentation/fastembed/fastembed-splade/)
- [SPLADE Paper](https://arxiv.org/abs/2102.06966)
- [Sparse Embeddings Guide](https://qdrant.tech/documentation/concepts/sparse-vectors/)
- [Term Expansion in Search](https://qdrant.tech/blog/term-expansion/)
