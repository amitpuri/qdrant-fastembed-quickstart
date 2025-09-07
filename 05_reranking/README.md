# Reranking

This folder demonstrates reranking capabilities to improve search results by re-scoring initial candidates.

## Overview

Reranking is a post-processing step that improves search results by re-scoring initial candidates using more sophisticated models. It's typically used as a second stage after initial retrieval to boost precision and improve the quality of top results.

## Features

- **Type**: Post-processing step
- **Use Case**: Improve initial retrieval results
- **Performance**: Slower than initial retrieval, higher precision
- **Models**: Cross-encoder reranking models

## Files

- `demo.py` - Interactive demonstration of reranking concepts
- `README.md` - This documentation

## Usage

```bash
python demo.py
```

## What You'll Learn

1. How reranking improves search results
2. When to use reranking in your pipeline
3. Different reranking approaches
4. Performance vs accuracy trade-offs
5. Integration with retrieval systems

## Key Concepts

### Two-Stage Retrieval
1. **Initial Retrieval**: Fast retrieval of many candidates (e.g., top 100)
2. **Reranking**: Slow, precise scoring of candidates (e.g., top 10)

### Reranking Approaches
- **Cross-encoder**: Query-document pairs scored together
- **Point-wise**: Each document scored independently
- **Pair-wise**: Document pairs compared
- **List-wise**: Entire result list optimized

### Performance Trade-offs
- **Speed**: Slower than initial retrieval
- **Accuracy**: Higher precision in top results
- **Memory**: Higher memory requirements
- **Cost**: More computational resources

## Example Improvement

**Query**: "FastEmbed and Qdrant integration"

**Before Reranking**:
1. [0.85] FastEmbed is lighter than Transformers & Sentence-Transformers.
2. [0.78] FastEmbed is supported by and maintained by Qdrant.
3. [0.65] Vector Graphics in Modern Web Design
4. [0.52] The Art of Search and Self-Discovery

**After Reranking**:
1. [0.95] FastEmbed is supported by and maintained by Qdrant. 📈
2. [0.88] FastEmbed is lighter than Transformers & Sentence-Transformers. 📈
3. [0.72] Efficient Vector Search Algorithms for Large Datasets 📈
4. [0.45] Vector Graphics in Modern Web Design 📉

## When to Use

✅ **Good for:**
- As a second stage after initial retrieval
- When precision is more important than recall
- Domain-specific search applications
- Limited result slots (e.g., top 5-10)
- Production search systems
- When combining multiple retrieval methods

❌ **Not ideal for:**
- As the only retrieval method
- When speed is critical
- Large-scale real-time applications
- When computational resources are limited
- Simple similarity search tasks

## Reranking Pipeline

```
Query → Initial Retrieval → Reranking → Final Results
       (Fast, many results)  (Slow, precise)  (Top N)
```

### Typical Pipeline
1. **Initial Retrieval**: Get top 100-1000 candidates
2. **Reranking**: Score top 100 candidates precisely
3. **Final Results**: Return top 10-20 results

## FastEmbed Reranker

FastEmbed provides pre-trained reranking models:

```python
from fastembed import Reranker

# Initialize reranker
reranker = Reranker()

# Rerank query-document pairs
query = "FastEmbed and Qdrant integration"
documents = [
    "FastEmbed is supported by and maintained by Qdrant.",
    "FastEmbed is lighter than Transformers & Sentence-Transformers.",
    "Vector Graphics in Modern Web Design"
]

# Get reranking scores
scores = reranker.rank(query, documents)
```

## Advantages

- **Higher Precision**: Better top results
- **Flexible**: Can combine multiple signals
- **Domain Adaptable**: Can be fine-tuned
- **Quality Improvement**: Significant user experience boost
- **False Positive Reduction**: Removes irrelevant results
- **Multi-Signal**: Combines semantic + lexical signals

## Performance Considerations

| Aspect | Initial Retrieval | Reranking |
|--------|-------------------|-----------|
| Speed | Fast (ms) | Slow (s) |
| Candidates | Many (100-1000) | Few (10-100) |
| Accuracy | Good | Very High |
| Memory | Low | High |
| Cost | Low | High |

## Integration Examples

### With Dense Embeddings
```python
# 1. Initial retrieval with dense embeddings
initial_results = dense_search(query, limit=100)

# 2. Rerank top candidates
reranked_results = reranker.rank(query, [doc.text for doc in initial_results[:20]])

# 3. Return top results
return reranked_results[:10]
```

### With Hybrid Search
```python
# 1. Combine dense and sparse results
dense_results = dense_search(query, limit=50)
sparse_results = sparse_search(query, limit=50)
combined = merge_results(dense_results, sparse_results)

# 2. Rerank combined results
reranked = reranker.rank(query, [doc.text for doc in combined[:30]])

# 3. Return final results
return reranked[:10]
```

## Next Steps

- Try the [comparison demo](../comparison/) for method selection
- Explore [Qdrant integration](../qdrant_integration/) for storage
- Learn about [hybrid approaches](../comparison/)
- See [basic embeddings](../basic_embeddings/) for initial retrieval

## Resources

- [Reranking Documentation](https://qdrant.tech/documentation/fastembed/fastembed-rerankers/)
- [Cross-Encoder Models](https://huggingface.co/models?pipeline_tag=sentence-similarity&sort=downloads)
- [Reranking in Search](https://qdrant.tech/blog/reranking/)
- [Two-Stage Retrieval](https://qdrant.tech/documentation/concepts/search/#two-stage-retrieval)
