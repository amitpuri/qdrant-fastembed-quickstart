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

## Demo Overview: Working with LLMs

This demo showcases reranking as a crucial component of production LLM systems, demonstrating how to improve the quality of retrieved context for better LLM responses. Here's what makes this demo essential for high-quality LLM applications:

### What This Demo Does

The demo demonstrates reranking's two-stage approach that:
- **Shows initial retrieval results**: Demonstrates fast, broad search that retrieves many candidates
- **Applies precise reranking**: Shows how specialized models re-score candidates for better precision
- **Compares before/after results**: Illustrates the significant improvement in result quality
- **Demonstrates the pipeline**: Shows the complete two-stage retrieval process

### How This is Useful for LLMs

**1. Production RAG Systems**
- **Improved Context Quality**: Reranking ensures LLMs receive the most relevant context for generation
- **Reduced Hallucinations**: Better context selection reduces the likelihood of LLM hallucinations
- **Higher Response Accuracy**: More precise retrieval leads to more accurate LLM responses
- **Better Source Attribution**: Helps LLMs cite more relevant and accurate sources

**2. LLM Response Quality**
- **Precision Over Recall**: Reranking prioritizes getting the most relevant results in the top positions
- **Context Relevance**: Ensures retrieved documents are highly relevant to the user's query
- **Multi-Signal Integration**: Can combine semantic, lexical, and other signals for better ranking
- **Domain-Specific Optimization**: Can be fine-tuned for specific domains and use cases

**3. Conversational AI and Chatbots**
- **Better Conversation Flow**: Improved context selection leads to more coherent conversations
- **Intent Understanding**: Reranking helps ensure retrieved context matches user intent
- **Multi-Turn Context**: Maintains relevance across conversation turns
- **Personalized Responses**: Better context selection enables more personalized LLM responses

**4. Enterprise LLM Applications**
- **Knowledge Base Search**: Perfect for corporate LLMs searching through company knowledge bases
- **Customer Support**: Ideal for support LLMs that need to find the most relevant help articles
- **Document Q&A**: Excellent for LLMs that answer questions about specific documents
- **Compliance and Legal**: Great for LLMs that need to find precise legal or compliance information

**5. Advanced LLM Architectures**
- **Multi-Stage Retrieval**: Combines fast initial retrieval with precise reranking
- **Hybrid Search Integration**: Can rerank results from multiple retrieval methods
- **Query-Dependent Reranking**: Adapts reranking based on query type and complexity
- **Real-Time Optimization**: Enables real-time improvement of retrieval quality

**6. LLM Performance Optimization**
- **Speed vs Quality Trade-off**: Balances retrieval speed with response quality
- **Resource Management**: Optimizes computational resources for better LLM performance
- **Scalable Solutions**: Provides scalable approaches for production LLM systems
- **Quality Monitoring**: Enables monitoring and improvement of retrieval quality

**7. Specialized LLM Applications**
- **Research Assistants**: Perfect for academic LLMs that need high-precision information retrieval
- **Medical LLMs**: Ideal for healthcare LLMs that need accurate medical information
- **Legal LLMs**: Excellent for legal LLMs that need precise legal document retrieval
- **Technical Support**: Great for technical LLMs that need accurate technical documentation

**8. LLM System Architecture**
- **Pipeline Integration**: Shows how to integrate reranking into existing LLM pipelines
- **Performance Monitoring**: Demonstrates how to monitor and optimize reranking performance
- **A/B Testing**: Enables testing different reranking approaches for LLM applications
- **Continuous Improvement**: Provides framework for continuously improving LLM response quality

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
