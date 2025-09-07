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

## Demo Overview: Working with LLMs

This demo showcases SPLADE's advanced sparse embedding approach that's particularly powerful for LLM applications requiring both lexical precision and semantic understanding. Here's what makes this demo crucial for sophisticated LLM workflows:

### What This Demo Does

The demo demonstrates SPLADE's sophisticated approach that:
- **Generates learned term weights**: Shows how SPLADE creates importance scores for terms based on training, not just statistical frequency
- **Demonstrates term expansion**: Illustrates how SPLADE can generate weights for related terms not present in the original text
- **Compares with traditional methods**: Highlights why learned weights outperform statistical TF-IDF for LLM applications
- **Shows sparse efficiency**: Demonstrates how only non-zero weights are stored for efficient retrieval

### How This is Useful for LLMs

**1. Advanced RAG Systems**
- **Learned Term Importance**: SPLADE's learned weights better understand which terms are important for retrieval
- **Term Expansion for Better Recall**: Automatically expands queries to include related terms, improving LLM context retrieval
- **Vocabulary Mismatch Handling**: Helps LLMs find relevant documents even when users use different terminology
- **Domain Adaptation**: Learned weights adapt to specific domains, improving retrieval for specialized LLM applications

**2. Technical Documentation LLMs**
- **Scientific Paper Search**: Perfect for LLMs working with research papers and technical documentation
- **Code Documentation**: Excellent for developer-focused LLMs searching through codebases and technical docs
- **Legal Document Analysis**: Ideal for legal LLMs that need precise term matching with semantic understanding
- **Medical Literature Search**: Great for healthcare LLMs working with medical literature and research

**3. LLM Query Understanding**
- **Synonym and Related Term Handling**: Automatically expands user queries to include related concepts
- **Typo Resilience**: Handles out-of-vocabulary tokens gracefully, improving LLM robustness
- **Context-Aware Term Weighting**: Understands which terms are important in specific contexts
- **Multi-Language Support**: Can handle cross-lingual term expansion for multilingual LLM applications

**4. Production LLM Applications**
- **Enterprise Search**: Ideal for corporate LLMs working with technical documentation and knowledge bases
- **Customer Support**: Perfect for support LLMs that need to find relevant documentation quickly
- **Research Assistants**: Excellent for academic LLMs that need to search through large document collections
- **Content Management**: Great for LLMs working with content management systems and documentation

**5. Hybrid LLM Architectures**
- **Multi-Modal Retrieval**: Can be combined with dense embeddings for comprehensive search coverage
- **Query Expansion**: Automatically improves user queries by expanding to related terms
- **Context Enrichment**: Provides richer context for LLM generation through better term understanding
- **Scalable Search**: Efficient sparse representation enables fast retrieval for real-time LLM applications

**6. LLM Response Quality**
- **Better Source Retrieval**: Finds more relevant documents through term expansion and learned weights
- **Reduced Hallucinations**: Provides better context through improved document retrieval
- **Domain-Specific Accuracy**: Learned weights improve accuracy for specialized domains
- **Comprehensive Coverage**: Term expansion ensures LLMs don't miss relevant information

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
