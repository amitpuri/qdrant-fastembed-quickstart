# Basic Text Embeddings

This folder demonstrates standard dense text embeddings using FastEmbed.

## Overview

Basic text embeddings are the foundation of semantic search. They convert text into high-dimensional vectors that capture semantic meaning, enabling similarity search and clustering.

## Features

- **Model**: BAAI/bge-small-en-v1.5 (384 dimensions)
- **Type**: Dense embeddings
- **Use Case**: General semantic search
- **Performance**: Fast inference, good accuracy

## Files

- `demo.py` - Interactive demonstration of basic embeddings
- `README.md` - This documentation

## Usage

```bash
python demo.py
```

## What You'll Learn

1. How to load and initialize FastEmbed models
2. How to generate embeddings for text documents
3. How to calculate similarity between embeddings
4. Understanding embedding dimensions and properties
5. When to use dense embeddings

## Key Concepts

### Dense Embeddings
- Convert entire text into a single vector
- Capture semantic meaning
- Good for similarity search
- Standard approach for most NLP tasks

### Similarity Calculation
- Cosine similarity between vectors
- Higher values indicate more similar content
- Range: -1 to 1 (typically 0 to 1 for normalized vectors)

### Model Properties
- **Dimensions**: 384 (BAAI/bge-small-en-v1.5)
- **Type**: Dense vector
- **Distance**: Cosine similarity
- **Language**: English

## Example Output

```
🔹 Basic Text Embeddings Demo
----------------------------------------
Loading FastEmbed model (BAAI/bge-small-en-v1.5)...
✅ Model loaded successfully!

Generating embeddings...
📊 Generated 5 embeddings
📏 Vector dimensions: 384
🔗 Cosine similarity between first two documents: 0.6717

📋 Sample embedding (first 10 dimensions):
   [-0.09479033  0.01007713 -0.03085082  0.02376419  0.00238941]
```

## When to Use

✅ **Good for:**
- General semantic search
- Similarity search and clustering
- Most NLP tasks
- When you need good semantic understanding

❌ **Not ideal for:**
- Exact keyword matching
- When you need term-level control
- Domain-specific terminology matching

## Next Steps

- Try the [miniCOIL demo](../minicoil/) for keyword-aware search
- Explore [SPLADE](../splade/) for sparse embeddings
- Learn about [ColBERT](../colbert/) for fine-grained matching
- See [reranking](../reranking/) for improving results

## Resources

- [FastEmbed Quickstart](https://qdrant.tech/documentation/fastembed/fastembed-quickstart/)
- [BAAI/bge-small-en-v1.5 Model](https://huggingface.co/BAAI/bge-small-en-v1.5)
- [Vector Similarity Search](https://qdrant.tech/documentation/concepts/search/)
