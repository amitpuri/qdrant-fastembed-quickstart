# Qdrant Integration

This folder demonstrates Qdrant vector database integration with FastEmbed.

## Overview

Qdrant is a high-performance vector database that works seamlessly with FastEmbed. It provides efficient storage, indexing, and retrieval of vector embeddings with support for dense, sparse, and multi-vector representations.

## Features

- **Vector Types**: Dense, sparse, and multi-vector support
- **Performance**: High-speed vector search
- **Scalability**: Horizontal scaling and clustering
- **APIs**: REST and gRPC interfaces
- **Management**: Web UI for administration

## Files

- `demo.py` - Interactive demonstration of Qdrant integration concepts
- `README.md` - This documentation

## Usage

```bash
python demo.py
```

## What You'll Learn

1. How Qdrant integrates with FastEmbed
2. Collection configuration for different vector types
3. Document upload with automatic inference
4. Querying and retrieval patterns
5. Performance and scalability considerations

## Key Concepts

### Seamless Integration
- Automatic model downloading and inference
- No need to manually generate embeddings
- Built-in support for all FastEmbed models
- Optimized for FastEmbed workflows

### Vector Types Support
- **Dense Vectors**: From TextEmbedding models
- **Sparse Vectors**: From SPLADE, miniCOIL
- **Multi-Vectors**: From ColBERT
- **Hybrid**: Combinations of different types

### Performance Features
- High-speed similarity search
- Efficient indexing algorithms
- Horizontal scaling
- Real-time updates
- Advanced filtering

## Collection Configuration

### Dense Vectors
```python
client.create_collection(
    collection_name="documents",
    vectors_config={
        "dense": models.VectorParams(
            size=384,  # BAAI/bge-small-en-v1.5
            distance=models.Distance.COSINE
        )
    }
)
```

### Sparse Vectors
```python
client.create_collection(
    collection_name="sparse_docs",
    sparse_vectors_config={
        "minicoil": models.SparseVectorParams(
            modifier=models.Modifier.IDF
        )
    }
)
```

### Multi-Vectors
```python
client.create_collection(
    collection_name="multi_docs",
    vectors_config={
        "colbert": models.VectorParams(
            size=384,
            distance=models.Distance.COSINE
        )
    }
)
```

## Document Upload

### With Automatic Inference
```python
client.upsert(
    collection_name="documents",
    points=[
        models.PointStruct(
            id=1,
            payload={"text": "FastEmbed is lightweight"},
            vector={
                "dense": models.Document(
                    text="FastEmbed is lightweight",
                    model="BAAI/bge-small-en-v1.5"
                )
            }
        )
    ]
)
```

### With Sparse Vectors
```python
client.upsert(
    collection_name="sparse_docs",
    points=[
        models.PointStruct(
            id=1,
            payload={"text": "Vector search technology"},
            vector={
                "minicoil": models.Document(
                    text="Vector search technology",
                    model="Qdrant/minicoil-v1",
                    options={"avg_len": avg_documents_length}
                )
            }
        )
    ]
)
```

## Querying

### Dense Vector Search
```python
results = client.query_points(
    collection_name="documents",
    query=models.Document(
        text="vector search technology",
        model="BAAI/bge-small-en-v1.5"
    ),
    using="dense",
    limit=10
)
```

### Sparse Vector Search
```python
results = client.query_points(
    collection_name="sparse_docs",
    query=models.Document(
        text="vector search technology",
        model="Qdrant/minicoil-v1"
    ),
    using="minicoil",
    limit=10
)
```

### With Filtering
```python
results = client.query_points(
    collection_name="documents",
    query=models.Document(
        text="machine learning",
        model="BAAI/bge-small-en-v1.5"
    ),
    using="dense",
    limit=10,
    query_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="category",
                match=models.MatchValue(value="technology")
            )
        ]
    )
)
```

## Deployment Options

### Local Development
```bash
# Using Docker
docker run -p 6333:6333 qdrant/qdrant

# Using Docker Compose
docker-compose up -d
```

### Qdrant Cloud
```python
# Connect to Qdrant Cloud
client = QdrantClient(
    url="https://your-cluster.qdrant.tech",
    api_key="your-api-key"
)
```

### Self-Hosted
```python
# Connect to self-hosted instance
client = QdrantClient(
    host="your-qdrant-host",
    port=6333
)
```

## Performance Features

### Distance Metrics
- **Cosine**: Most common for text embeddings
- **Dot Product**: For normalized vectors
- **Euclidean**: For general similarity

### Indexing
- **HNSW**: Hierarchical Navigable Small World
- **IVF**: Inverted File Index
- **Auto-tuning**: Automatic parameter optimization

### Scaling
- **Horizontal**: Multiple nodes
- **Sharding**: Data distribution
- **Replication**: High availability

## Web UI

Access the Qdrant Web UI at `http://localhost:6333/dashboard` for:
- Collection management
- Data visualization
- Query testing
- Performance monitoring
- Configuration management

## Use Cases

✅ **Perfect for:**
- Semantic search applications
- Recommendation systems
- Question-answering systems
- Document retrieval
- Image and multimedia search
- Chatbots and conversational AI
- Knowledge graphs and RAG

## Advantages

- **High Performance**: Optimized for vector search
- **Multiple APIs**: REST and gRPC
- **Rich Filtering**: Complex query capabilities
- **Payload Storage**: Metadata with vectors
- **Horizontal Scaling**: Cluster support
- **Real-time Updates**: Live data synchronization
- **Web UI**: Easy management interface

## Next Steps

- Try the [MCP Server demo](../mcp_server/) for AI tool integration
- Explore [comparison demo](../comparison/) for method selection
- Learn about [hybrid approaches](../comparison/)
- Set up your own Qdrant instance

## Resources

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [FastEmbed Integration](https://qdrant.tech/documentation/fastembed/)
- [Qdrant Cloud](https://cloud.qdrant.io/)
- [Docker Setup](https://qdrant.tech/documentation/quick-start/)
- [Performance Tuning](https://qdrant.tech/documentation/guides/optimize-performance/)
