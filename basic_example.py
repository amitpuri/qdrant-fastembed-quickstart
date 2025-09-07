"""
FastEmbed Quickstart Example
Based on: https://qdrant.tech/documentation/fastembed/fastembed-quickstart/

This example demonstrates how to generate text embeddings using FastEmbed.
"""

from typing import List
import numpy as np
from fastembed import TextEmbedding


def main():
    """Main function demonstrating FastEmbed usage."""
    
    # Add sample data
    documents: List[str] = [
        "FastEmbed is lighter than Transformers & Sentence-Transformers.",
        "FastEmbed is supported by and maintained by Qdrant.",
    ]
    
    print("Sample documents:")
    for i, doc in enumerate(documents, 1):
        print(f"{i}. {doc}")
    print()
    
    # Download and initialize the model
    print("Loading FastEmbed model (BAAI/bge-small-en-v1.5)...")
    embedding_model = TextEmbedding()
    print("The model BAAI/bge-small-en-v1.5 is ready to use.")
    print()
    
    # Generate embeddings for both documents
    print("Generating embeddings...")
    embeddings_generator = embedding_model.embed(documents)
    embeddings_list = list(embeddings_generator)
    
    # Display results
    print(f"Number of documents: {len(embeddings_list)}")
    print(f"Vector dimensions: {len(embeddings_list[0])}")
    print()
    
    # Show detailed information for each document
    for i, (doc, embedding) in enumerate(zip(documents, embeddings_list), 1):
        print(f"Document {i}: {doc}")
        print(f"Vector of type: {type(embedding)} with shape: {embedding.shape}")
        print()
    
    # Visualize embeddings (first few dimensions)
    print("Embeddings (first 10 dimensions):")
    for i, embedding in enumerate(embeddings_list, 1):
        print(f"Document {i}: {embedding[:10]}")
    print()
    
    # Calculate similarity between the two documents
    similarity = np.dot(embeddings_list[0], embeddings_list[1])
    print(f"Cosine similarity between documents: {similarity:.4f}")


if __name__ == "__main__":
    main()
