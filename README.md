![AI Generated](https://img.shields.io/badge/AI-Generated-blueviolet?style=for-the-badge&logo=openai&logoColor=white)

# FastEmbed + Qdrant: the retrieval stack, end-to-end

This project demonstrates the complete **FastEmbed + Qdrant retrieval stack** through a modular, well-organized structure. Each FastEmbed method has its own dedicated folder with detailed documentation and interactive demos. Based on the comprehensive [Qdrant FastEmbed documentation](https://qdrant.tech/documentation/fastembed/).

Below is a practical map of **embedding types (dense, sparse, multi-vector)**, **advanced retrieval (miniCOIL, SPLADE, ColBERT)**, and **reranking**—all using the FastEmbed + Qdrant tooling.

---

## 1) Multiple Embedding Types

### A. Dense embeddings (single vector per text)

**Overview:** Encode each text as **one vector**. Great first-stage retriever: fast, compact, easy to scale. FastEmbed focuses on speed (quantized ONNX models, CPU-first).

**Key concepts (quick bullets)**
* **One vector per item** → cosine/dot similarity for k-NN search.
* **Throughput**: ONNX Runtime + quantization → strong CPU performance.
* **Use cases**: semantic search; first pass for multi-stage pipelines.

**Details:** Dense embeddings excel when wording varies (semantic matches) but they can miss exact keywords/IDs. In Qdrant you store these as standard vectors and query via similarity; later, you can fuse with sparse signals (RRF/DBSF) to keep both "meaning" and "must-have words."

### B. Sparse embeddings (lexical/term-aware)

**Overview:** Encode text into a **very high-dimensional sparse vector** (most entries zero). Scores behave like learned keyword search—interpretable and great for exact terms, IDs, formulas, acronyms. Qdrant stores them natively (indices + values).

**Key concepts (quick bullets)**
* **Vocab-indexed** space (e.g., BERT WordPiece IDs).
* **Interpretable**: each non-zero weight maps to a token/term.
* **Hybrid-friendly**: combine with dense results to cover both semantics and exact matches.

**Details:**
* **SPLADE (via FastEmbed):** learns sparse vectors that often **outperform BM25** and remain interpretable. FastEmbed exposes models like `prithivida/Splade_PP_en_v1` (Apache-2.0 variant), returning `(indices, values)` you can store directly in Qdrant. Typical corpora end up with ~tens to low hundreds of non-zeros per item, despite ~30k vocab size.
* **miniCOIL (via FastEmbed/Qdrant):** a lightweight **sparse neural retriever**: think "BM25 that understands meaning." It builds a bag-of-stems weighted by **BM25**, but each term gets a small **semantic embedding** so matches are context-aware (e.g., 'vector' in medicine vs graphics). In Qdrant you enable **IDF modifier** and supply corpus avg length; inference + upload can be handled transparently via the client.

### C. Multi-vector embeddings (late interaction)

**Overview:** Models like **ColBERT** output **one vector per token** (a matrix per text). At query time, they compare **query tokens vs document tokens** using **MaxSim** and aggregate to a relevance score—capturing fine-grained matches (names, entities, snippets).

**Key concepts (quick bullets)**
* **Token-level vectors** (e.g., 96–128 dims each).
* **MaxSim** late-interaction scoring at query time.
* **Trade-off**: much better matching granularity, but more memory/compute; often used as a **reranker** for top-K.

**Details:** Qdrant supports **multivector** collections and **MAX_SIM** comparators, so you can store ColBERT outputs as matrices and score with late interaction natively. The Qdrant docs recommend using ColBERT primarily as a **reranker** for 100–500 dense/sparse candidates in production for speed.

---

## 2) Advanced Retrieval with FastEmbed

### A. miniCOIL

**Overview:** **Sparse neural retrieval** that augments BM25 with term semantics. Best when **exact keyword presence is required**, but you want **context-aware ranking** (e.g., "vector control" in public health vs graphics).

**Important concepts**
* **BM25-based scoring × semantic similarity** between matched terms.
* Needs **IDF** modifier and **avg document length** (BM25 ingredients) when creating the Qdrant collection.

**Details:** The HF card notes miniCOIL creates small (4-dim) **meaning embeddings per stem**, combined into a sparse BoW and **weighted by BM25**—so ranking becomes both term-anchored and disambiguation-aware. Qdrant's example shows end-to-end ingestion and querying.

### B. SPLADE

**Overview:** Learns **sparse vocab-space vectors** that are **efficient and interpretable** and often **beat BM25**. Excellent for large-scale retrieval, logs analysis, tech docs, and anything that benefits from seeing important terms directly.

**Important concepts**
* **Expansion in vocab space**: the model assigns weights to vocab entries a text implies (not just explicit words).
* **Compact at query time**: only tens of non-zeros per query on average; works well with ANN-like sparse indexing.

**Details:** FastEmbed exposes SPLADE++ as **Apache-licensed** models; the tutorial shows listing models, embedding, and getting `(indices, values)` for Qdrant. This makes SPLADE trivial to adopt alongside dense vectors for hybrid search.

### C. ColBERT

**Overview:** **Late-interaction** retriever using token embeddings and **MaxSim** aggregation. Dramatically improves matching **granularity** (entities, numbers, code identifiers), and is well-supported in Qdrant via **MultiVectorConfig**.

**Important concepts**
* **Independent encoding** (documents and queries separately).
* **MaxSim**: for each query token, take the **max similarity** over document tokens; aggregate.
* **Use as reranker** for efficiency on large corpora.

**Details:** FastEmbed provides `LateInteractionTextEmbedding` with ready ColBERT models (e.g., `colbert-ir/colbertv2.0` with dim=128). Qdrant lets you configure **MAX_SIM** as the multivector comparator and store the matrices directly.

---

## 3) Reranking (post-processing that boosts result quality)

**Why rerank?** First-stage retrieval (dense/sparse) is cheap and recall-oriented. **Reranking** spends more compute on a **small candidate set** (e.g., top-100) to sharpen precision. Three common options in this stack:

1. **Cross-encoder rerankers (e.g., Jina Reranker v2 in FastEmbed)**
   * Take **[query, document]** together and output a **relevance score** (0–1).
   * Very accurate, most expensive per pair; ideal as a final pass on small K. FastEmbed documents a full example.

2. **ColBERT as a reranker (late interaction)**
   * Faster than cross-encoders at inference (no joint encoding), still **fine-grained** via MaxSim.
   * Qdrant docs explicitly recommend using ColBERT mainly for reranking (100–500 candidates).

3. **Rank-fusion reranking (no model)**
   * Fuse dense + sparse (and/or late-interaction) result lists with algorithms like **RRF** (and DBSF).
   * **Built into Qdrant** Query API—simple, robust, and cheap; great when scores aren't comparable.

> **Rule of thumb:** Retrieve broadly with **dense + sparse**, **fuse (RRF/DBSF)**, then **rerank top-K** with a **cross-encoder or ColBERT** depending on your latency budget.

---

## Putting it together (suggested patterns)

* **General semantic search with exact-term guarantees:**
  Dense (FastEmbed) **+** SPLADE or miniCOIL → **RRF/DBSF fusion** → optional **cross-encoder** for the top-50.

* **Entity/ID-heavy corpora (APIs, legal, code):**
  First stage: SPLADE or miniCOIL (keyword-anchored). Second stage: **ColBERT rerank** for fine-grained token matches.

* **Latency-sensitive, CPU-only RAG:**
  Dense (FastEmbed ONNX) **+** SPLADE mini-fusion (RRF). Only add cross-encoder/ColBERT rerank when absolutely needed.

---

## (Optional) Minimal code sketch

> *Illustrative only—focus on how pieces fit. See the linked docs for full examples.*

```python
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding

# 1) Models
dense = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")            # example
sparse = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")  # SPLADE
late  = LateInteractionTextEmbedding(model_name="colbert-ir/colbertv2.0")

# 2) Collections
client = QdrantClient(":memory:")

# Dense (single vector)
client.create_collection(
  "docs_dense",
  vectors_config=models.VectorParams(size=dense.embedding_size, distance=models.Distance.COSINE),
)

# Sparse (SPLADE)
client.create_collection(
  "docs_sparse",
  sparse_vectors_config={"splade": models.SparseVectorParams()},
)

# Multi-vector (ColBERT with MAX_SIM)
client.create_collection(
  "docs_colbert",
  vectors_config=models.VectorParams(
      size=late.embedding_size, distance=models.Distance.COSINE,
      multivector_config=models.MultiVectorConfig(
          comparator=models.MultiVectorComparator.MAX_SIM
      ),
  ),
)

# 3) Retrieval + fusion + rerank (pseudo)
dense_hits  = client.query_points("docs_dense", query_vector=dense.embed("query")[0], limit=200)
sparse_vec  = next(sparse.embed(["query"]))
sparse_hits = client.query_points("docs_sparse", query=models.SparseVector(**sparse_vec.model_dump()), using="splade", limit=200)

# Fuse (RRF/DBSF; supported in Qdrant Query API)
# ... then rerank top-K with cross-encoder or ColBERT token-level scoring ...
```

Cites for API capabilities and recommended config: **multivector/MAX_SIM** (ColBERT), **sparse vectors**, **hybrid fusion** (RRF/DBSF), **cross-encoder rerankers**.

---

## What is FastEmbed?

FastEmbed is a lightweight and fast library for generating text embeddings and sparse representations. It's designed to be faster and lighter than other embedding libraries like Transformers and Sentence-Transformers, and is supported and maintained by Qdrant.

## Features

- **Modular Structure**: Each method in its own folder with documentation
- **Multiple Embedding Types**: Dense, sparse, and multi-vector embeddings
- **Advanced Retrieval**: miniCOIL, SPLADE, and ColBERT support
- **Reranking**: Post-processing to improve search results
- **Qdrant Integration**: Seamless vector database integration
- **Interactive Demos**: Each method has its own demo with explanations

## Prerequisites

Before running the examples, make sure you have:

- **Python 3.7+** installed
- **Qdrant server** running locally on http://localhost:6333


## Installation

### 1. Clone this repository:
```bash
git clone <repository-url>
cd qdrant-fastembed-quickstart
```

### 2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Set up Qdrant server:

#### Option 1: Using Docker (Recommended)
```bash
docker run -p 6333:6333 qdrant/qdrant
```

#### Option 2: Using Docker Compose
Create a `docker-compose.yml` file:
```yaml
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - ./qdrant_storage:/qdrant/storage
```

Then run:
```bash
docker-compose up -d
```

#### Option 3: From Source
Follow the [official installation guide](https://qdrant.tech/documentation/quick-start/).

### 4. Configure environment variables (Optional):
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env file with your Qdrant configuration
# QDRANT_URL=http://localhost:6333
# QDRANT_API_KEY=your-api-key-here
```

**Note**: If you don't create a `.env` file, the demos will use default values (localhost:6333).

## Usage

### Main Menu (Recommended)
Run the main menu to access all demos:

```bash
python main.py
```

This launches an interactive menu that runs each demo in its dedicated folder.

### Individual Demos
You can also run individual demos directly:

```bash
# Basic embeddings
python 01_basic_embeddings/demo.py

# miniCOIL sparse retrieval
python 02_minicoil/demo.py

# SPLADE sparse embeddings
python 03_splade/demo.py

# ColBERT multi-vector search
python 04_colbert/demo.py

# Reranking
python 05_reranking/demo.py

# Qdrant integration
python 06_qdrant_integration/demo.py


# Method comparison
python 08_comparison/demo.py
```

### Available Demos

1. **[01 - Basic Text Embeddings](01_basic_embeddings/)** - Standard semantic embeddings
2. **[02 - miniCOIL Sparse Retrieval](02_minicoil/)** - Keyword-aware semantic search
3. **[03 - SPLADE Sparse Embeddings](03_splade/)** - Learned sparse representations
4. **[04 - ColBERT Multi-Vector Search](04_colbert/)** - Fine-grained token matching
5. **[05 - Reranking](05_reranking/)** - Post-processing to improve results
6. **[06 - Qdrant Integration](06_qdrant_integration/)** - Vector database integration
7. **[07 - Model Content Protocol Integration](07_mcp_server/)** - MCP integration
8. **[08 - Method Comparison](08_comparison/)** - Side-by-side comparison

Each demo folder contains:
- `demo.py` - Interactive demonstration
- `README.md` - Detailed documentation and explanations

## Example Output

```
🚀 FastEmbed Comprehensive Demo
============================================================
1. Basic Text Embeddings (Dense)
2. miniCOIL Sparse Retrieval
3. SPLADE Sparse Embeddings
4. ColBERT Multi-Vector Search
5. Reranking
6. Qdrant Integration Demo
7. Compare All Methods
9. Exit
============================================================

Enter your choice (1-8): 1

🔹 Basic Text Embeddings Demo
----------------------------------------
Loading FastEmbed model (BAAI/bge-small-en-v1.5)...
✅ Model loaded successfully!

Generating embeddings...
📊 Generated 3 embeddings
📏 Vector dimensions: 384
🔗 Cosine similarity between first two documents: 0.6717

📋 Sample embedding (first 5 dimensions):
   [-0.09479033  0.01007713 -0.03085082  0.02376419  0.00238941]
```

## Method Comparison

| Method        | Type      | Use Case                    |
|---------------|-----------|-----------------------------|
| Dense (BGE)   | Dense     | General semantic search     |
| miniCOIL      | Sparse    | Keyword + semantic hybrid   |
| SPLADE        | Sparse    | Lexical + semantic hybrid   |
| ColBERT       | Multi     | Fine-grained token matching |
| Reranking     | Post-proc | Improve initial results     |

## Project Structure

```
qdrant-fastembed-quickstart/
├── main.py                    # Main menu interface
├── basic_example.py           # Original basic example
├── requirements.txt           # Python dependencies
├── README.md                  # This main documentation
├── .gitignore                 # Git ignore rules for security and cleanliness
├── .env.example               # Environment configuration template
├── docker-compose.yml         # Qdrant setup
├── 01_basic_embeddings/       # Dense embeddings demo
│   ├── demo.py               # Interactive demonstration
│   └── README.md             # Detailed documentation
├── 02_minicoil/               # miniCOIL sparse retrieval
│   ├── demo.py               # Interactive demonstration
│   └── README.md             # Detailed documentation
├── 03_splade/                 # SPLADE sparse embeddings
│   ├── demo.py               # Interactive demonstration
│   └── README.md             # Detailed documentation
├── 04_colbert/                # ColBERT multi-vector search
│   ├── demo.py               # Interactive demonstration
│   └── README.md             # Detailed documentation
├── 05_reranking/              # Reranking demo
│   ├── demo.py               # Interactive demonstration
│   └── README.md             # Detailed documentation
├── 06_qdrant_integration/     # Qdrant integration
│   ├── demo.py               # Interactive demonstration
│   └── README.md             # Detailed documentation
├── 07_mcp_server/            # Model Context Protocol integration
│   └── README.md             # Detailed documentation
└── 08_comparison/            # Method comparison
    ├── demo.py               # Interactive demonstration
    └── README.md             # Detailed documentation
```

## Key Components

### Main Menu System
The main menu (`main.py`) provides easy access to all FastEmbed demonstrations through a clean, organized interface.

### Modular Demo Structure
Each FastEmbed method has its own dedicated folder containing:
- **Interactive Demo**: `demo.py` with hands-on examples
- **Detailed Documentation**: `README.md` with comprehensive explanations
- **Focused Learning**: Each demo focuses on one specific method

### Available Methods

1. **[01 - Basic Embeddings](01_basic_embeddings/)**: Standard dense text embeddings using BAAI/bge-small-en-v1.5
2. **[02 - miniCOIL](02_minicoil/)**: Sparse neural retrieval combining BM25 with semantic understanding
3. **[03 - SPLADE](03_splade/)**: Sparse lexical and dense embeddings with learned weights
4. **[04 - ColBERT](04_colbert/)**: Multi-vector search with fine-grained token matching
5. **[05 - Reranking](05_reranking/)**: Post-processing to improve search result quality
6. **[06 - Qdrant Integration](06_qdrant_integration/)**: Vector database integration examples
7. **[07 - MCP Server Integration](07_mcp_server/)**: MCP Server Integration
8. **[08 - Comparison](08_comparison/)**: Side-by-side method comparison and selection guide

### Learning Path
1. **Start with 01 - Basic Embeddings** to understand fundamentals
2. **Explore 02 - miniCOIL and 03 - SPLADE** for keyword-aware search
3. **Try 04 - ColBERT** for fine-grained matching
4. **Learn about 05 - Reranking** for improving results
5. **Understand 06 - Qdrant Integration**
6. **Use 08 - Comparison** to choose the right method for your use case

## When to Use Each Method

- **Dense Embeddings**: General purpose semantic search
- **miniCOIL**: When exact keyword matches matter but context is important
- **SPLADE**: When you need both lexical and semantic matching
- **ColBERT**: When fine-grained token-level matching is crucial
- **Reranking**: As a second stage to improve any initial retrieval

## Next Steps

- Explore different embedding models available in FastEmbed
- Integrate with Qdrant vector database for storage and search
- Build semantic search applications with hybrid approaches
- Experiment with different text preprocessing techniques
- Set up a Qdrant instance for full functionality testing

## Resources

### FastEmbed + Qdrant Documentation
- [FastEmbed Documentation](https://qdrant.tech/documentation/fastembed/) - Getting started with FastEmbed
- [FastEmbed SPLADE Guide](https://qdrant.tech/documentation/fastembed/fastembed-splade/) - Working with SPLADE sparse embeddings
- [FastEmbed miniCOIL Guide](https://qdrant.tech/documentation/fastembed/fastembed-minicoil/) - Working with miniCOIL sparse retrieval
- [FastEmbed ColBERT Guide](https://qdrant.tech/documentation/fastembed/fastembed-colbert/) - Working with ColBERT multi-vector search
- [FastEmbed Reranking Guide](https://qdrant.tech/documentation/fastembed/fastembed-rerankers/) - Reranking with FastEmbed
- [Hybrid Search with FastEmbed](https://qdrant.tech/documentation/beginner-tutorials/hybrid-search-fastembed/) - Setup hybrid search

### Qdrant Core Concepts
- [Qdrant Vectors Documentation](https://qdrant.tech/documentation/concepts/vectors/) - Understanding vector storage
- [Qdrant Sparse Vectors](https://qdrant.tech/documentation/concepts/vectors/) - Sparse vector concepts
- [Qdrant Hybrid Queries](https://qdrant.tech/documentation/concepts/hybrid-queries/) - Combining dense and sparse search
- [What is a Sparse Vector?](https://qdrant.tech/articles/sparse-vectors/) - Sparse vector deep dive

### Research Papers & Background
- [ColBERT: Efficient and Effective Passage Search](https://arxiv.org/pdf/2004.12832) - ColBERT research paper
- [miniCOIL: on the Road to Usable Sparse Neural Retrieval](https://qdrant.tech/articles/minicoil/) - miniCOIL background
- [SPLADE with FastEmbed Example](https://qdrant.github.io/fastembed/examples/SPLADE_with_FastEmbed/) - SPLADE implementation example

### Model Resources
- [Qdrant/minicoil-v1 on Hugging Face](https://huggingface.co/Qdrant/minicoil-v1) - miniCOIL model card
- [Sparse Vectors Benchmark](https://github.com/qdrant/sparse-vectors-benchmark) - Performance comparisons

### GitHub Repositories
- [FastEmbed GitHub Repository](https://github.com/qdrant/fastembed) - Source code and issues
- [Qdrant Documentation](https://qdrant.tech/documentation/) - Complete Qdrant documentation
