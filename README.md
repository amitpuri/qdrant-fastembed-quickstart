![AI Generated](https://img.shields.io/badge/AI-Generated-blueviolet?style=for-the-badge&logo=openai&logoColor=white)

# FastEmbed Comprehensive Demo

This project demonstrates all FastEmbed capabilities through a modular, well-organized structure. Each FastEmbed method has its own dedicated folder with detailed documentation and interactive demos. Based on the comprehensive [Qdrant FastEmbed documentation](https://qdrant.tech/documentation/fastembed/).

## What is FastEmbed?

FastEmbed is a lightweight and fast library for generating text embeddings and sparse representations. It's designed to be faster and lighter than other embedding libraries like Transformers and Sentence-Transformers, and is supported and maintained by Qdrant.

## Features

- **Modular Structure**: Each method in its own folder with documentation
- **Multiple Embedding Types**: Dense, sparse, and multi-vector embeddings
- **Advanced Retrieval**: miniCOIL, SPLADE, and ColBERT support
- **Reranking**: Post-processing to improve search results
- **Qdrant Integration**: Seamless vector database integration
- **MCP Server**: Complete Model Context Protocol server for AI tool integration
- **Interactive Demos**: Each method has its own demo with explanations

## Prerequisites

Before running the examples, make sure you have:

- **Python 3.7+** installed
- **Qdrant server** running locally on http://localhost:6333

## Security

This project follows security best practices:

- **No hardcoded credentials**: All sensitive data uses environment variables
- **Comprehensive .gitignore**: Excludes sensitive files, cache, and temporary data
- **API key masking**: Sensitive data is masked when displayed
- **Template configuration**: `.env.example` provides safe configuration templates

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

# MCP Server integration
python 07_mcp_server/demo.py

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
7. **[07 - MCP Server Integration](07_mcp_server/)** - Model Context Protocol support
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
7. MCP Server Integration
8. Compare All Methods
9. Exit
============================================================

Enter your choice (1-9): 1

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
├── 07_mcp_server/             # MCP Server integration
│   ├── mcp_server.py         # Complete MCP server implementation
│   ├── demo.py               # Interactive demonstration
│   ├── mcp_demo.py           # Complete working demonstration
│   └── README.md             # Detailed documentation
└── 08_comparison/             # Method comparison
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
7. **[07 - MCP Server](07_mcp_server/)**: Model Context Protocol server for AI tools
8. **[08 - Comparison](08_comparison/)**: Side-by-side method comparison and selection guide

### Learning Path
1. **Start with 01 - Basic Embeddings** to understand fundamentals
2. **Explore 02 - miniCOIL and 03 - SPLADE** for keyword-aware search
3. **Try 04 - ColBERT** for fine-grained matching
4. **Learn about 05 - Reranking** for improving results
5. **Understand 06 - Qdrant Integration** and 07 - MCP Server
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

- [FastEmbed Documentation](https://qdrant.tech/documentation/fastembed/)
- [miniCOIL Guide](https://qdrant.tech/documentation/fastembed/fastembed-minicoil/)
- [SPLADE Guide](https://qdrant.tech/documentation/fastembed/fastembed-splade/)
- [ColBERT Guide](https://qdrant.tech/documentation/fastembed/fastembed-colbert/)
- [Reranking Guide](https://qdrant.tech/documentation/fastembed/fastembed-rerankers/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [FastEmbed GitHub Repository](https://github.com/qdrant/fastembed)
- [Qdrant MCP Server](https://github.com/qdrant/mcp-server-qdrant)
