# MCP Server Integration

This folder demonstrates Model Context Protocol (MCP) Server integration with Qdrant.

## Overview

MCP (Model Context Protocol) Server provides standardized access to Qdrant, enabling integration with various AI tools and frameworks. It offers a consistent interface for vector operations across different AI applications.

## Features

- **Standardized API**: Consistent interface for vector operations
- **AI Tool Integration**: Works with various AI assistants and tools
- **Real-time Access**: Live data access and manipulation
- **Cross-platform**: Compatible across different systems
- **Extensible**: Protocol versioning and compatibility

## Files

- `demo.py` - Interactive demonstration of MCP Server concepts
- `README.md` - This documentation

## Usage

```bash
python demo.py
```

## What You'll Learn

1. What MCP Server is and how it works
2. How to integrate with AI tools
3. Available operations and capabilities
4. Setup and configuration
5. Use cases and benefits

## Key Concepts

### Model Context Protocol
- Standardized communication protocol
- Tool discovery and description
- Resource management
- Streaming support
- Error handling and authentication

### AI Tool Integration
- Works with Claude, ChatGPT, and other AI assistants
- Provides vector search capabilities to AI tools
- Enables RAG (Retrieval-Augmented Generation)
- Supports conversational AI systems

### Server Architecture
- Client-server communication
- Tool registration and discovery
- Resource management
- Streaming responses
- Security and authentication

## Available Operations

### Collection Management
- Create and delete collections
- List collections and their properties
- Configure vector parameters
- Manage collection settings

### Vector Operations
- Insert and update vectors
- Batch operations for efficiency
- Delete vectors and points
- Metadata management

### Search and Retrieval
- Similarity search
- Filtered search
- Aggregation queries
- Batch search operations

### Monitoring
- Health checks and status
- Performance metrics
- Error reporting
- Configuration management

## Setup and Installation

### Install MCP Server
```bash
# Install MCP Server for Qdrant
pip install mcp-server-qdrant

# Or install from source
git clone https://github.com/qdrant/mcp-server-qdrant
cd mcp-server-qdrant
pip install -e .
```

### Start MCP Server
```bash
# Start with default settings
mcp-server-qdrant

# Start with custom configuration
mcp-server-qdrant --host localhost --port 6333 --api-key your-key
```

### Client Connection
```python
import mcp

# Connect to MCP Server
client = mcp.Client('localhost', 6333)

# List available tools
tools = client.list_tools()

# Use vector search tool
results = client.call_tool(
    'vector_search',
    {
        'collection': 'documents',
        'query': 'machine learning',
        'limit': 10
    }
)
```

## AI Tool Integration

### Claude Integration
```json
{
  "mcpServers": {
    "qdrant": {
      "command": "mcp-server-qdrant",
      "args": ["--host", "localhost", "--port", "6333"]
    }
  }
}
```

### Custom AI Application
```python
import mcp

class AIVectorSearch:
    def __init__(self):
        self.mcp_client = mcp.Client('localhost', 6333)
    
    def search_documents(self, query: str, limit: int = 10):
        return self.mcp_client.call_tool(
            'vector_search',
            {
                'collection': 'documents',
                'query': query,
                'limit': limit
            }
        )
    
    def get_context_for_rag(self, query: str):
        results = self.search_documents(query, limit=5)
        context = "\n".join([doc['text'] for doc in results])
        return context
```

## Use Cases

### RAG (Retrieval-Augmented Generation)
1. AI assistant receives user query
2. MCP Server searches relevant documents
3. Retrieved context used for response generation
4. More accurate and contextual responses

### Knowledge Base Queries
- Search company documents
- Find relevant information
- Provide citations and sources
- Maintain up-to-date knowledge

### Conversational AI
- Chatbots with document search
- Customer support with knowledge base
- Technical assistance with documentation
- Real-time information retrieval

## Benefits

### For AI Tools
- **Standardized Interface**: Consistent API across tools
- **Real-time Access**: Live data without delays
- **Rich Operations**: Full vector database capabilities
- **Easy Integration**: Simple setup and configuration

### For Developers
- **Protocol Compliance**: Follows MCP standards
- **Extensibility**: Easy to add new features
- **Security**: Built-in authentication and authorization
- **Monitoring**: Health checks and performance metrics

### For Organizations
- **AI Tool Flexibility**: Works with multiple AI platforms
- **Data Consistency**: Single source of truth
- **Scalability**: Handles growing data and usage
- **Maintenance**: Centralized management

## Supported AI Tools

### Anthropic Claude
- Native MCP support
- Easy configuration
- Rich tool descriptions
- Streaming responses

### OpenAI ChatGPT
- Custom tool integration
- Function calling support
- Plugin architecture
- API compatibility

### LangChain
- MCP integration
- Chain composition
- Agent support
- Custom tools

### LlamaIndex
- Data connector support
- Query engine integration
- RAG pipeline support
- Custom retrievers

## Security Considerations

### Authentication
- API key authentication
- Token-based access
- Role-based permissions
- Secure communication

### Data Privacy
- Encrypted connections
- Access logging
- Data isolation
- Compliance support

## Next Steps

- Try the [comparison demo](../comparison/) for method selection
- Explore [Qdrant integration](../qdrant_integration/) for setup
- Learn about [hybrid approaches](../comparison/)
- Set up your own MCP Server

## Resources

- [MCP Specification](https://modelcontextprotocol.io/)
- [Qdrant MCP Server](https://github.com/qdrant/mcp-server-qdrant)
- [MCP Documentation](https://docs.modelcontextprotocol.io/)
- [Claude MCP Integration](https://docs.anthropic.com/claude/mcp)
- [AI Tool Integration Guide](https://qdrant.tech/blog/mcp-server/)
