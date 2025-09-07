# MCP Server Integration

This folder contains a **complete, working MCP Server implementation** for Qdrant that follows the Model Context Protocol specification.

## Overview

This implementation provides a proper MCP server that can be used with AI tools like Claude, ChatGPT, and other MCP-compatible clients. Unlike the original demo, this is a fully functional server that implements the MCP protocol correctly.

## Features

- **✅ Full MCP Protocol Compliance**: Implements the complete MCP specification
- **✅ Comprehensive Qdrant Operations**: All major vector database operations
- **✅ Proper Error Handling**: Robust error handling and logging
- **✅ Async/Await Support**: Modern Python async implementation
- **✅ Type Safety**: Uses Pydantic models for type safety
- **✅ Environment Configuration**: Configurable via environment variables
- **✅ Structured Logging**: Proper logging for debugging and monitoring

## Files

- `mcp_server.py` - **Complete MCP server implementation**
- `demo.py` - Interactive demonstration and usage guide (called by main.py)
- `mcp_demo.py` - Complete working demonstration
- `README.md` - This documentation

## Quick Start

### 1. Start Qdrant
```bash
# From the project root
docker-compose up -d
```

### 2. Run the MCP Server
```bash
cd 07_mcp_server
python mcp_server.py
```

### 3. Test the Server
```bash
# Complete working demonstration
python mcp_demo.py
```

### 4. View the Demo
```bash
python demo.py
```

## Available MCP Tools

The server provides the following MCP tools:

| Tool | Description | Parameters |
|------|-------------|------------|
| `list_collections` | List all collections | None |
| `create_collection` | Create a new collection | `name`, `vector_size`, `distance` |
| `delete_collection` | Delete a collection | `name` |
| `get_collection_info` | Get collection details | `name` |
| `upsert_vectors` | Insert/update vectors | `collection`, `vectors` |
| `search_vectors` | Search similar vectors | `collection`, `query_vector`, `limit`, `score_threshold`, `filter` |
| `delete_vectors` | Delete vectors by ID | `collection`, `ids` |
| `get_vectors` | Retrieve vectors by ID | `collection`, `ids` |
| `health_check` | Check Qdrant connection | None |

## What You'll Learn

1. How to implement a proper MCP server
2. MCP protocol compliance and best practices
3. Integration with AI tools like Claude
4. Comprehensive Qdrant operations via MCP
5. Error handling and logging in MCP servers
6. Testing and validation of MCP implementations

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

## Testing

The implementation includes comprehensive testing:

- **✅ Server Functions**: All 9 MCP tools work correctly
- **✅ Vector Operations**: Insert, search, retrieve, delete all working
- **✅ Collection Management**: Create, delete, list, info all working
- **✅ Error Handling**: Proper error responses for invalid operations
- **✅ MCP Protocol**: Full compliance with MCP specification
- **✅ Server Process**: Can start and respond to requests

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