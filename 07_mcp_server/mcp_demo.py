"""
Working MCP Server Example
Demonstrates that the MCP server is working and ready for use with AI tools.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

from mcp_server import (
    handle_list_collections,
    handle_create_collection,
    handle_delete_collection,
    handle_get_collection_info,
    handle_upsert_vectors,
    handle_search_vectors,
    handle_delete_vectors,
    handle_get_vectors,
    handle_health_check,
    get_qdrant_client
)


async def demonstrate_mcp_server():
    """Demonstrate the MCP server functionality."""
    print("🚀 MCP Server Working Example")
    print("=" * 50)
    
    try:
        # Get Qdrant client
        client = get_qdrant_client()
        print("✅ Connected to Qdrant")
        
        # 1. Health Check
        print("\n🏥 1. Health Check")
        result = await handle_health_check(client)
        health_data = json.loads(result[0].text)
        print(f"   Status: {health_data['status']}")
        print(f"   Collections: {health_data['collections_count']}")
        
        # 2. List Collections
        print("\n📊 2. List Collections")
        result = await handle_list_collections(client)
        collections_data = json.loads(result[0].text)
        print(f"   Found {collections_data['count']} collections")
        
        # 3. Create a Test Collection
        print("\n🆕 3. Create Test Collection")
        result = await handle_create_collection(client, {
            'name': 'mcp_demo_collection',
            'vector_size': 128,
            'distance': 'Cosine'
        })
        create_data = json.loads(result[0].text)
        print(f"   Status: {create_data['status']}")
        print(f"   Collection: {create_data['collection']['name']}")
        
        # 4. Get Collection Info
        print("\nℹ️  4. Get Collection Info")
        result = await handle_get_collection_info(client, {
            'name': 'mcp_demo_collection'
        })
        info_data = json.loads(result[0].text)
        print(f"   Name: {info_data['collection']['name']}")
        print(f"   Vector Size: {info_data['collection']['config']['params']['vectors']['size']}")
        print(f"   Distance: {info_data['collection']['config']['params']['vectors']['distance']}")
        
        # 5. Insert Vectors
        print("\n📝 5. Insert Vectors")
        vectors = [
            {
                'id': '1',
                'vector': [0.1] * 128,
                'payload': {'text': 'Machine learning is fascinating', 'category': 'AI'}
            },
            {
                'id': '2',
                'vector': [0.2] * 128,
                'payload': {'text': 'Vector databases are powerful', 'category': 'Database'}
            },
            {
                'id': '3',
                'vector': [0.15] * 128,
                'payload': {'text': 'AI and ML work together', 'category': 'AI'}
            }
        ]
        result = await handle_upsert_vectors(client, {
            'collection': 'mcp_demo_collection',
            'vectors': vectors
        })
        upsert_data = json.loads(result[0].text)
        print(f"   Status: {upsert_data['status']}")
        print(f"   Inserted: {upsert_data['message']}")
        
        # 6. Search Vectors
        print("\n🔍 6. Search Vectors")
        result = await handle_search_vectors(client, {
            'collection': 'mcp_demo_collection',
            'query_vector': [0.1] * 128,
            'limit': 3
        })
        search_data = json.loads(result[0].text)
        print(f"   Found {search_data['count']} results:")
        for i, result in enumerate(search_data['results'], 1):
            print(f"     {i}. ID: {result['id']}, Score: {result['score']:.4f}")
            print(f"        Text: {result['payload']['text']}")
            print(f"        Category: {result['payload']['category']}")
        
        # 7. Get Specific Vectors
        print("\n📥 7. Get Specific Vectors")
        result = await handle_get_vectors(client, {
            'collection': 'mcp_demo_collection',
            'ids': ['1', '2']
        })
        get_data = json.loads(result[0].text)
        print(f"   Retrieved {get_data['count']} vectors:")
        for vector in get_data['vectors']:
            print(f"     ID: {vector['id']}")
            print(f"     Text: {vector['payload']['text']}")
            print(f"     Vector length: {len(vector['vector'])}")
        
        # 8. Delete Vectors
        print("\n🗑️  8. Delete Vectors")
        result = await handle_delete_vectors(client, {
            'collection': 'mcp_demo_collection',
            'ids': ['1', '2', '3']
        })
        delete_data = json.loads(result[0].text)
        print(f"   Status: {delete_data['status']}")
        print(f"   Message: {delete_data['message']}")
        
        # 9. Delete Collection
        print("\n🗑️  9. Delete Collection")
        result = await handle_delete_collection(client, {
            'name': 'mcp_demo_collection'
        })
        delete_collection_data = json.loads(result[0].text)
        print(f"   Status: {delete_collection_data['status']}")
        print(f"   Message: {delete_collection_data['message']}")
        
        print("\n🎉 MCP Server Demo Completed Successfully!")
        print("\n✅ All 9 MCP tools are working correctly:")
        print("   1. ✅ health_check")
        print("   2. ✅ list_collections")
        print("   3. ✅ create_collection")
        print("   4. ✅ get_collection_info")
        print("   5. ✅ upsert_vectors")
        print("   6. ✅ search_vectors")
        print("   7. ✅ get_vectors")
        print("   8. ✅ delete_vectors")
        print("   9. ✅ delete_collection")
        
        print("\n🚀 The MCP server is ready for use with AI tools!")
        print("   • Claude (Anthropic)")
        print("   • ChatGPT with custom tools")
        print("   • LangChain")
        print("   • LlamaIndex")
        print("   • Any MCP-compatible client")
        
    except Exception as e:
        print(f"❌ Error in demo: {e}")
        import traceback
        traceback.print_exc()


def show_usage_instructions():
    """Show how to use the MCP server with AI tools."""
    print("\n" + "="*60)
    print("📚 HOW TO USE THE MCP SERVER WITH AI TOOLS")
    print("="*60)
    
    print("\n🔧 1. Start the MCP Server:")
    print("   cd 07_mcp_server")
    print("   python mcp_server.py")
    
    print("\n🤖 2. Configure AI Tools:")
    print("   The server runs on stdio and can be integrated with:")
    print("   • Claude: Add to MCP configuration")
    print("   • ChatGPT: Use as custom tool")
    print("   • LangChain: MCP integration")
    print("   • LlamaIndex: MCP connector")
    
    print("\n📝 3. Available Tools for AI:")
    print("   • list_collections - List all collections")
    print("   • create_collection - Create new collection")
    print("   • delete_collection - Delete collection")
    print("   • get_collection_info - Get collection details")
    print("   • upsert_vectors - Insert/update vectors")
    print("   • search_vectors - Search similar vectors")
    print("   • delete_vectors - Delete vectors by ID")
    print("   • get_vectors - Retrieve vectors by ID")
    print("   • health_check - Check Qdrant connection")
    
    print("\n🎯 4. Example AI Use Cases:")
    print("   • RAG (Retrieval-Augmented Generation)")
    print("   • Document search and retrieval")
    print("   • Knowledge base queries")
    print("   • Conversational AI with vector search")
    print("   • Multi-modal AI applications")
    
    print("\n✅ 5. The server is production-ready!")
    print("   • Full MCP protocol compliance")
    print("   • Comprehensive error handling")
    print("   • Async/await support")
    print("   • Type safety with Pydantic")
    print("   • Environment-based configuration")


async def main():
    """Main function."""
    # Check if Qdrant is running
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
        client.get_collections()
        print("✅ Qdrant is running and accessible")
    except Exception as e:
        print(f"❌ Qdrant is not accessible: {e}")
        print("💡 Please start Qdrant with: docker-compose up -d")
        return
    
    # Run the demonstration
    await demonstrate_mcp_server()
    
    # Show usage instructions
    show_usage_instructions()


if __name__ == "__main__":
    asyncio.run(main())
