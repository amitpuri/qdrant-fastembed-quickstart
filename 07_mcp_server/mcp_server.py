"""
MCP Server Implementation for Qdrant
A proper Model Context Protocol (MCP) server that provides Qdrant vector database operations.

This implementation follows the MCP specification and provides tools for:
- Collection management
- Vector operations (insert, search, update, delete)
- Metadata operations
- Health monitoring
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4

import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
server = Server("qdrant-mcp-server")

# Global Qdrant client
qdrant_client: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    """Get or create Qdrant client."""
    global qdrant_client
    if qdrant_client is None:
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key
        )
        logger.info(f"Connected to Qdrant at {qdrant_url}")
    
    return qdrant_client


@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """List available MCP tools."""
    return [
        types.Tool(
            name="list_collections",
            description="List all collections in the Qdrant database",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="create_collection",
            description="Create a new collection in Qdrant",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the collection to create"
                    },
                    "vector_size": {
                        "type": "integer",
                        "description": "Size of the vectors in this collection"
                    },
                    "distance": {
                        "type": "string",
                        "enum": ["Cosine", "Euclid", "Dot"],
                        "description": "Distance metric for vector similarity",
                        "default": "Cosine"
                    }
                },
                "required": ["name", "vector_size"]
            }
        ),
        types.Tool(
            name="delete_collection",
            description="Delete a collection from Qdrant",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the collection to delete"
                    }
                },
                "required": ["name"]
            }
        ),
        types.Tool(
            name="get_collection_info",
            description="Get information about a specific collection",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the collection"
                    }
                },
                "required": ["name"]
            }
        ),
        types.Tool(
            name="upsert_vectors",
            description="Insert or update vectors in a collection",
            inputSchema={
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": "Name of the collection"
                    },
                    "vectors": {
                        "type": "array",
                        "description": "List of vectors to upsert",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "description": "Unique identifier for the vector"
                                },
                                "vector": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "description": "Vector data"
                                },
                                "payload": {
                                    "type": "object",
                                    "description": "Metadata payload"
                                }
                            },
                            "required": ["id", "vector"]
                        }
                    }
                },
                "required": ["collection", "vectors"]
            }
        ),
        types.Tool(
            name="search_vectors",
            description="Search for similar vectors in a collection",
            inputSchema={
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": "Name of the collection to search"
                    },
                    "query_vector": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Query vector for similarity search"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10
                    },
                    "score_threshold": {
                        "type": "number",
                        "description": "Minimum similarity score threshold",
                        "default": 0.0
                    },
                    "filter": {
                        "type": "object",
                        "description": "Filter conditions for the search"
                    }
                },
                "required": ["collection", "query_vector"]
            }
        ),
        types.Tool(
            name="delete_vectors",
            description="Delete vectors from a collection",
            inputSchema={
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": "Name of the collection"
                    },
                    "ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of vector IDs to delete"
                    }
                },
                "required": ["collection", "ids"]
            }
        ),
        types.Tool(
            name="get_vectors",
            description="Retrieve vectors by their IDs",
            inputSchema={
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": "Name of the collection"
                    },
                    "ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of vector IDs to retrieve"
                    }
                },
                "required": ["collection", "ids"]
            }
        ),
        types.Tool(
            name="health_check",
            description="Check the health status of the Qdrant connection",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle tool calls."""
    try:
        client = get_qdrant_client()
        
        if name == "list_collections":
            return await handle_list_collections(client)
        elif name == "create_collection":
            return await handle_create_collection(client, arguments)
        elif name == "delete_collection":
            return await handle_delete_collection(client, arguments)
        elif name == "get_collection_info":
            return await handle_get_collection_info(client, arguments)
        elif name == "upsert_vectors":
            return await handle_upsert_vectors(client, arguments)
        elif name == "search_vectors":
            return await handle_search_vectors(client, arguments)
        elif name == "delete_vectors":
            return await handle_delete_vectors(client, arguments)
        elif name == "get_vectors":
            return await handle_get_vectors(client, arguments)
        elif name == "health_check":
            return await handle_health_check(client)
        else:
            return [types.TextContent(
                type="text",
                text=f"Unknown tool: {name}"
            )]
    
    except Exception as e:
        logger.error(f"Error handling tool {name}: {e}")
        return [types.TextContent(
            type="text",
            text=f"Error: {str(e)}"
        )]


async def handle_list_collections(client: QdrantClient) -> List[types.TextContent]:
    """List all collections."""
    try:
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        result = {
            "collections": collection_names,
            "count": len(collection_names)
        }
        
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"Error listing collections: {str(e)}"
        )]


async def handle_create_collection(client: QdrantClient, arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Create a new collection."""
    try:
        name = arguments["name"]
        vector_size = arguments["vector_size"]
        distance = arguments.get("distance", "Cosine")
        
        # Map distance string to Qdrant distance enum
        distance_map = {
            "Cosine": models.Distance.COSINE,
            "Euclid": models.Distance.EUCLID,
            "Dot": models.Distance.DOT
        }
        
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=distance_map[distance]
            )
        )
        
        result = {
            "status": "success",
            "message": f"Collection '{name}' created successfully",
            "collection": {
                "name": name,
                "vector_size": vector_size,
                "distance": distance
            }
        }
        
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"Error creating collection: {str(e)}"
        )]


async def handle_delete_collection(client: QdrantClient, arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Delete a collection."""
    try:
        name = arguments["name"]
        client.delete_collection(name)
        
        result = {
            "status": "success",
            "message": f"Collection '{name}' deleted successfully"
        }
        
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"Error deleting collection: {str(e)}"
        )]


async def handle_get_collection_info(client: QdrantClient, arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Get collection information."""
    try:
        name = arguments["name"]
        info = client.get_collection(name)
        
        result = {
            "collection": {
                "name": name,
                "status": info.status,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "config": {
                    "params": {
                        "vectors": {
                            "size": info.config.params.vectors.size,
                            "distance": info.config.params.vectors.distance
                        }
                    }
                }
            }
        }
        
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"Error getting collection info: {str(e)}"
        )]


async def handle_upsert_vectors(client: QdrantClient, arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Upsert vectors into a collection."""
    try:
        collection = arguments["collection"]
        vectors_data = arguments["vectors"]
        
        points = []
        for vector_data in vectors_data:
            # Convert string IDs to integers if they are numeric
            point_id = vector_data["id"]
            if isinstance(point_id, str) and point_id.isdigit():
                point_id = int(point_id)
            
            point = models.PointStruct(
                id=point_id,
                vector=vector_data["vector"],
                payload=vector_data.get("payload", {})
            )
            points.append(point)
        
        operation_info = client.upsert(
            collection_name=collection,
            points=points
        )
        
        result = {
            "status": "success",
            "message": f"Upserted {len(points)} vectors to collection '{collection}'",
            "operation_id": operation_info.operation_id,
            "status_info": operation_info.status
        }
        
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"Error upserting vectors: {str(e)}"
        )]


async def handle_search_vectors(client: QdrantClient, arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Search for similar vectors."""
    try:
        collection = arguments["collection"]
        query_vector = arguments["query_vector"]
        limit = arguments.get("limit", 10)
        score_threshold = arguments.get("score_threshold", 0.0)
        filter_conditions = arguments.get("filter")
        
        # Build search parameters
        search_params = {
            "collection_name": collection,
            "query_vector": query_vector,
            "limit": limit,
            "score_threshold": score_threshold
        }
        
        if filter_conditions:
            search_params["query_filter"] = models.Filter(**filter_conditions)
        
        search_results = client.search(**search_params)
        
        # Format results
        results = []
        for result in search_results:
            results.append({
                "id": result.id,
                "score": result.score,
                "payload": result.payload
            })
        
        response = {
            "status": "success",
            "query": {
                "collection": collection,
                "limit": limit,
                "score_threshold": score_threshold
            },
            "results": results,
            "count": len(results)
        }
        
        return [types.TextContent(
            type="text",
            text=json.dumps(response, indent=2)
        )]
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"Error searching vectors: {str(e)}"
        )]


async def handle_delete_vectors(client: QdrantClient, arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Delete vectors by IDs."""
    try:
        collection = arguments["collection"]
        ids = arguments["ids"]
        
        # Convert string IDs to integers if they are numeric
        converted_ids = []
        for point_id in ids:
            if isinstance(point_id, str) and point_id.isdigit():
                converted_ids.append(int(point_id))
            else:
                converted_ids.append(point_id)
        
        operation_info = client.delete(
            collection_name=collection,
            points_selector=models.PointIdsList(points=converted_ids)
        )
        
        result = {
            "status": "success",
            "message": f"Deleted {len(ids)} vectors from collection '{collection}'",
            "operation_id": operation_info.operation_id,
            "status_info": operation_info.status
        }
        
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"Error deleting vectors: {str(e)}"
        )]


async def handle_get_vectors(client: QdrantClient, arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Get vectors by IDs."""
    try:
        collection = arguments["collection"]
        ids = arguments["ids"]
        
        # Convert string IDs to integers if they are numeric
        converted_ids = []
        for point_id in ids:
            if isinstance(point_id, str) and point_id.isdigit():
                converted_ids.append(int(point_id))
            else:
                converted_ids.append(point_id)
        
        points = client.retrieve(
            collection_name=collection,
            ids=converted_ids,
            with_vectors=True,
            with_payload=True
        )
        
        results = []
        for point in points:
            results.append({
                "id": point.id,
                "vector": point.vector,
                "payload": point.payload
            })
        
        response = {
            "status": "success",
            "vectors": results,
            "count": len(results)
        }
        
        return [types.TextContent(
            type="text",
            text=json.dumps(response, indent=2)
        )]
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"Error getting vectors: {str(e)}"
        )]


async def handle_health_check(client: QdrantClient) -> List[types.TextContent]:
    """Check Qdrant health."""
    try:
        # Try to get collections to test connection
        collections = client.get_collections()
        
        result = {
            "status": "healthy",
            "message": "Qdrant connection is working",
            "collections_count": len(collections.collections),
            "timestamp": asyncio.get_event_loop().time()
        }
        
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    except Exception as e:
        result = {
            "status": "unhealthy",
            "message": f"Qdrant connection failed: {str(e)}",
            "timestamp": asyncio.get_event_loop().time()
        }
        
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]


async def main():
    """Main function to run the MCP server."""
    # Initialize Qdrant client
    try:
        get_qdrant_client()
        logger.info("MCP Server for Qdrant initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant client: {e}")
        return
    
    # Run the server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="qdrant-mcp-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
