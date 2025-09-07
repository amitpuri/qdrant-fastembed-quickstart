"""
FastEmbed Comprehensive Demo - Main Menu
A menu-driven application that links to individual FastEmbed capability demos.

This main menu provides easy access to all FastEmbed demonstrations:
- Basic text embeddings
- miniCOIL sparse retrieval
- SPLADE sparse embeddings
- ColBERT multi-vector search
- Reranking
- Qdrant integration
- MCP Server integration
- Method comparison

Each demo is located in its own folder with detailed documentation.
"""

import os
import sys
import subprocess
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class FastEmbedDemo:
    """Main class for FastEmbed demonstrations."""
    
    def __init__(self):
        self.demo_folders = {
            "1": ("01_basic_embeddings", "Basic Text Embeddings (Dense)"),
            "2": ("02_minicoil", "miniCOIL Sparse Retrieval"),
            "3": ("03_splade", "SPLADE Sparse Embeddings"),
            "4": ("04_colbert", "ColBERT Multi-Vector Search"),
            "5": ("05_reranking", "Reranking"),
            "6": ("06_qdrant_integration", "Qdrant Integration Demo"),
            "7": ("07_mcp_server", "MCP Server Integration"),
            "8": ("08_comparison", "Compare All Methods"),
            "9": ("cleanup", "Clean Up Demo Resources"),
            "10": ("exit", "Exit")
        }
        
        # Load Qdrant configuration from environment variables
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY", None)
        self.qdrant_timeout = int(os.getenv("QDRANT_TIMEOUT", "60"))
        self.qdrant_prefer_grpc = os.getenv("QDRANT_PREFER_GRPC", "false").lower() == "true"
    
    def show_menu(self):
        """Display the main menu."""
        print("\n" + "="*70)
        print("🚀 FastEmbed Comprehensive Demo - Main Menu")
        print("="*70)
        print("Each option runs a dedicated demo with detailed explanations:")
        print()
        
        for key, (folder, description) in self.demo_folders.items():
            print(f"{key}. {description}")
        print("="*70)
        print("💡 Each demo is in its own folder with README documentation")
        print("📚 Run individual demos: python <folder>/demo.py")
        print("="*70)
        print(f"🔧 Qdrant Configuration: {self.qdrant_url}")
        if self.qdrant_api_key:
            print(f"🔑 API Key: {'*' * (len(self.qdrant_api_key) - 4) + self.qdrant_api_key[-4:]}")
        print("="*70)
    
    def run_demo(self, folder_name: str, description: str):
        """Run a specific demo from its folder."""
        print(f"\n🔹 {description}")
        print("-" * 50)
        
        demo_path = os.path.join(folder_name, "demo.py")
        
        if not os.path.exists(demo_path):
            print(f"❌ Demo file not found: {demo_path}")
            return
        
        try:
            print(f"🚀 Running demo from {folder_name}/")
            print(f"📚 See {folder_name}/README.md for detailed documentation")
            print(f"🔧 Using Qdrant: {self.qdrant_url}")
            print()
            
            # Set environment variables for the demo
            env = os.environ.copy()
            env["QDRANT_URL"] = self.qdrant_url
            if self.qdrant_api_key:
                env["QDRANT_API_KEY"] = self.qdrant_api_key
            env["QDRANT_TIMEOUT"] = str(self.qdrant_timeout)
            env["QDRANT_PREFER_GRPC"] = str(self.qdrant_prefer_grpc)
            
            # Run the demo
            result = subprocess.run([sys.executable, demo_path], 
                                  capture_output=False, 
                                  text=True,
                                  env=env)
            
            if result.returncode == 0:
                print(f"\n✅ Demo completed successfully!")
            else:
                print(f"\n❌ Demo exited with code {result.returncode}")
                
        except Exception as e:
            print(f"❌ Error running demo: {e}")
        
        print(f"\n📖 For more information, see: {folder_name}/README.md")
    
    def cleanup_collections(self):
        """Clean up all demo resources from Qdrant."""
        print("\n🧹 Clean Up Demo Resources")
        print("-" * 50)
        print(f"🔧 Connecting to Qdrant: {self.qdrant_url}")
        print()
        
        try:
            from qdrant_client import QdrantClient
            
            # Connect to Qdrant
            client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key
            )
            
            # Get all collections
            collections = client.get_collections()
            print(f"📊 Found {len(collections.collections)} collections in Qdrant")
            
            # Define demo collection names (comprehensive list)
            demo_collections = [
                # Basic embeddings demo
                "basic_embeddings_demo",
                "basic_embeddings",
                "dense_embeddings_demo",
                
                # miniCOIL demo
                "minicoil_demo", 
                "minicoil_collection",
                "sparse_minicoil_demo",
                
                # SPLADE demo
                "splade_demo",
                "splade_collection",
                "sparse_splade_demo",
                
                # ColBERT demo
                "colbert_demo",
                "colbert_collection",
                "multivector_demo",
                "colbert_multivector_demo",
                
                # Reranking demo
                "reranking_demo",
                "reranking_collection",
                "rerank_demo",
                
                # Qdrant integration demo
                "fastembed_demo_collection",
                "qdrant_integration_demo",
                "integration_demo",
                
                # MCP server demo
                "mcp_demo_collection",
                "mcp_server_demo",
                "mcp_demo",
                
                # Comparison demo
                "comparison_demo_collection",
                "comparison_demo",
                "all_methods_demo",
                
                # Generic demo patterns
                "demo_collection",
                "test_collection",
                "sample_collection",
                "example_collection",
                "tutorial_collection",
                "quickstart_collection"
            ]
            
            # Find and delete demo collections
            deleted_count = 0
            kept_count = 0
            
            for collection in collections.collections:
                collection_name = collection.name
                
                # Check if it's a demo collection
                is_demo = False
                for demo_pattern in demo_collections:
                    if demo_pattern.lower() in collection_name.lower():
                        is_demo = True
                        break
                
                # Also check for patterns that indicate demo collections
                demo_indicators = [
                    "_demo", "demo_", "_test", "test_", "_sample", "sample_",
                    "_example", "example_", "_tutorial", "tutorial_",
                    "_quickstart", "quickstart_", "_playground", "playground_"
                ]
                
                for indicator in demo_indicators:
                    if indicator in collection_name.lower():
                        is_demo = True
                        break
                
                if is_demo:
                    try:
                        print(f"🗑️  Deleting demo collection: {collection_name}")
                        client.delete_collection(collection_name)
                        deleted_count += 1
                        print(f"   ✅ Deleted successfully")
                    except Exception as e:
                        print(f"   ❌ Failed to delete: {e}")
                else:
                    print(f"ℹ️  Keeping collection: {collection_name}")
                    kept_count += 1
            
            # Additional cleanup: Check for any snapshots or other resources
            print(f"\n🔍 Checking for additional demo resources...")
            
            # Try to get cluster info to see if there are any other resources
            try:
                cluster_info = client.get_cluster_info()
                print(f"   ℹ️  Cluster status: {cluster_info.status}")
            except Exception:
                pass  # Cluster info not available in all Qdrant versions
            
            print(f"\n📊 Cleanup Summary:")
            print(f"   • Demo collections deleted: {deleted_count}")
            print(f"   • User collections kept: {kept_count}")
            print(f"   • Total collections processed: {len(collections.collections)}")
            
            if deleted_count > 0:
                print(f"\n✅ Cleanup completed successfully!")
                print(f"💡 All demo resources have been removed from Qdrant")
                print(f"💡 Your Qdrant instance is now clean and ready for production use")
            else:
                print(f"\nℹ️  No demo resources found to clean up")
                print(f"💡 Your Qdrant instance is already clean")
            
            print(f"\n💡 Tips:")
            print(f"   • Run this cleanup after testing demos")
            print(f"   • Your production collections are safe")
            print(f"   • Demo collections are automatically cleaned up after each demo")
            print(f"   • Use this for a complete reset of demo data")
                
        except ImportError:
            print("❌ Qdrant client not available. Please install: pip install qdrant-client[fastembed]")
        except Exception as e:
            print(f"❌ Cleanup failed: {e}")
            print("💡 Make sure Qdrant is running at the configured URL")
            print("💡 Check the Qdrant Web UI at http://localhost:6333/dashboard")
    
    def show_project_structure(self):
        """Show the project structure and available demos."""
        print("\n📁 Project Structure:")
        print("="*50)
        print("qdrant-fastembed-quickstart/")
        print("├── main.py                    # This main menu")
        print("├── basic_example.py           # Original basic example")
        print("├── requirements.txt           # Dependencies")
        print("├── README.md                  # Main documentation")
        print("├── basic_embeddings/          # Dense embeddings demo")
        print("│   ├── demo.py")
        print("│   └── README.md")
        print("├── minicoil/                  # miniCOIL sparse retrieval")
        print("│   ├── demo.py")
        print("│   └── README.md")
        print("├── splade/                    # SPLADE sparse embeddings")
        print("│   ├── demo.py")
        print("│   └── README.md")
        print("├── colbert/                   # ColBERT multi-vector search")
        print("│   ├── demo.py")
        print("│   └── README.md")
        print("├── reranking/                 # Reranking demo")
        print("│   ├── demo.py")
        print("│   └── README.md")
        print("├── qdrant_integration/        # Qdrant integration")
        print("│   ├── demo.py")
        print("│   └── README.md")
        print("├── mcp_server/                # MCP Server integration")
        print("│   ├── demo.py")
        print("│   └── README.md")
        print("└── comparison/                # Method comparison")
        print("    ├── demo.py")
        print("    └── README.md")
        print()
        print("💡 Each folder contains:")
        print("   • demo.py - Interactive demonstration")
        print("   • README.md - Detailed documentation")
    
    def run(self):
        """Main application loop."""
        while True:
            self.show_menu()
            
            try:
                choice = input("\nEnter your choice (1-10): ").strip()
                
                if choice in self.demo_folders:
                    folder_name, description = self.demo_folders[choice]
                    if folder_name == "cleanup":
                        self.cleanup_collections()
                    elif folder_name == "exit":
                        print("\n👋 Thank you for using FastEmbed Demo!")
                        break
                    else:
                        self.run_demo(folder_name, description)
                elif choice.lower() == "help":
                    self.show_project_structure()
                else:
                    print("❌ Invalid choice. Please enter a number between 1-10.")
                    print("💡 Type 'help' to see project structure")
                
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ An error occurred: {e}")
                input("Press Enter to continue...")


def main():
    """Main entry point."""
    print("🚀 Starting FastEmbed Comprehensive Demo...")
    print("📚 This menu provides access to all FastEmbed capabilities")
    print("🔍 Each demo is in its own folder with detailed documentation")
    
    # Check if .env file exists
    if os.path.exists(".env"):
        print("✅ Found .env file - using custom configuration")
    else:
        print("ℹ️  No .env file found - using default configuration")
        print("💡 Create .env from .env.example for custom settings")
    
    # Run the demo
    demo = FastEmbedDemo()
    demo.run()


if __name__ == "__main__":
    main()