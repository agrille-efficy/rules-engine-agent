"""
Workflow Graph Visualizer
Generates and displays the LangGraph workflow structure.
"""
import sys
from pathlib import Path

# Add Agent to path
sys.path.insert(0, str(Path(__file__).parent))

from Agent.core.graph_builder import WorkflowGraphBuilder


def main():
    """Generate and display workflow graph visualization."""
    print("=" * 80)
    print("WORKFLOW GRAPH VISUALIZATION")
    print("=" * 80)
    
    # Build the graph
    builder = WorkflowGraphBuilder(use_checkpointer=False)
    graph = builder.build()
    
    print("\n✓ Graph compiled successfully!")
    
    # Try to generate visual representation
    try:
        # Get the graph as ASCII/Mermaid
        print("\n" + "=" * 80)
        print("GRAPH STRUCTURE (Mermaid Format)")
        print("=" * 80)
        print("\nYou can paste this into https://mermaid.live for visualization:\n")
        
        # LangGraph's get_graph() returns a drawable representation
        drawable = graph.get_graph()
        mermaid = drawable.draw_mermaid()
        print(mermaid)
        
        # Save to file
        output_file = Path(__file__).parent / "workflow_graph.mmd"
        output_file.write_text(mermaid)
        print(f"\n✓ Saved to: {output_file}")
        
        # Try to generate PNG if graphviz is available
        try:
            png_data = drawable.draw_mermaid_png()
            png_file = Path(__file__).parent / "workflow_graph.png"
            png_file.write_bytes(png_data)
            print(f"✓ PNG saved to: {png_file}")
        except Exception as e:
            print(f"\n⚠ PNG generation not available (install graphviz): {e}")
        
    except Exception as e:
        print(f"\n⚠ Visualization failed: {e}")
        print("\nManual graph structure:")
        print_manual_structure()


def print_manual_structure():
    """Print a manual text representation of the workflow."""
    print("""
    ┌─────────────────┐
    │     START       │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ file_analysis   │  ← Analyzes CSV structure & semantics
    └────────┬────────┘
             │
             ├─► [success] ──────┐
             └─► [error] ────────┼──┐
                                 │  │
                                 ▼  │
                        ┌─────────────────┐
                        │  rag_matching   │  ← Vector search for tables
                        └────────┬────────┘
                                 │
                                 ├─► [success] ──────┐
                                 └─► [error] ────────┼──┐
                                                     │  │
                                                     ▼  │
                                            ┌──────────────────┐
                                            │ table_selection  │  ← Select best table(s)
                                            └────────┬─────────┘
                                                     │
                                                     ├─► [success] ──────┐
                                                     └─► [error] ────────┼──┐
                                                                         │  │
                                                                         ▼  │
                                                                ┌──────────────────┐
                                                                │  field_mapping   │  ← Map columns to fields
                                                                └────────┬─────────┘
                                                                         │
                                                                         ├─► [validation] ──► END
                                                                         ├─► [review] ──────┐
                                                                         ├─► [end] ─────────┼──► END
                                                                         └─► [error] ───────┼──┐
                                                                                            │  │
                                                                                            ▼  │
                                                                                    ┌─────────────┐
                                                                                    │   review    │
                                                                                    └──────┬──────┘
                                                                                           │
                                                                                           ▼
                                    ┌─────────────┐                                     END
                                    │    error    │ ◄──────────────────────────────────────┘
                                    └──────┬──────┘
                                           │
                                           ▼
                                         END
    
    Nodes:
    ------
    • file_analysis    : Reads CSV, analyzes structure, translates columns
    • rag_matching     : Vector search to find candidate database tables  
    • table_selection  : Ranks and selects best matching table(s)
    • field_mapping    : Maps CSV columns to database fields
    • error            : Handles failures and logs errors
    • review           : Flags mappings needing human validation
    
    Routing Logic:
    --------------
    • After each node, conditional routing decides next step
    • Success paths continue to next stage
    • Errors route to error handler
    • Low confidence mappings route to review
    """)


if __name__ == "__main__":
    main()