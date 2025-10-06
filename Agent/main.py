"""
Main entry point for the Rules Engine Agent.
Provides CLI and programmatic access to the workflow.
"""
import os
import sys
import logging
from pathlib import Path
from typing import Optional

from .core.workflow_engine import WorkflowEngine


def setup_logging(level: str = "INFO"):
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def run_workflow(
    file_path: str,
    user_context: Optional[str] = None,
    table_preference: Optional[str] = None,
    log_level: str = "INFO"
) -> dict:
    """
    Run the complete workflow for a given file.
    
    Args:
        file_path: Path to the CSV file to process
        user_context: Optional context about the data
        table_preference: Optional preferred table name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Final workflow state as dictionary
    """
    # Setup logging
    setup_logging(log_level)
    
    # Validate file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Initialize and run workflow engine
    engine = WorkflowEngine(use_checkpointer=True)
    
    # Visualize workflow structure
    engine.visualize_workflow()
    
    # Execute workflow
    result = engine.run(
        file_path=file_path,
        user_context=user_context,
        table_preference=table_preference
    )
    
    return result


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Rules Engine Agent - Automated file-to-database mapping"
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to the CSV file to process"
    )
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Optional context about the data"
    )
    parser.add_argument(
        "--table",
        type=str,
        default=None,
        help="Optional preferred table name"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    try:
        result = run_workflow(
            file_path=args.file_path,
            user_context=args.context,
            table_preference=args.table,
            log_level=args.log_level
        )
        
        # Print final status
        print("\n" + "=" * 80)
        print("WORKFLOW COMPLETE")
        print("=" * 80)
        print(f"Status: {result.get('workflow_status')}")
        print(f"Final step: {result.get('workflow_step')}")
        
        if result.get('workflow_status') == 'failed':
            print(f"Error: {result.get('last_error')}")
            sys.exit(1)
        elif result.get('workflow_status') == 'requires_review':
            print("Workflow requires human review")
            sys.exit(2)
        else:
            print("Workflow completed successfully")
            sys.exit(0)
            
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
