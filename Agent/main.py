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
from .models.validators import PipelineInput, sanitize_for_logging


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
    log_level: str = "INFO",
    max_file_size_mb: int = 100
) -> dict:
    """
    Run the complete workflow for a given file with input validation.
    
    Args:
        file_path: Path to the CSV file to process
        user_context: Optional context about the data (will be sanitized)
        table_preference: Optional preferred table name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        max_file_size_mb: Maximum file size in MB (default 100)
        
    Returns:
        Final workflow state as dictionary
        
    Raises:
        ValueError: If inputs are invalid or contain malicious patterns
        FileNotFoundError: If file doesn't exist
    """
    # Setup logging
    setup_logging(log_level)
    
    # Get workspace root for validation
    workspace_root = str(Path.cwd())
    
    # Validate all inputs using Pydantic
    try:
        validated_input = PipelineInput(
            file_path=file_path,
            user_context=user_context,
            workspace_root=workspace_root,
            max_file_size_mb=max_file_size_mb
        )
        
        # Use validated inputs
        safe_file_path = validated_input.file_path
        safe_user_context = validated_input.user_context
        
        logging.info(f"✅ Input validation passed")
        logging.info(f"   File: {sanitize_for_logging(safe_file_path, 80)}")
        if safe_user_context:
            logging.info(f"   Context: {sanitize_for_logging(safe_user_context, 80)}")
        
    except ValueError as e:
        logging.error(f"❌ Input validation failed: {e}")
        raise ValueError(f"Invalid input: {e}")
    
    # Initialize and run workflow engine
    engine = WorkflowEngine(use_checkpointer=True)
    
    # Visualize workflow structure
    # engine.visualize_workflow()
    
    # Execute workflow with validated inputs
    result = engine.run(
        file_path=safe_file_path,
        user_context=safe_user_context,
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
    parser.add_argument(
        "--max-file-size",
        type=int,
        default=100,
        help="Maximum file size in MB (default: 100)"
    )
    
    args = parser.parse_args()
    
    try:
        result = run_workflow(
            file_path=args.file_path,
            user_context=args.context,
            table_preference=args.table,
            log_level=args.log_level,
            max_file_size_mb=args.max_file_size
        )
        
        # Display comprehensive results
        print("\n" + "=" * 100)
        print("📊 WORKFLOW SUMMARY")
        print("=" * 100)

        status = result.get('workflow_status')
        print(f"\nStatus: {status}")

        # Display file analysis
        file_analysis = result.get('file_analysis_result')
        if file_analysis:
            print(f"\n📄 FILE: {file_analysis.structure.file_name}")
            print(f"   Rows: {file_analysis.structure.total_rows}")
            print(f"   Columns: {file_analysis.structure.total_columns}")

        # Display RAG matching results
        rag_result = result.get('rag_match_result')
        if rag_result:
            print(f"\n🔍 RAG MATCHES: {len(rag_result.matched_tables)} candidate tables found")
            print(f"\n   Top 5 Candidates:")
            for i, match in enumerate(rag_result.matched_tables[:5], 1):
                print(f"   {i}. {match.table_name:30s} (score: {match.similarity_score:.3f}, confidence: {match.confidence})")

        opportunity_match = next((m for m in rag_result.matched_tables if m.table_name == "Opportunity"), None)
        if opportunity_match:
            print("\n" + "=" * 100)
            print("📋 OPPORTUNITY TABLE METADATA")
            print("=" * 100)
            print(opportunity_match)
            
        # print(f"\nResult:\n\n {result} \n")
        if result.get('field_mapping_result'):

            mapping = result['field_mapping_result']

            print("\n" + "=" * 100)
            print("🎯 FIELD MAPPINGS")
            print("=" * 100)
            print("Import file fields                        → Target database fields                  Confidence      Type")
            print("-" * 100)
            for m in mapping.mappings:
                # Safely coerce confidence to float and handle None or string values
                raw_conf = getattr(m, 'confidence_score', 0.0)
                try:
                    conf = float(raw_conf) if raw_conf is not None else 0.0
                except Exception:
                    try:
                        s = str(raw_conf).strip()
                        if s.endswith('%'):
                            conf = float(s.strip('%')) / 100.0
                        else:
                            conf = float(s)
                    except Exception:
                        conf = 0.0
                # Clamp between 0 and 1
                conf = max(0.0, min(1.0, conf))
                confidence_bat = "█" * int(conf * 10)
                conf_display = f"{conf:.2f}"
                print(f"{m.source_column:<40} → {m.target_column or 'None':<40} {conf_display:>6} {confidence_bat:<10} {m.match_type}")
            
            #Unmapped columns
            # unmapped = [m for m in mapping.mappings if m.match_type == 'UNMAPPED']
            # if unmapped:
            #     print("\n" + "=" * 100)
            #     print("❌ UNMAPPED COLUMNS")
            #     print("=" * 100)
            #     for m in unmapped:
            #         print(f"{m.source_column}")
            # print(f"\nMAPPING RESULT:\n\n {mapping} \n")
            
            if hasattr(mapping, 'table_mappings'):
                # Multi-table result
                print("\n" + "=" * 100)
                print("🎯 MULTI-TABLE MAPPING RESULTS")
                print("=" * 100)
                
                total = sum(len(tm.mappings) for tm in mapping.table_mappings)
                print(f"\n📈 OVERALL STATISTICS:")
                print(f"   Total Source Columns: {mapping.total_source_columns}")
                print(f"   Mapped Columns: {total}")
                print(f"   Unmapped Columns: {len(mapping.unmapped_columns)}")
                print(f"   Coverage: {mapping.overall_coverage:.1f}%")
                print(f"   Overall Confidence: {mapping.overall_confidence}")
                print(f"   Valid: {'✅ Yes' if mapping.is_valid else '❌ No'}")
                print(f"   Requires Review: {'⚠️  Yes' if mapping.requires_review else '✅ No'}")
                print(f"   Tables Used: {len(mapping.table_mappings)}")
                
                # Display detailed mappings by table
                print("\n" + "=" * 100)
                print("📋 DETAILED MAPPINGS BY TABLE")
                print("=" * 100)
                
                for table_idx, table_mapping in enumerate(mapping.table_mappings, 1):
                    print(f"\n{'=' * 100}")
                    print(f"TABLE #{table_idx}: {table_mapping.table_name} ({table_mapping.table_type})")
                    print(f"{'=' * 100}")
                    print(f"Insertion Order: {table_mapping.insertion_order}")
                    print(f"Columns Mapped: {len(table_mapping.mappings)}")
                    print(f"Confidence Score: {table_mapping.confidence_score:.2f}")
                    print(f"Coverage: {table_mapping.validation.mapping_coverage_percent:.1f}%")
                    print(f"Confidence Level: {table_mapping.validation.confidence_level}")
                    
                    print(f"\n{'─' * 100}")
                    print(f"{'SOURCE COLUMN':<40} {'TARGET FIELD':<40} {'CONFIDENCE':<12} {'TYPE'}")
                    print(f"{'─' * 100}")
                    
                    for mapping_item in table_mapping.mappings:
                        confidence_bar = "█" * int(mapping_item.confidence_score * 10)
                        print(f"{mapping_item.source_column:<40} {mapping_item.target_column:<40} {mapping_item.confidence_score:.2f} {confidence_bar:<10} {mapping_item.match_type}")
                    
                    # Show validation issues/warnings
                    if table_mapping.validation.issues:
                        print(f"\n⚠️  ISSUES:")
                        for issue in table_mapping.validation.issues:
                            print(f"   - {issue}")
                    
                    if table_mapping.validation.warnings:
                        print(f"\n⚠️  WARNINGS:")
                        for warning in table_mapping.validation.warnings[:3]:
                            print(f"   - {warning}")
                
                # Display unmapped columns
                if mapping.unmapped_columns:
                    print(f"\n{'=' * 100}")
                    print(f"❌ UNMAPPED COLUMNS ({len(mapping.unmapped_columns)})")
                    print(f"{'=' * 100}")
                    
                    for col in mapping.unmapped_columns:
                        # Try to find why it wasn't mapped
                        col_data = next((c for c in file_analysis.columns if c.name == col), None) if file_analysis else None
                        if col_data:
                            print(f"\n   • {col}")
                            print(f"     English: {col_data.english_name if col_data.english_name else col}")
                            print(f"     Type: {col_data.data_type}")
                        else:
                            print(f"\n   • {col}")
                
                # Analysis & Recommendations
                print("\n" + "=" * 100)
                print("💡 ANALYSIS & RECOMMENDATIONS")
                print("=" * 100)
                
                if len(mapping.table_mappings) == 1:
                    print("\n⚠️  Only mapped to 1 table")
                    print("   This means the multi-table mapper didn't distribute columns across multiple tables.")
                    print("\n   POSSIBLE REASONS:")
                    print("   1. Thresholds are too high (min_confidence_threshold = 0.5)")
                    print("   2. Other tables didn't have good field matches")
                    print("   3. Semantic grouping assigned everything to one table")
                    
                    print("\n   RECOMMENDATIONS:")
                    print("   ✓ Lower min_confidence_threshold to 0.3-0.4")
                    print("   ✓ Check if unmapped columns belong to related tables")
                    print("   ✓ Improve semantic grouping patterns")
                    print("   ✓ Force distribution across entity vs relation tables")
                
                if mapping.unmapped_columns:
                    print(f"\n⚠️  {len(mapping.unmapped_columns)} columns remain unmapped")
                    print("   Consider reviewing these columns for potential table assignments")
        
        # Final status
        if status == 'requires_review':
            print("\n" + "=" * 100)
            print("⚠️  Workflow requires human review")
            print("=" * 100)
            sys.exit(2)
        elif status == 'failed':
            print("\n" + "=" * 100)
            print(f"❌ Workflow failed: {result.get('last_error')}")
            print("=" * 100)
            sys.exit(1)
        else:
            print("\n" + "=" * 100)
            print("✅ Workflow completed successfully")
            print("=" * 100)
            sys.exit(0)
            
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
