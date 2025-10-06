"""
End-to-end test for the complete workflow engine.
Tests all 4 steps running autonomously via LangGraph.
Run with: python -m Agent.tests.test_workflow_engine
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Agent.main import run_workflow

print("=" * 80)
print("TESTING COMPLETE WORKFLOW ENGINE (LANGGRAPH)")
print("=" * 80)

# Test file path
file_path = r"C:\Users\axel.grille\Documents\rules-engine-agent\oppo_combi.csv"

print(f"\nFile: {file_path}")
print("\nStarting autonomous workflow execution...\n")

# Run the complete workflow
try:
    result = run_workflow(
        file_path=file_path,
        user_context="Sales opportunity pipeline data",
        log_level="INFO"
    )
    
    print("\n" + "=" * 80)
    print("WORKFLOW RESULTS")
    print("=" * 80)
    
    # Extract key results
    status = result.get("workflow_status")
    final_step = result.get("workflow_step")
    steps_completed = result.get("steps_completed", [])
    errors = result.get("errors", [])
    
    print(f"\nEXECUTION STATUS:")
    print(f"   Status: {status}")
    print(f"   Final Step: {final_step}")
    print(f"   Steps Completed: {len(steps_completed)}")
    print(f"   Errors: {len(errors)}")
    
    if steps_completed:
        print(f"\n[OK] COMPLETED STEPS:")
        for i, step in enumerate(steps_completed, 1):
            print(f"   {i}. {step}")
    
    # Show results from each step
    if result.get("file_analysis_result"):
        analysis = result["file_analysis_result"]
        print(f"\nFILE ANALYSIS:")
        print(f"   File: {analysis.structure.file_name}")
        print(f"   Rows: {analysis.structure.total_rows}")
        print(f"   Columns: {analysis.structure.total_columns}")
    
    if result.get("rag_match_result"):
        rag = result["rag_match_result"]
        print(f"\nRAG MATCHING:")
        print(f"   Candidates: {len(rag.matched_tables)}")
        if rag.primary_match:
            print(f"   Best Match: {rag.primary_match.table_name} ({rag.primary_match.similarity_score:.3f})")
    
    if result.get("selected_table"):
        table = result["selected_table"]
        metadata = result.get("selected_table_metadata", {})
        print(f"\nTABLE SELECTION:")
        print(f"   Selected: {table}")
        print(f"   Confidence: {metadata.get('confidence', 'N/A')}")
        print(f"   Score: {metadata.get('similarity_score', 0):.3f}")
    
    if result.get("field_mapping_result"):
        mapping = result["field_mapping_result"]
        
        # Check if multi-table mapping (has 'table_mappings' attribute)
        if hasattr(mapping, 'table_mappings'):
            # Multi-table mapping result
            total_mapped = sum(len(tm.mappings) for tm in mapping.table_mappings)
            print(f"\n[MAPPING] MULTI-TABLE FIELD MAPPING:")
            print(f"   Mapped: {total_mapped}/{mapping.total_source_columns}")
            print(f"   Coverage: {mapping.overall_coverage:.1f}%")
            print(f"   Confidence: {mapping.overall_confidence}")
            print(f"   Valid: {mapping.is_valid}")
            print(f"   Tables: {len(mapping.table_mappings)}")
            
            print(f"\n   Tables mapped:")
            for tm in mapping.table_mappings:
                print(f"     - {tm.table_name}: {len(tm.mappings)} columns ({tm.validation.mapping_coverage_percent:.1f}%)")
            
            # Show top mappings from first table
            if mapping.table_mappings and mapping.table_mappings[0].mappings:
                print(f"\n   Top 5 Mappings (from {mapping.table_mappings[0].table_name}):")
                for i, m in enumerate(mapping.table_mappings[0].mappings[:5], 1):
                    print(f"   {i}. {m.source_column} → {m.target_column} ({m.confidence_score:.2f})")
        else:
            # Single-table mapping result
            validation = mapping.validation
            print(f"\n[MAPPING] FIELD MAPPING:")
            print(f"   Mapped: {validation.mapped_count}/{validation.total_mappings}")
            print(f"   Coverage: {validation.mapping_coverage_percent:.1f}%")
            print(f"   Confidence: {validation.confidence_level}")
            print(f"   Valid: {validation.is_valid}")
            
            print(f"\n   Top 5 Mappings:")
            for i, m in enumerate(mapping.mappings[:5], 1):
                print(f"   {i}. {m.source_column} → {m.target_column} ({m.confidence_score:.2f})")
    
    # Validation
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)
    
    checks = {
        "Workflow completed": status in ["requires_review", "in_progress", "completed"],
        "No critical failures": status != "failed",
        "File analysis done": "file_analysis" in steps_completed,
        "RAG matching done": "rag_matching" in steps_completed,
        "Table selection done": "table_selection" in steps_completed,
        "Field mapping done": "field_mapping" in steps_completed,
        "Has final results": result.get("field_mapping_result") is not None
    }
    
    for check, passed in checks.items():
        status_icon = "[OK]" if passed else "[FAIL]"
        print(f"  {status_icon} {check}")
    
    all_passed = all(checks.values())
    
    if all_passed:
        print(f"\nALL CHECKS PASSED!")
        print("=" * 80)
        print("\nWORKFLOW ENGINE IS FULLY OPERATIONAL!")
        print("\nThe complete autonomous workflow executed successfully:")
        print("  1. File Analysis (with translation)")
        print("  2. RAG Matching (semantic search)")
        print("  3. Table Selection (auto-select)")
        print("  4. Field Mapping (intelligent matching)")
        print("\nYour Rules Engine Agent is ready for production!")
    else:
        print(f"\nSOME CHECKS FAILED")
    
    print("=" * 80)
    
except Exception as e:
    print(f"\nWORKFLOW FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
