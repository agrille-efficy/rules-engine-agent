"""
Quick test for field mapping integration.
Run with: python -m Agent.tests.test_field_mapping
"""
from Agent.models.workflow_state import WorkflowState
from Agent.nodes.file_analysis_node import file_analysis_node
from Agent.nodes.rag_matching_node import rag_matching_node
from Agent.nodes.table_selection_node import table_selection_node
from Agent.nodes.field_mapping_node import field_mapping_node

print("=" * 80)
print("TESTING FIELD MAPPING INTEGRATION")
print("=" * 80)

# Step 1: File Analysis
print("\n[STEP] STEP 1: File Analysis")
print("-" * 80)

initial_state: WorkflowState = {
    "file_path": r"C:\Users\axel.grille\Documents\rules-engine-agent\oppo_combi.csv",
    "workflow_step": "start",
    "steps_completed": [],
    "messages": [],
    "errors": []
}

state_after_analysis = file_analysis_node(initial_state)
print(f"[OK] Analysis complete: {state_after_analysis['workflow_step']}")

# Step 2: RAG Matching
print("\n[STEP] STEP 2: RAG Matching")
print("-" * 80)

state_after_rag = rag_matching_node(state_after_analysis)
print(f"[OK] RAG complete: {state_after_rag['workflow_step']}")

# Step 3: Table Selection
print("\n[STEP] STEP 3: Table Selection")
print("-" * 80)

state_after_selection = table_selection_node(state_after_rag)
print(f"[OK] Selection complete: {state_after_selection['workflow_step']}")
print(f"   Selected: {state_after_selection.get('selected_table')}")

# Step 4: Field Mapping
print("\n[STEP] STEP 4: Field Mapping")
print("-" * 80)

state_after_mapping = field_mapping_node(state_after_selection)
print(f"[OK] Mapping complete: {state_after_mapping['workflow_step']}")

# Display mapping results
print("\n" + "=" * 80)
print("FIELD MAPPING RESULTS")
print("=" * 80)

if "field_mapping_result" in state_after_mapping:
    mapping_result = state_after_mapping["field_mapping_result"]
    validation = mapping_result.validation
    
    print(f"\n[SUMMARY] MAPPING SUMMARY:")
    print(f"   Source: {mapping_result.source_table_name}")
    print(f"   Target: {mapping_result.target_table_name}")
    print(f"   Method: {mapping_result.mapping_method}")
    print(f"   Mapped: {validation.mapped_count}/{validation.total_mappings} columns")
    print(f"   Coverage: {validation.mapping_coverage_percent:.1f}%")
    print(f"   Confidence: {validation.confidence_level}")
    print(f"   Valid: {validation.is_valid}")
    print(f"   Requires Review: {validation.requires_review}")
    
    if validation.issues:
        print(f"\n[WARNING] ISSUES:")
        for issue in validation.issues:
            print(f"   - {issue}")
    
    if validation.warnings:
        print(f"\n[WARNING] WARNINGS:")
        for warning in validation.warnings[:3]:
            print(f"   - {warning}")
    
    print(f"\n[MAPPINGS] FIELD MAPPINGS (Top 10):")
    for i, mapping in enumerate(mapping_result.mappings[:10], 1):
        confidence_icon = "[OK]" if mapping.confidence_score >= 0.8 else "[WARN]" if mapping.confidence_score >= 0.6 else "[LOW]"
        print(f"{confidence_icon} {i}. {mapping.source_column} → {mapping.target_column}")
        print(f"      English: {mapping.source_column_english}")
        print(f"      Confidence: {mapping.confidence_score:.2f} ({mapping.match_type})")
        if mapping.requires_transformation:
            print(f"      Transform: {mapping.transformation_type}")
    
    if validation.unmapped_source_columns:
        print(f"\n[UNMAPPED] UNMAPPED SOURCE COLUMNS ({len(validation.unmapped_source_columns)}):")
        for col in validation.unmapped_source_columns[:5]:
            print(f"   - {col}")
        if len(validation.unmapped_source_columns) > 5:
            print(f"   ... and {len(validation.unmapped_source_columns) - 5} more")
    
    # Show high confidence mappings
    high_conf = mapping_result.get_high_confidence_mappings()
    print(f"\n[OK] HIGH CONFIDENCE MAPPINGS: {len(high_conf)}")
    
else:
    print("\n[ERROR] NO FIELD MAPPING RESULT IN STATE")

# Validation
print("\n" + "=" * 80)
print("VALIDATION")
print("=" * 80)

checks = {
    "File analysis completed": "file_analysis" in state_after_mapping.get("steps_completed", []),
    "RAG matching completed": "rag_matching" in state_after_mapping.get("steps_completed", []),
    "Table selection completed": "table_selection" in state_after_mapping.get("steps_completed", []),
    "Field mapping completed": "field_mapping" in state_after_mapping.get("steps_completed", []),
    "Has field mapping result": "field_mapping_result" in state_after_mapping,
    "Mapping has validation": state_after_mapping.get("field_mapping_result", {}).validation is not None if "field_mapping_result" in state_after_mapping else False,
    "No critical errors": state_after_mapping.get("workflow_status") != "failed"
}

for check, passed in checks.items():
    status = "[OK]" if passed else "[FAIL]"
    print(f"  {status} {check}")

all_passed = all(checks.values())
print(f"\n{'ALL CHECKS PASSED!' if all_passed else 'SOME CHECKS FAILED'}")

# Summary stats
if "field_mapping_result" in state_after_mapping:
    result = state_after_mapping["field_mapping_result"]
    val = result.validation
    
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"  Total source columns: {val.total_mappings}")
    print(f"  Successfully mapped: {val.mapped_count}")
    print(f"  Unmapped source: {len(val.unmapped_source_columns)}")
    print(f"  Unmapped target: {len(val.unmapped_target_columns)}")
    print(f"  Average confidence: {sum(m.confidence_score for m in result.mappings) / len(result.mappings):.2f}" if result.mappings else "  Average confidence: N/A")
    print(f"  Exact matches: {len([m for m in result.mappings if m.match_type == 'exact'])}")
    print(f"  Fuzzy matches: {len([m for m in result.mappings if m.match_type == 'fuzzy'])}")
    print(f"  Semantic matches: {len([m for m in result.mappings if m.match_type == 'semantic'])}")

print("=" * 80)
