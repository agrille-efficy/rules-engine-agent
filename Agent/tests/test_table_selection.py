"""
Quick test for table selection integration.
Run with: python -m Agent.tests.test_table_selection
"""
from Agent.models.workflow_state import WorkflowState
from Agent.nodes.file_analysis_node import file_analysis_node
from Agent.nodes.rag_matching_node import rag_matching_node
from Agent.nodes.table_selection_node import table_selection_node

print("=" * 80)
print("TESTING TABLE SELECTION INTEGRATION")
print("=" * 80)

# Step 1: File Analysis
print("\n[STEP] STEP 1: File Analysis")

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

state_after_rag = rag_matching_node(state_after_analysis)
print(f"[OK] RAG complete: {state_after_rag['workflow_step']}")

if state_after_rag.get("rag_match_result"):
    print(f"   Found {len(state_after_rag['rag_match_result'].matched_tables)} matches")

# Step 3: Table Selection
print("\n[STEP] STEP 3: Table Selection")

state_after_selection = table_selection_node(state_after_rag)
print(f"[OK] Selection complete: {state_after_selection['workflow_step']}")

# Display selection results
print("\n" + "=" * 80)
print("TABLE SELECTION RESULTS")
print("=" * 80)

if "selected_table" in state_after_selection:
    selected_table = state_after_selection["selected_table"]
    metadata = state_after_selection.get("selected_table_metadata", {})
    
    print(f"\n[PRIMARY] SELECTED TABLE: {selected_table}")
    print(f"   Schema: {state_after_selection.get('selected_schema', 'N/A')}")
    print(f"   Confidence: {metadata.get('confidence', 'N/A')}")
    print(f"   Score: {metadata.get('similarity_score', 0):.3f}")
    print(f"   Reason: {metadata.get('selection_reason', 'N/A')}")
    print(f"   Requires Review: {metadata.get('requires_review', False)}")
    
    if metadata.get('metadata'):
        print(f"\n[METADATA] Table Metadata:")
        table_meta = metadata['metadata']
        print(f"   Table Kind: {table_meta.get('table_kind', 'N/A')}")
        print(f"   Field Count: {table_meta.get('field_count', 'N/A')}")
else:
    print("\n[ERROR] NO TABLE SELECTED")

# Show alternative matches (what wasn't selected)
if state_after_rag.get("rag_match_result"):
    result = state_after_rag["rag_match_result"]
    print(f"\n[ALTERNATIVES] ALTERNATIVE MATCHES (Top 5):")
    for i, match in enumerate(result.matched_tables[:5], 1):
        selected_marker = "[OK]" if match.table_name == state_after_selection.get("selected_table") else "  "
        print(f"{selected_marker} {i}. {match.table_name} (score: {match.similarity_score:.3f}, {match.confidence})")

# Validation
print("\n" + "=" * 80)
print("VALIDATION")
print("=" * 80)

checks = {
    "File analysis completed": "file_analysis" in state_after_selection.get("steps_completed", []),
    "RAG matching completed": "rag_matching" in state_after_selection.get("steps_completed", []),
    "Table selection completed": "table_selection" in state_after_selection.get("steps_completed", []),
    "File analysis completed": "file_analysis" in state_after_selection.get("steps_completed", []),
    "RAG matching completed": "rag_matching" in state_after_selection.get("steps_completed", []),
    "Table selection completed": "table_selection" in state_after_selection.get("steps_completed", []),
    "Next step is field_mapping": state_after_selection.get("workflow_step") == "field_mapping",
    "Has selected table": "selected_table" in state_after_selection and state_after_selection["selected_table"] is not None,
    "Has table metadata": "selected_table_metadata" in state_after_selection,
    "No critical errors": len(state_after_selection.get("errors", [])) == 0 or state_after_selection.get("workflow_status") != "failed"
}

for check, passed in checks.items():
    status = "[OK]" if passed else "[FAIL]"
    print(f"  {status} {check}")

all_passed = all(checks.values())
print(f"\n{'ALL CHECKS PASSED!' if all_passed else 'SOME CHECKS FAILED'}")

# Test with user preference
print("\n" + "=" * 80)
print("TESTING USER PREFERENCE OVERRIDE")
print("=" * 80)

if state_after_rag.get("rag_match_result") and len(state_after_rag["rag_match_result"].matched_tables) > 1:
    # Try to select the second-best match
    second_match = state_after_rag["rag_match_result"].matched_tables[1]
    print(f"\nTesting preference for: {second_match.table_name}")
    
    state_with_preference = {
        **state_after_rag,
        "table_preference": second_match.table_name
    }
    
    state_preference_result = table_selection_node(state_with_preference)
    
    if state_preference_result.get("selected_table") == second_match.table_name:
        print(f"[OK] User preference honored: {second_match.table_name} selected")
    else:
        print(f"[FAIL] User preference ignored: {state_preference_result.get('selected_table')} selected instead")
else:
    print("Not enough matches to test preference override")

print("=" * 80)
