"""
Quick test for RAG matching integration.
Run with: python -m Agent.tests.test_rag_matching
"""
from Agent.models.workflow_state import WorkflowState
from Agent.nodes.file_analysis_node import file_analysis_node
from Agent.nodes.rag_matching_node import rag_matching_node

print("=" * 80)
print("TESTING RAG MATCHING INTEGRATION")
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
print(f"   Columns analyzed: {state_after_analysis['file_analysis_result'].structure.total_columns}")

# Step 2: RAG Matching
print("\n[STEP] STEP 2: RAG Matching")

state_after_rag = rag_matching_node(state_after_analysis)

print(f"[OK] RAG complete: {state_after_rag['workflow_step']}")

# Check results
if "rag_match_result" in state_after_rag:
    result = state_after_rag["rag_match_result"]
    
    print("\n" + "=" * 80)
    print("RAG MATCHING RESULTS")
    print("=" * 80)
    
    print(f"\n[STATS] Total matches found: {len(result.matched_tables)}")
    
    if result.primary_match:
        print(f"\n[PRIMARY] PRIMARY MATCH:")
        print(f"   Table: {result.primary_match.table_name}")
        print(f"   Score: {result.primary_match.similarity_score:.3f}")
        print(f"   Confidence: {result.primary_match.confidence}")
        print(f"   Reason: {result.primary_match.reason}")
    
    print(f"\n[TOP_5] TOP 5 MATCHES:")
    for i, match in enumerate(result.matched_tables[:5], 1):
        print(f"\n{i}. {match.table_name}")
        print(f"   Score: {match.similarity_score:.3f}")
        print(f"   Confidence: {match.confidence}")
        print(f"   Kind: {match.metadata.get('table_kind', 'unknown')}")
        print(f"   Fields: {match.metadata.get('field_count', 'unknown')}")
    
    print(f"\n[QUERY] Search Query: {result.search_query}")
    
else:
    print("\n[ERROR] NO RAG MATCH RESULT IN STATE")

# Validation
print("\n" + "=" * 80)
print("VALIDATION")
print("=" * 80)

# Safe check for rag_match_result
rag_result = state_after_rag.get("rag_match_result")
has_matches = False
has_primary = False

if rag_result and hasattr(rag_result, 'matched_tables'):
    has_matches = len(rag_result.matched_tables) > 0
    has_primary = rag_result.primary_match is not None

checks = {
    "File analysis completed": "file_analysis" in state_after_rag.get("steps_completed", []),
    "RAG matching completed": "rag_matching" in state_after_rag.get("steps_completed", []),
    "Next step is table_selection": state_after_rag.get("workflow_step") == "table_selection",
    "Has RAG results": "rag_match_result" in state_after_rag,
    "Found matches": has_matches,
    "Has primary match": has_primary,
    "No errors": len(state_after_rag.get("errors", [])) == 0
}

for check, passed in checks.items():
    status = "[OK]" if passed else "[FAIL]"
    print(f"  {status} {check}")

all_passed = all(checks.values())
print(f"\n{'[SUCCESS] ALL CHECKS PASSED!' if all_passed else '[WARNING] SOME CHECKS FAILED'}")
print("=" * 80)