"""
Quick test for file_analysis_node
Run with: python -m Agent.tests.test_file_analysis
"""

from Agent.models.workflow_state import WorkflowState
from Agent.nodes.file_analysis_node import file_analysis_node
from Agent.models.file_analysis_model import CSVAnalysisResult

print("=" * 80)
print("TESTING FILE ANALYSIS NODE")
print("=" * 80)

# Create initial state
initial_state: WorkflowState = {
    "file_path": r"C:\Users\axel.grille\Documents\rules-engine-agent\oppo_combi.csv",
    "workflow_step": "start",
    "steps_completed": [],
    "messages": [],
    "errors": []
}

print("\nINPUT STATE:")
print(f"  File Path: {initial_state['file_path']}")
print(f"  Workflow Step: {initial_state['workflow_step']}")

print("\nRUNNING NODE...")
print("-" * 80)

# Run the node (it will call the service internally)
result_state = file_analysis_node(initial_state)

print("-" * 80)

print("\nOUTPUT STATE:")
print(f"  Next step: {result_state['workflow_step']}")
print(f"  Steps completed: {result_state['steps_completed']}")
print(f"  Status: {result_state.get('workflow_status', 'N/A')}")

print("\n" + "=" * 80)
print("FILE ANALYSIS RESULT (Structured Model)")
print("=" * 80)

# Get result from the state
result = result_state["file_analysis_result"]

# Test the summary method
print(result.get_summary())

print("\n" + "=" * 80)
print("DETAILED INFORMATION")
print("=" * 80)

# Test all the properties
print(f"File Type: {result.structure.file_type}")
print(f"File Name: {result.structure.file_name}")
print(f"Total Rows: {result.structure.total_rows:,}")
print(f"Total Columns: {result.structure.total_columns}")

# CSV-specific properties
if isinstance(result, CSVAnalysisResult):
    print(f"Delimiter: {result.delimiter}")
    print(f"Encoding: {result.encoding}")
    print(f"Has Header: {result.has_header}")

# Quality metrics
print(f"\nQuality Metrics:")
print(f"  • Total null values: {result.quality_metrics.total_null_values:,}")
print(f"  • Null percentage: {result.quality_metrics.null_percentage:.2f}%")
print(f"  • Potential issues: {len(result.quality_metrics.potential_issues)}")

if result.quality_metrics.potential_issues:
    print("\n  Quality Issues Found:")
    for i, issue in enumerate(result.quality_metrics.potential_issues[:5], 1):
        print(f"    {i}. {issue}")

# Column details
print(f"\nColumn Details (first 5):")
for col in result.columns[:5]:
    print(f"\n  • {col.name}")
    print(f"    Type: {col.data_type}")
    if col.max_length:
        print(f"    Max Length: {col.max_length}")
    print(f"    Nulls: {col.null_count} | Unique: {col.unique_count}")
    if col.sample_values:
        print(f"    Samples: {col.sample_values[:3]}")  # Show first 3 samples

# Sample data
if result.sample_data:
    print(f"\nSample Data (first 2 rows):")
    for i, row in enumerate(result.sample_data[:2], 1):
        print(f"\n  Row {i}:")
        for key, value in list(row.items())[:5]:  # Show first 5 columns
            print(f"    {key}: {value}")

print("\n" + "=" * 80)
print(f"Analysis Success: {result.analysis_success}")
print(f"Timestamp: {result.analysis_timestamp}")
print("=" * 80)

# Test to_dict() method
print("\n" + "=" * 80)
print("TESTING to_dict() SERIALIZATION")
print("=" * 80)
result_dict = result.to_dict()
print(f"Serialized to dict successfully: {len(result_dict)} keys")
print(f"Keys: {list(result_dict.keys())}")

# Quick validation
print("\n" + "=" * 80)
print("VALIDATION")
print("=" * 80)

checks = {
    "Workflow step is rag_matching": result_state["workflow_step"] == "rag_matching",
    "File analysis in steps_completed": "file_analysis" in result_state["steps_completed"],
    "Analysis was successful": result.analysis_success,
    "Has rows": result.structure.total_rows and result.structure.total_rows > 0,
    "Has columns": result.structure.total_columns and result.structure.total_columns > 0,
    "Has column metadata": len(result.columns) > 0,
    "No errors in state": len(result_state.get("errors", [])) == 0
}

for check, passed in checks.items():
    status = "SUCCED" if passed else "FAILED"
    print(f"  {status} {check}")

all_passed = all(checks.values())
print(f"\n{'ALL CHECKS PASSED!' if all_passed else 'SOME CHECKS FAILED'}")
print("=" * 80)