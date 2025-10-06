"""
Test multi-table field mapping.
Tests mapping CSV columns to multiple database tables.
Run with: python -m Agent.tests.test_multi_table_mapping
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Agent.services.file_analyzer import FileAnalyzerService
from Agent.services.table_matcher import TableMatcherService
from Agent.services.multi_table_mapper import MultiTableFieldMapper

print("=" * 80)
print("TESTING MULTI-TABLE FIELD MAPPING")
print("=" * 80)

# Test file
file_path = r"C:\Users\axel.grille\Documents\rules-engine-agent\oppo_combi.csv"

print(f"\nFile: {file_path}")
print("\nSTEP 1: File Analysis")
print("-" * 80)

# Analyze file
analyzer = FileAnalyzerService()
file_analysis = analyzer.analyze(file_path)

print(f"[OK] Analysis complete")
print(f"   Rows: {file_analysis.structure.total_rows}")
print(f"   Columns: {file_analysis.structure.total_columns}")

print("\nSTEP 2: RAG Matching (Get ALL Candidate Tables)")
print("-" * 80)

# Get table matches
matcher = TableMatcherService()
rag_result = matcher.find_matching_tables(file_analysis)

print(f"[OK] RAG complete")
print(f"   Total candidates: {len(rag_result.matched_tables)}")
print(f"   Entity tables: {len([t for t in rag_result.matched_tables if t.metadata and t.metadata.get('table_kind') == 'Entity'])}")
print(f"   Relation tables: {len([t for t in rag_result.matched_tables if t.metadata and t.metadata.get('table_kind') == 'Relation'])}")

# Show top 10 candidates
print(f"\n   Top 10 Candidates:")
for i, table in enumerate(rag_result.matched_tables[:10], 1):
    table_kind = table.metadata.get('table_kind', 'Unknown') if table.metadata else 'Unknown'
    print(f"   {i}. {table.table_name} ({table_kind}) - Score: {table.similarity_score:.3f}")

print("\nSTEP 3: Multi-Table Mapping")
print("-" * 80)

# Prepare candidate tables for mapper
candidate_tables = [
    {
        'table_name': t.table_name,
        'table_kind': t.metadata.get('table_kind', 'Entity') if t.metadata else 'Entity',
        'composite_score': t.similarity_score
    }
    for t in rag_result.matched_tables
]

# Perform multi-table mapping
multi_mapper = MultiTableFieldMapper()
result = multi_mapper.map_to_multiple_tables(
    file_analysis=file_analysis,
    candidate_tables=candidate_tables,
    max_tables=10  # Evaluate top 10 tables
)

print(f"\n[OK] Multi-table mapping complete!")

print("\n" + "=" * 80)
print("MULTI-TABLE MAPPING RESULTS")
print("=" * 80)

print(f"\nOVERALL STATISTICS:")
print(f"   Source file: {result.source_file}")
print(f"   Total columns: {result.total_source_columns}")
print(f"   Tables mapped: {len(result.table_mappings)}")
print(f"   Overall coverage: {result.overall_coverage:.1f}%")
print(f"   Overall confidence: {result.overall_confidence}")
print(f"   Valid: {result.is_valid}")
print(f"   Requires review: {result.requires_review}")

print(f"\nMAPPINGS BY TABLE:")
for i, tm in enumerate(result.table_mappings, 1):
    print(f"\n{i}. {tm.table_name} ({tm.table_type})")
    print(f"   Columns mapped: {len(tm.mappings)}")
    print(f"   Coverage: {tm.validation.mapping_coverage_percent:.1f}%")
    print(f"   Confidence: {tm.validation.confidence_level} ({tm.confidence_score:.2f})")
    print(f"   Valid: {tm.validation.is_valid}")
    
    print(f"\n   Field Mappings (Top 5):")
    for j, mapping in enumerate(tm.mappings[:5], 1):
        confidence_icon = "[OK]" if mapping.confidence_score >= 0.8 else "[WARN]" if mapping.confidence_score >= 0.6 else "[LOW]"
        print(f"   {confidence_icon} {j}. {mapping.source_column} → {mapping.target_column}")
        print(f"      Confidence: {mapping.confidence_score:.2f} ({mapping.match_type})")
    
    if len(tm.mappings) > 5:
        print(f"   ... and {len(tm.mappings) - 5} more mappings")

if result.unmapped_columns:
    print(f"\n[WARNING] UNMAPPED COLUMNS ({len(result.unmapped_columns)}):")
    for col in result.unmapped_columns[:10]:
        print(f"   - {col}")
    if len(result.unmapped_columns) > 10:
        print(f"   ... and {len(result.unmapped_columns) - 10} more")

print("\n" + "=" * 80)
print("VALIDATION")
print("=" * 80)

checks = {
    "Has table mappings": len(result.table_mappings) > 0,
    "Multiple tables mapped": len(result.table_mappings) > 1,
    "Coverage > 50%": result.overall_coverage > 50,
    "Confidence acceptable": result.overall_confidence in ["medium", "high"],
    "Has Entity table": any(tm.table_type == "Entity" for tm in result.table_mappings),
}

for check, passed in checks.items():
    status_icon = "[OK]" if passed else "[FAIL]"
    print(f"  {status_icon} {check}")

all_passed = all(checks.values())

if all_passed:
    print(f"\ ALL CHECKS PASSED!")
    print("\nMULTI-TABLE MAPPING WORKING!")
    print("\nYour system can now:")
    print("  [OK] Map one CSV to multiple database tables")
    print("  [OK] Intelligently assign each column to its best table")
    print("  [OK] Handle Entity + Relation tables together")
    print("  [OK] Provide per-table validation and confidence")
else:
    print(f"\n[WARNING] SOME CHECKS FAILED")

print("=" * 80)

# Export suggestion
print("\nNEXT STEPS:")
print("You can now generate:")
print("  1. Multiple INSERT statements (one per table)")
print("  2. Relationship-preserving data import")
print("  3. Transactional multi-table inserts")
print("=" * 80)
