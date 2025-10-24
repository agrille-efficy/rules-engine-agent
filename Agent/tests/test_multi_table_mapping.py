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
from Agent.services.mapper import Mapper

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
    table_kind = table.metadata.get('table_kind', 'unknown table kind') if table.metadata else 'Unknown table kind (metadata missing)'
    print(f"   {i}. {table.table_name} ({table_kind}) - Score: {table.similarity_score:.3f}")

print("\nSTEP 3: Multi-Table Mapping")
print("-" * 80)

# Prepare candidate tables for mapper
candidate_tables = [
    {
        'table_name': t.table_name,
        'fields': t.metadata.get('fields', []) if t.metadata else [],
        'metadata': t.metadata if t.metadata else {},
        'table_kind': t.metadata.get('table_kind', 'Entity') if t.metadata else 'Entity',
        'composite_score': t.similarity_score
    }
    for t in rag_result.matched_tables
]

# Perform multi-table mapping
mapper = Mapper()
result = mapper.map_to_multiple_tables(
    file_analysis=file_analysis,
    candidate_tables=candidate_tables,
    max_tables=10  # Evaluate top 10 tables
)

print(f"\n[OK] Multi-table mapping complete!")

print("\n" + "=" * 80)
print("MULTI-TABLE MAPPING RESULTS")
print("=" * 80)

print(f"\nOVERALL STATISTICS:")
print(f"   Source file: {file_analysis.structure.file_name}")
print(f"   Total columns: {file_analysis.structure.total_columns}")
print(f"   Tables mapped: {len(set([v['table'] for v in result.values() if isinstance(v, dict)]))}")
print(f"   Unmapped columns: {len([k for k, v in result.items() if v == 'UNMAPPED'])}")

print(f"\nMAPPINGS BY TABLE:")
table_groups = {}
for col, mapping in result.items():
    if mapping == 'UNMAPPED':
        continue
    table = mapping['table']
    if table not in table_groups:
        table_groups[table] = []
    table_groups[table].append(mapping)
for i, (table, mappings) in enumerate(table_groups.items(), 1):
    print(f"\n{i}. {table}")
    print(f"   Columns mapped: {len(mappings)}")
    print(f"   Example mappings:")
    for j, mapping in enumerate(mappings[:5], 1):
        confidence_icon = "[OK]" if mapping['confidence_score'] >= 0.8 else "[WARN]" if mapping['confidence_score'] >= 0.6 else "[LOW]"
        print(f"   {confidence_icon} {j}. {mapping['source_column']} → {mapping['target_column']}")
        print(f"      Confidence: {mapping['confidence_score']:.2f} ({mapping['match_type']})")
    if len(mappings) > 5:
        print(f"   ... and {len(mappings) - 5} more mappings")
if [k for k, v in result.items() if v == 'UNMAPPED']:
    print(f"\n[WARNING] UNMAPPED COLUMNS ({len([k for k, v in result.items() if v == 'UNMAPPED'])}):")
    for col in [k for k, v in result.items() if v == 'UNMAPPED'][:10]:
        print(f"   - {col}")
    if len([k for k, v in result.items() if v == 'UNMAPPED']) > 10:
        print(f"   ... and {len([k for k, v in result.items() if v == 'UNMAPPED']) - 10} more")

print("\n" + "=" * 80)
print("VALIDATION")
print("=" * 80)

checks = {
    "Has table mappings": len(set([v['table'] for v in result.values() if isinstance(v, dict)])) > 0,
    "Multiple tables mapped": len(set([v['table'] for v in result.values() if isinstance(v, dict)])) > 1,
    "Coverage > 50%": len([k for k, v in result.items() if v != 'UNMAPPED']) / file_analysis.structure.total_columns > 0.5,
    "Confidence acceptable": all(mapping['confidence_score'] >= 0.6 for mapping in result.values() if isinstance(mapping, dict)),
    "Has Entity table": any(mapping['table_kind'] == "Entity" for mapping in result.values() if isinstance(mapping, dict)),
}

for check, passed in checks.items():
    status_icon = "[OK]" if passed else "[FAIL]"
    print(f"  {status_icon} {check}")

all_passed = all(checks.values())

if all_passed:
    print(f"\n ALL CHECKS PASSED!")
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
