"""
Detailed Mapping Results Viewer
Shows complete field mappings with full analysis before tuning thresholds.
Usage: python view_mapping_results.py
"""
from Agent.main import run_workflow
import json

FILE_PATH = "oppo_combi.csv"
USER_CONTEXT = "Sales opportunity pipeline data"

print("=" * 100)
print("COMPLETE MAPPING RESULTS ANALYSIS")
print("=" * 100)

# Run workflow
result = run_workflow(
    file_path=FILE_PATH,
    user_context=USER_CONTEXT,
    log_level="WARNING"  # Suppress info logs for cleaner output
)

print("\n" + "=" * 100)
print("📊 WORKFLOW SUMMARY")
print("=" * 100)

file_analysis = result.get('file_analysis_result')
if file_analysis:
    print(f"\n📄 FILE: {file_analysis.structure.file_name}")
    print(f"   Rows: {file_analysis.structure.total_rows}")
    print(f"   Columns: {file_analysis.structure.total_columns}")

rag_result = result.get('rag_match_result')
if rag_result:
    print(f"\n🔍 RAG MATCHES: {len(rag_result.matched_tables)} candidate tables found")
    print(f"\n   Top 5 Candidates:")
    for i, match in enumerate(rag_result.matched_tables[:5], 1):
        print(f"   {i}. {match.table_name:30s} (score: {match.similarity_score:.3f}, confidence: {match.confidence})")

mapping_result = result.get('field_mapping_result')

if mapping_result and hasattr(mapping_result, 'table_mappings'):
    print("\n" + "=" * 100)
    print("🎯 MULTI-TABLE MAPPING RESULTS")
    print("=" * 100)
    
    print(f"\n📈 OVERALL STATISTICS:")
    total_mapped = sum(len(tm.mappings) for tm in mapping_result.table_mappings)
    print(f"   Total Source Columns: {mapping_result.total_source_columns}")
    print(f"   Mapped Columns: {total_mapped}")
    print(f"   Unmapped Columns: {len(mapping_result.unmapped_columns)}")
    print(f"   Coverage: {mapping_result.overall_coverage:.1f}%")
    print(f"   Overall Confidence: {mapping_result.overall_confidence}")
    print(f"   Valid: {'✅ Yes' if mapping_result.is_valid else '❌ No'}")
    print(f"   Requires Review: {'⚠️  Yes' if mapping_result.requires_review else '✅ No'}")
    print(f"   Tables Used: {len(mapping_result.table_mappings)}")
    
    # Display each table's mappings
    print("\n" + "=" * 100)
    print("📋 DETAILED MAPPINGS BY TABLE")
    print("=" * 100)
    
    for table_idx, table_mapping in enumerate(mapping_result.table_mappings, 1):
        print(f"\n{'=' * 100}")
        print(f"TABLE #{table_idx}: {table_mapping.table_name} ({table_mapping.table_type})")
        print(f"{'=' * 100}")
        print(f"Insertion Order: {table_mapping.insertion_order}")
        print(f"Columns Mapped: {len(table_mapping.mappings)}")
        print(f"Confidence Score: {table_mapping.confidence:.2f}")
        print(f"Coverage: {table_mapping.validation.mapping_coverage_percent:.1f}%")
        print(f"Confidence Level: {table_mapping.validation.confidence_level}")
        
        print(f"\n{'─' * 100}")
        print(f"{'SOURCE COLUMN':<40} {'TARGET FIELD':<40} {'CONFIDENCE':<12} {'TYPE'}")
        print(f"{'─' * 100}")
        
        for mapping in table_mapping.mappings:
            confidence_bar = "█" * int(mapping.confidence_score * 10)
            print(f"{mapping.source_column:<40} {mapping.target_column:<40} {mapping.confidence_score:.2f} {confidence_bar:<10} {mapping.match_type}")
        
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
    if mapping_result.unmapped_columns:
        print(f"\n{'=' * 100}")
        print(f"❌ UNMAPPED COLUMNS ({len(mapping_result.unmapped_columns)})")
        print(f"{'=' * 100}")
        
        for col in mapping_result.unmapped_columns:
            # Try to find why it wasn't mapped
            col_data = next((c for c in file_analysis.columns if c.name == col), None)
            if col_data:
                print(f"\n   • {col}")
                print(f"     English: {col_data.english_name if col_data.english_name else col}")
                print(f"     Type: {col_data.data_type}")

print("\n" + "=" * 100)
print("💡 ANALYSIS & RECOMMENDATIONS")
print("=" * 100)

if mapping_result and hasattr(mapping_result, 'table_mappings'):
    if len(mapping_result.table_mappings) == 1:
        print("\n⚠️  Only mapped to 1 table (Opportunity)")
        print("   This means the multi-table mapper didn't distribute columns across multiple tables.")
        print("\n   POSSIBLE REASONS:")
        print("   1. Thresholds are too high (min_confidence_threshold = 0.5)")
        print("   2. Other tables didn't have good field matches")
        print("   3. Semantic grouping assigned everything to Opportunity")
        
        print("\n   RECOMMENDATIONS:")
        print("   ✓ Lower min_confidence_threshold to 0.3-0.4")
        print("   ✓ Check if unmapped columns belong to related tables (Company, Contact)")
        print("   ✓ Improve semantic grouping patterns")
        print("   ✓ Force distribution across entity vs relation tables")
    
    if mapping_result.unmapped_columns:
        print(f"\n⚠️  {len(mapping_result.unmapped_columns)} columns remain unmapped")
        print("   These might belong to related tables like:")
        print("   • Company (compUniqueID, Nom du compte)")
        print("   • Contact (ID_Personne en charge, responsible)")
        print("   • Opportunity metadata (Commentaire, Origine)")

# print("\n" + "=" * 100)
# print("🎯 NEXT STEPS")
# print("=" * 100)
# print("\n1. Review the mappings above")
# print("2. Identify which unmapped columns should go to other tables")
# print("3. Tune thresholds in multi_table_mapper.py:")
# print("   - min_confidence_threshold (currently 0.5)")
# print("   - min_columns_per_table (currently 1)")
# print("   - semantic_similarity_threshold (currently 0.75)")
# print("4. Re-run to see if more tables are used")

# print("\n" + "=" * 100)
