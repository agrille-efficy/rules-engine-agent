"""
Quick runner script for the Rules Engine Agent.
Usage: python run_agent.py
"""
from Agent.main import run_workflow

# Configure your file here
FILE_PATH = "oppo_combi.csv"
USER_CONTEXT = "Sales opportunity pipeline data"

print("=" * 80)
print("RULES ENGINE AGENT - QUICK RUN")
print("=" * 80)
print(f"File: {FILE_PATH}")
print(f"Context: {USER_CONTEXT}")
print("=" * 80)

# Run the workflow
result = run_workflow(
    file_path=FILE_PATH,
    user_context=USER_CONTEXT,
    log_level="INFO"
)

# Display results
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

# Display field mapping refinement info (if available in logs)
print(f"\n💡 TIP: Check the logs above for refinement details:")
print(f"   Look for: '🔍 Starting refinement analysis'")
print(f"   Look for: '⚠️  Detected X mappings to...'")
print(f"   Look for: '🔧 Refinement: Removed X suspicious mappings'")

if result.get('field_mapping_result'):
    mapping = result['field_mapping_result']
    
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
        
        if mapping.unmapped_columns:
            print(f"\n⚠️  {len(mapping.unmapped_columns)} columns remain unmapped")
            print("   These might belong to related tables like:")
            print("   • Company (compUniqueID, Nom du compte)")
            print("   • Contact (ID_Personne en charge, responsible)")
            print("   • Opportunity metadata (Commentaire, Origine)")
    
if status == 'requires_review':
    print("\n" + "=" * 100)
    print("⚠️  Workflow requires human review")
    print("=" * 100)
elif status == 'failed':
    print("\n" + "=" * 100)
    print(f"❌ Workflow failed: {result.get('last_error')}")
    print("=" * 100)
else:
    print("\n" + "=" * 100)
    print("✅ Workflow completed successfully")
    print("=" * 100)

# # Next steps
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
