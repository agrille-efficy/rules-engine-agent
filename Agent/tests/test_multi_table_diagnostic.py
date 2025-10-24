"""
Test multi-table field mapping functionality.
"""
import os
from Agent.services.file_analyzer import FileAnalyzerService
from Agent.services.table_matcher import TableMatcherService
from Agent.services.multi_table_mapper import MultiTableFieldMapper


def test_multi_table_mapping():
    """Test mapping a CSV to multiple database tables."""
    print("=" * 80)
    print("TESTING MULTI-TABLE FIELD MAPPING (DIAGNOSTIC)")
    print("=" * 80)
    
    file_path = r"C:\Users\axel.grille\Documents\rules-engine-agent\oppo_combi.csv"
    print(f"File: {file_path}")
    
    # Step 1: File Analysis
    print("\nSTEP 1: File Analysis")
    print("-" * 80)
    analyzer = FileAnalyzerService()
    file_analysis = analyzer.analyze(file_path)
    print(f"Analysis complete")
    print(f"   Rows: {file_analysis.structure.total_rows}")
    print(f"   Columns: {file_analysis.structure.total_columns}")
    
    # Check translation status
    print(f"\nTRANSLATION STATUS:")
    translated_count = sum(1 for col in file_analysis.columns if col.translation_used)
    print(f"   Translated columns: {translated_count}/{len(file_analysis.columns)}")
    
    print(f"\n   Sample translations:")
    for col in file_analysis.columns[:10]:
        if col.translation_used:
            print(f"     {col.name} -> {col.english_name} [OK]")
        else:
            print(f"     {col.name} (no translation)")
    
    # Show unmapped columns with their English names
    unmapped_french = ['Nom du compte', 'Origine de l\'opportunité', 'Date de validité de l\'offre', 
                       'Commentaire', 'Date de passage en affaire chaude', 'Année Budgétaire CAPEX',
                       'ID_Personne en charge', 'responsible', 'creator', 'intr_ vat']
    
    print(f"\n   Key unmapped columns and their translations:")
    for col in file_analysis.columns:
        if col.name in unmapped_french:
            print(f"     '{col.name}' -> '{col.english_name or 'NO TRANSLATION'}' (used: {col.translation_used})")
    
    # Step 2: RAG Matching
    print("\nSTEP 2: RAG Matching (Get ALL Candidate Tables)")
    print("-" * 80)
    matcher = TableMatcherService()
    rag_result = matcher.find_matching_tables(file_analysis)
    
    print(f"[OK] RAG complete")
    print(f"   Total candidates: {len(rag_result.matched_tables)}")
    
    # Convert TableMatch objects to dicts for multi-table mapper
    candidate_tables = []
    for table_match in rag_result.matched_tables:
        candidate_tables.append({
            'table_name': table_match.table_name,
            'composite_score': table_match.similarity_score,
            'table_kind': table_match.metadata.get('table_kind', 'Entity'),
            'table_code': table_match.metadata.get('table_code', ''),
            'field_count': table_match.metadata.get('field_count', 0),
            'query_coverage': table_match.metadata.get('query_coverage', 0.0)
        })
    
    entity_count = sum(1 for t in candidate_tables if t.get('table_kind') == 'Entity')
    relation_count = sum(1 for t in candidate_tables if t.get('table_kind') == 'Relation')
    print(f"   Entity tables: {entity_count}")
    print(f"   Relation tables: {relation_count}")
    
    print(f"   Top 10 Candidates:")
    for i, table in enumerate(candidate_tables[:10], 1):
        print(f"   {i}. {table['table_name']} ({table.get('table_kind', 'Unknown table kind')}) - Score: {table['composite_score']:.3f}")
    
    # Step 3: Multi-Table Mapping with DIAGNOSTICS
    print("\nSTEP 3: Multi-Table Mapping (DIAGNOSTIC MODE)")
    print("-" * 80)
    mapper = MultiTableFieldMapper()
    
    # Get semantic grouping
    column_groups = mapper._group_columns_semantically(file_analysis)
    print(f"\nSEMANTIC COLUMN GROUPS:")
    for group_name, columns in column_groups.items():
        print(f"\n  {group_name.upper()} ({len(columns)} columns):")
        for col in columns[:5]:  # Show first 5
            print(f"    - {col}")
        if len(columns) > 5:
            print(f"    ... and {len(columns) - 5} more")
    
    # Get table schemas
    table_schemas = mapper._get_candidate_schemas(
        candidate_tables,
        max_tables=10
    )
    
    # Map columns to all tables
    column_to_table_mappings = mapper._map_columns_to_all_tables(
        file_analysis,
        table_schemas
    )
    
    print(f"\nCOLUMN TO TABLE MAPPINGS:")
    mapped_count = 0
    unmapped_columns = []
    
    for col_name, candidates in column_to_table_mappings.items():
        if candidates:
            mapped_count += 1
            best = candidates[0]
            print(f"\n  [OK] {col_name}")
            print(f"     -> {best['table']} ({best['mapping'].target_column}) - Confidence: {best['mapping'].confidence_score:.2f}")
            
            # Show alternative tables if any
            if len(candidates) > 1:
                print(f"     Alternatives:")
                for alt in candidates[1:3]:  # Show top 2 alternatives
                    print(f"       - {alt['table']} ({alt['mapping'].target_column}) - {alt['mapping'].confidence_score:.2f}")
        else:
            unmapped_columns.append(col_name)
    
    print(f"\n  [FAIL] UNMAPPED COLUMNS ({len(unmapped_columns)}):")
    for col in unmapped_columns:
        print(f"     - {col}")
    
    print(f"\nMAPPING STATISTICS:")
    print(f"   Mappable columns: {mapped_count}/{len(column_to_table_mappings)} ({mapped_count/len(column_to_table_mappings)*100:.1f}%)")
    print(f"   Unmappable columns: {len(unmapped_columns)}/{len(column_to_table_mappings)} ({len(unmapped_columns)/len(column_to_table_mappings)*100:.1f}%)")
    
    # Calculate group affinities
    group_affinities = mapper._calculate_group_table_affinity(
        column_groups,
        column_to_table_mappings,
        table_schemas
    )
    
    print(f"\nGROUP TO TABLE AFFINITIES:")
    for group_name, affinities in group_affinities.items():
        print(f"\n  {group_name.upper()}:")
        if affinities:
            for aff in affinities[:3]:  # Top 3 tables per group
                print(f"    {aff['table']}: affinity={aff['affinity']:.2f}, coverage={aff['coverage']:.2f}, avg_conf={aff['avg_confidence']:.2f}")
        else:
            print(f"    [FAIL] No table affinity found (no columns mappable)")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    test_multi_table_mapping()
