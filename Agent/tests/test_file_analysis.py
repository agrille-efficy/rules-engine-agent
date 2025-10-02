from Agent.services.file_analyzer import FileAnalyzerService
from Agent.models.file_analysis_model import CSVAnalysisResult

# Initialize the service
service = FileAnalyzerService()

# Test with your CSV file
csv_path = r"C:\Users\axel.grille\Documents\rules-engine-agent\oppo_combi.csv"
result = service.analyze(csv_path)

print("=" * 80)
print("FILE ANALYSIS RESULT (Structured Model)")
print("=" * 80)

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
    print(f"    Samples: {col.sample_values}")

# Sample data
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