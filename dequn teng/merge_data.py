import pandas as pd

# Read the CSV files
property_df = pd.read_csv('BEIS Electrification of Heat Project - Property, Design and Installation Information.csv')
performance_df = pd.read_csv('EOH - Heat Pump Performance Data Summary.csv')

# Merge the dataframes on Property_ID
merged_df = pd.merge(
    property_df, 
    performance_df,
    on='Property_ID',
    how='outer'  # Use outer join to keep all records from both datasets
)

# Save the merged data to a new CSV file
output_file = 'merged_heat_pump_data.csv'
merged_df.to_csv(output_file, index=False)

# Print merge statistics
print(f"Original property records: {len(property_df)}")
print(f"Original performance records: {len(performance_df)}")
print(f"Merged records: {len(merged_df)}")
print(f"\nMerged data saved to: {output_file}")

# Print any unmatched records
unmatched_property = property_df[~property_df['Property_ID'].isin(performance_df['Property_ID'])]
unmatched_performance = performance_df[~performance_df['Property_ID'].isin(property_df['Property_ID'])]

print(f"\nUnmatched property records: {len(unmatched_property)}")
print(f"Unmatched performance records: {len(unmatched_performance)}") 