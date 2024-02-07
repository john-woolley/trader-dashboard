import pandas as pd

# Read the CSV files
merged_pharma = pd.read_csv('data/merged_pharma.csv', parse_dates=True, index_col='date')
macro = pd.read_csv('data/macro.csv', parse_dates=True, index_col=0)

# Merge the dataframes
merged_data = pd.merge(merged_pharma, macro, left_index=True, right_index=True, how='left')

# Save the merged data to a new CSV file
merged_data.reset_index(names=['date']).to_csv('data/master.csv')
