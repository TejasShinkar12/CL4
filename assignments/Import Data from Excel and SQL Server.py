import pandas as pd
import pyodbc

# 1. Import data from Excel
excel_file = 'sample_data.xlsx'
df_excel = pd.read_excel(excel_file)

# 2. Import data from SQL Server
sql_server_conn_str = (
    "DRIVER={SQL Server};"
    "SERVER=your_server_name;"
    "DATABASE=your_database_name;"
    "Trusted_Connection=yes;"
)
sql_query = "SELECT * FROM your_table"
sql_conn = pyodbc.connect(sql_server_conn_str)
df_sql = pd.read_sql(sql_query, sql_conn)
sql_conn.close()

# 3. Combine data into a single DataFrame
# Ensure column names are consistent (modify as per your data)
df_excel.columns = df_excel.columns.str.lower().str.replace(' ', '_')
df_sql.columns = df_sql.columns.str.lower().str.replace(' ', '_')

# Concatenate DataFrames (assuming similar structure)
df_combined = pd.concat([df_excel, df_sql], ignore_index=True)

# Basic cleaning: Remove duplicates and handle missing values
df_combined = df_combined.drop_duplicates()
df_combined = df_combined.fillna(df_combined.mean(numeric_only=True))

# Save combined DataFrame to CSV for further processing
df_combined.to_csv('combined_data.csv', index=False)

# Print sample of combined data
print("Sample of Combined DataFrame:")
print(df_combined.head())