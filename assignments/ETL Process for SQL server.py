import pandas as pd
import pyodbc
from sqlalchemy import create_engine

# Extraction: Load data from a CSV file
csv_file = 'sample_data.csv'
df = pd.read_csv(csv_file)

# Transformation: Clean and process the data
# Remove missing values
df = df.dropna()

# Convert date column to datetime (assuming a 'date' column exists)
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])

# Standardize numerical column (assuming a 'sales' column exists)
if 'sales' in df.columns:
    df['sales'] = df['sales'].astype(float)

# Create a new column for categorization (e.g., sales range)
if 'sales' in df.columns:
    df['sales_category'] = pd.cut(df['sales'], 
                                 bins=[0, 1000, 5000, float('inf')], 
                                 labels=['Low', 'Medium', 'High'])

# Loading: Save transformed data to SQL Server
# SQL Server connection string
sql_server_conn_str = (
    "DRIVER={SQL Server};"
    "SERVER=your_server_name;"
    "DATABASE=your_database_name;"
    "Trusted_Connection=yes;"
)

# Create SQLAlchemy engine for SQL Server
engine = create_engine('mssql+pyodbc:///?odbc_connect={}'.format(sql_server_conn_str))

# Save dataframe to SQL Server
df.to_sql('processed_data', engine, if_exists='replace', index=False)

# Verify data in SQL Server
with engine.connect() as conn:
    result = pd.read_sql("SELECT TOP 5 * FROM processed_data", conn)
    print("First 5 rows of data in SQL Server:")
    print(result)

# Note: For Power BI, the SQL Server table 'processed_data' can be accessed via Power BI's SQL Server connector