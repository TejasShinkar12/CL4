import pandas as pd
import sqlite3
from sqlalchemy import create_engine
import pyodbc
import cx_Oracle

# 1. Import from Excel
excel_file = 'sample_data.xlsx'
df_excel = pd.read_excel(excel_file)

# 2. Import from SQL Server
sql_server_conn_str = (
    "DRIVER={SQL Server};"
    "SERVER=your_server_name;"
    "DATABASE=your_database_name;"
    "Trusted_Connection=yes;"
)
sql_query = "SELECT * FROM your_table"
sql_conn = pyodbc.connect(sql_server_conn_str)
df_sql_server = pd.read_sql(sql_query, sql_conn)
sql_conn.close()

# 3. Import from Oracle
oracle_conn_str = 'username/password@host:port/service'
oracle_conn = cx_Oracle.connect(oracle_conn_str)
oracle_query = "SELECT * FROM your_table"
df_oracle = pd.read_sql(oracle_query, oracle_conn)
oracle_conn.close()

# 4. Load data into a SQLite database (target system)
sqlite_engine = create_engine('sqlite:///target_database.db')

# Save each dataframe to SQLite
df_excel.to_sql('excel_data', sqlite_engine, if_exists='replace', index=False)
df_sql_server.to_sql('sql_server_data', sqlite_engine, if_exists='replace', index=False)
df_oracle.to_sql('oracle_data', sqlite_engine, if_exists='replace', index=False)

# Verify data in SQLite
with sqlite_engine.connect() as conn:
    print("Excel Data:")
    print(pd.read_sql("SELECT * FROM excel_data LIMIT 5", conn))
    print("\nSQL Server Data:")
    print(pd.read_sql("SELECT * FROM sql_server_data LIMIT 5", conn))
    print("\nOracle Data:")
    print(pd.read_sql("SELECT * FROM oracle_data LIMIT 5", conn))