import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

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

# Loading: Save transformed data to SQLite
conn = sqlite3.connect('etl_database.db')
df.to_sql('processed_data', conn, if_exists='replace', index=False)

# Data Visualization
# 1. Bar Plot: Sales by Category
plt.figure(figsize=(8, 6))
if 'sales_category' in df.columns:
    sns.countplot(data=df, x='sales_category')
    plt.title('Distribution of Sales Categories')
    plt.xlabel('Sales Category')
    plt.ylabel('Count')
    plt.savefig('sales_category_bar.png')
    plt.close()

# 2. Line Plot: Sales over Time (if date and sales columns exist)
if 'date' in df.columns and 'sales' in df.columns:
    plt.figure(figsize=(10, 6))
    df_grouped = df.groupby(df['date'].dt.date)['sales'].sum()
    df_grouped.plot(kind='line')
    plt.title('Sales Trend Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    plt.grid(True)
    plt.savefig('sales_trend_line.png')
    plt.close()

# Close SQLite connection
conn.close()