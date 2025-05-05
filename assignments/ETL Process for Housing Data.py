import pandas as pd
import sqlite3
from sqlalchemy import create_engine

# Extraction: Load housing data
url = "https://raw.githubusercontent.com/huzaifsayed/Linear-Regression-Model-for-House-Price-Prediction/master/USA_Housing.csv"
df = pd.read_csv(url)

# Transformation: Create new attributes
# 1. Price per room
df['price_per_room'] = df['Price'] / df['Avg. Area Number of Rooms'].replace(0, 1)  # Avoid division by zero

# 2. Price category (Low, Medium, High)
df['price_category'] = pd.qcut(df['Price'], q=3, labels=['Low', 'Medium', 'High'])

# 3. Standardize column names (lowercase, no spaces)
df.columns = df.columns.str.lower().str.replace(' ', '_')

# 4. Handle missing values (if any)
df = df.fillna(df.mean(numeric_only=True))

# Loading: Save transformed data to SQLite
# Create SQLite database
engine = create_engine('sqlite:///housing_data.db')

# Save DataFrame to SQLite
df.to_sql('housing', engine, if_exists='replace', index=False)

# Verify data in SQLite
with engine.connect() as conn:
    result = pd.read_sql("SELECT * FROM housing LIMIT 5", conn)
    print("Sample of Transformed Data in SQLite:")
    print(result)

# Note: The 'housing' table in SQLite can be accessed in Power BI using the SQLite ODBC driver