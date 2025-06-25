import os
import pandas as pd
from sqlalchemy import create_engine

# DB connection
user = os.getenv("DB_USER", "sql12786766")
pw = os.getenv("DB_PASSWORD", "3WUxNFxSyc")
host = os.getenv("DB_HOST", "sql12.freesqldatabase.com")
port = os.getenv("DB_PORT", "3306")
db = os.getenv("DB_NAME", "sql12786766")
engine = create_engine(f"mysql+pymysql://{user}:{pw}@{host}:{port}/{db}")

# Execute and fetch
def execute_sql(sql: str):
    df = pd.read_sql(sql + " LIMIT 100", engine)
    return df.to_dict(orient="records")