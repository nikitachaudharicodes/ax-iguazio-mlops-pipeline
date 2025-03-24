# data_loader.py

import psycopg2
import pandas as pd
from dotenv import load_dotenv
import os

# load env
load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT", 5432),  # default 5432
}

def get_connection():
    """Establish and return a database connection."""
    return psycopg2.connect(**DB_CONFIG)


def fetch_data(query):
    """Fetch data from the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(query)
    data = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    cursor.close()  # Close cursor before closing connection
    conn.close()
    return pd.DataFrame(data, columns=columns)


def load_ratings(limit=50000):
    query = f"SELECT * FROM rating LIMIT {limit};"
    return fetch_data(query)


def load_movies():
    """Load movies data from the database."""
    return fetch_data("SELECT * FROM movie LIMIT {limit};")


def load_users():
    """Load user information from the database."""
    return fetch_data("SELECT * FROM user_info LIMIT {limit};")


