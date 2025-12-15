import psycopg2
import os
from typing import List
from dotenv import load_dotenv 
load_dotenv()

DB_CONFIG = {
    'database': os.getenv("PG_DATABASE"),
    'user': os.getenv("PG_USER"),
    'password': os.getenv("PG_PASSWORD"),
    'host': os.getenv("PG_HOST"),
    'port': os.getenv("PG_PORT")
}

if not all(DB_CONFIG.values()):
    missing = [k for k, v in DB_CONFIG.items() if not v]
    raise ValueError(f"Missing required PostgreSQL environment variables: {', '.join(missing)}")

def insert_company(company_name: str, founding_date: str, founders: List[str]):
    """
    Inserts company data into PostgreSQL using psycopg2.
    """
    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        insert_query = """
            INSERT INTO companies (company_name, founding_date, founders)
            VALUES (%s, %s, %s)
        """
        
        data_to_insert = (
            company_name, 
            founding_date, 
            founders      
        )

        cursor.execute(insert_query, data_to_insert)
        
        conn.commit()
        
    except psycopg2.Error as e:
        if conn:
            conn.rollback() 
        raise Exception(f"PostgreSQL Insertion Failed: {e}") 
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()