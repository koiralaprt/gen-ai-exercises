from langchain.tools import tool
from app.database import insert_company
from datetime import date
from typing import List

@tool
def db_insert_tool(
    company_name: str, 
    founding_date: str, 
    founders: List[str]
) -> str:
    """Inserts structured company data..."""
    try:
        insert_company(
            company_name=company_name,
            founding_date=founding_date,
            founders=founders
        )
        return f"Successfully inserted company: {company_name}"
    except Exception as e:
        error_message = f"Database Insertion Failed for {company_name}. Error: {e.__class__.__name__}: {e}"
        print(f"ERROR: {error_message}") 
        return error_message 