from typing import List
from pydantic import BaseModel, Field
from datetime import date

class CompanySchema(BaseModel):
    company_name: str = Field(description="The full legal name of the company.")
    # forcing YYYY-MM-DD string and convert it to a Python date object
    founding_date: date = Field(description="The company's founding date, strictly in YYYY-MM-DD format.")
    founders: List[str] = Field(description="A list of the names of the company's founders.")