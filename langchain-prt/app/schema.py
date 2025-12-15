from typing import List
from pydantic import BaseModel, Field

class CompanySchema(BaseModel):
    company_name: str = Field(description="The full legal name of the company.")
    founding_date: str = Field(description="The company's founding date as string. If month/day not given, take january 1 as default. Keep the date in YYYY-MM-DD as str. If empty, keep as an empty string.")
    founders: List[str] = Field(description="A list of the names of the company's founders.")

class CompaniesSchema(BaseModel):
    companies: List[CompanySchema] = Field(description="A list of companies extracted from the paragraph.")