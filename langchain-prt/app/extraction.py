from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.config import llm
from app.schema import CompaniesSchema

parser = PydanticOutputParser(pydantic_object=CompaniesSchema)

extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are an expert information extractor.
    Process each paragraph independently.
    Extract one or more companies per paragraph if present.
    Return founders as a clean list of person names (no roles or extra text).
    Return the founding date text span exactly as found in the paragraph when possible.
    If the same founder is associated with multiple companies, list them for each company.
    
    Date rules:
    - If only the year is provided, use YYYY-01-01.
    - If the year and month are provided, use YYYY-MM-01.
    - Convert all dates to the YYYY-MM-DD format. Save in str(string) format. even if the date are given as july 4, etc.

    
    Format instructions: {format_instructions}"""),
    ("human", "{paragraph}")
])

extraction_chain = (
    {
        "paragraph": RunnablePassthrough(),
        "format_instructions": RunnableLambda(lambda _: parser.get_format_instructions())
    }
    | extraction_prompt 
    | llm
    | parser
)

def extract_company(paragraph: str) -> CompaniesSchema:
    """
    Extracts company data from a single paragraph using LCEL + Bedrock LLM.
    Returns a validated CompaniesSchema instance containing a list of companies.
    """
    return extraction_chain.invoke(paragraph)