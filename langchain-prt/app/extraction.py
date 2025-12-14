from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.config import llm
from app.schema import CompanySchema

parser = PydanticOutputParser(pydantic_object=CompanySchema)

extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert data extraction bot. Your task is to accurately extract company details from the user's text and format the output as a JSON object strictly following the provided format instructions.
    
    Date rules:
    - If only the year is provided, use YYYY-01-01.
    - If the year and month are provided, use YYYY-MM-01.
    - Convert all dates to the YYYY-MM-DD format. even if the date are given as july 4, etc.

    
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

def extract_company(paragraph: str) -> CompanySchema:
    """
    Extracts company data from a single paragraph using LCEL + Bedrock LLM.
    Returns a validated CompanySchema instance.
    """
    return extraction_chain.invoke(paragraph)