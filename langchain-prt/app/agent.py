from langchain.agents import create_agent
from app.tools import db_insert_tool
from app.config import llm

tools = [db_insert_tool]
SYSTEM_PROMPT = """
You are a PostgreSQL data insertion bot. Your SOLE function is to receive data 
from the user and immediately use the available 'insert_company_to_postgres' 
tool to store it. Do NOT respond with natural language confirmation; only use 
the tool with the provided input.
"""

agent_executor = create_agent(
    tools=tools,
    model=llm,
    system_prompt=SYSTEM_PROMPT,
)