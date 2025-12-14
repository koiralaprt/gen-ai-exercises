import os
from dotenv import load_dotenv
load_dotenv()
from langchain_aws import ChatBedrock
llm = ChatBedrock(
    model_id=os.getenv("MODEL_ID"),
    region_name="ap-south-1",
    provider="amazon",
    model_kwargs={"temperature": 0}
)