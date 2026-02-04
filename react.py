from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()

@tool()
def triple(num: float) -> float:
    """
    description: Triple the input number
    param num: a number to triple
    return: triple of the input number
    """
    return 3*num

tools = [TavilySearch(max_results=1),triple]
llm = ChatOpenAI(model="gpt-40-mini",temperature=0).bind_tools(tools=tools)