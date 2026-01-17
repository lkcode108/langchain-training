from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage
from langchain_tavily import TavilySearch

from typing import List
from pydantic import BaseModel,Field
from dotenv import load_dotenv
load_dotenv()

class Source(BaseModel):
    """ schema for the source of the agent response"""
    url:str = Field("URL of the source")

class AgentResponse(BaseModel):
    """ schema for the agent response with answer and sources"""
    answer:str = Field("The agent's answer to the query")
    sources:List[Source] = Field( default_factory=list,description="List of sources used to generate the answer")

tools = [TavilySearch()]
llm = ChatOpenAI()
agent = create_agent(model=llm,tools=tools,response_format=AgentResponse)

def main():
    print("Hello from langchain-training!")
    results = agent.invoke({"messages": HumanMessage(content="What are the top 3 AI Engineer job requirements in linkedIn for Bangalore location?") })
    print(results)
    print("######################################################")
    print(results['structured_response'].answer)
    print(results['structured_response'].sources)
if __name__ == "__main__":
    main()
