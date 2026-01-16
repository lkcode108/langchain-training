from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage
from langchain_tavily import TavilySearch

from dotenv import load_dotenv
load_dotenv()

tools = [TavilySearch()]
llm = ChatOpenAI()
agent = create_agent(model=llm,tools=tools)

def main():
    print("Hello from langchain-training!")
    results = agent.invoke({"messages": HumanMessage(content="How is Tokyo's weather currently?") })
    print(results)
if __name__ == "__main__":
    main()
