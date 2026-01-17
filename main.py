from langchain_openai import ChatOpenAI
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4")
tools= [TavilySearch()]
prompt = hub.pull("hwchase17/react")

react_agent= create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=react_agent,
    tools=tools,
    verbose=True
)

chain = agent_executor

def main():
    print("Hello from langchain-training!")
    results = chain.invoke({"input":"Find top 3 AI Engineer job postings in linkedin for Bangalore location"})
    print(results)

if __name__ == "__main__":
    main()
