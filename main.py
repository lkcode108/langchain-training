from dotenv import load_dotenv
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schema import AgentResponse

load_dotenv()

llm = ChatOpenAI(model="gpt-4")
tools = [TavilySearch()]
structured_llm = llm.with_structured_output(AgentResponse)

react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=[
        "tools",
        "tool_names",
        "format_instructions",
        "input",
        "agent_scratchpad",
    ],
).partial(format_instructions="")

react_agent = create_react_agent(
    llm=llm, tools=tools, prompt=react_prompt_with_format_instructions
)

agent_executor = AgentExecutor(agent=react_agent, tools=tools, verbose=True)
extract_output = RunnableLambda(lambda x: x["output"])
chain = agent_executor | extract_output | structured_llm


def main():
    print("Hello from langchain-training!")
    results = chain.invoke(
        {
            "input": "Find top 3 AI Engineer job postings in linkedin for Bangalore location"
        }
    )
    print(results)


if __name__ == "__main__":
    main()
