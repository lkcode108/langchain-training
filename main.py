from dotenv import load_dotenv
from langchain.tools import tool
from langchain_classic.agents.format_scratchpad import format_log_to_str
from langchain_classic.agents.output_parsers import \
    ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import render_text_description
from langchain_openai import ChatOpenAI

from callback import AgentCallbackHandler

load_dotenv()


@tool
def get_length_of_text(text: str) -> int:
    """Return the length of a text by characters"""
    print(f"the input text is {text}")
    text = text.strip("'\n").strip('"')
    return len(text)


def find_the_tool_to_use(tools: [], tool_name: str):
    tools = tools
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"No Tool with {tool_name} has been found")


def main():
    print("Hello from langchain-training!")
    llm = ChatOpenAI(
        temperature=0,
        model_kwargs={"stop": ["\nObservation", "Observation"]},
        callbacks=[AgentCallbackHandler()],
    )
    intermediary_steps = []
    tools = [get_length_of_text]
    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {input}
    Thought: {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=",".join([t.name for t in tools]),
    )

    agent_step = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )

    agent_step_results = ""
    while not isinstance(agent_step_results, AgentFinish):
        agent_step_results: AgentAction | AgentFinish = agent_step.invoke(
            {
                "input": "What is the length in characters of the text: DOG?",
                "agent_scratchpad": intermediary_steps,
            }
        )
        print(agent_step_results)

        if isinstance(agent_step_results, AgentAction):
            tool_name = agent_step_results.tool
            tool_to_use = find_the_tool_to_use(tools, tool_name)
            tool_input = agent_step_results.tool_input
            observation = tool_to_use.func(tool_input)
            intermediary_steps.append((agent_step_results, str(observation)))
            print(observation)

    if isinstance(agent_step_results, AgentFinish):
        print(agent_step_results.return_values)


if __name__ == "__main__":
    main()
