from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import Tool, tool
from langchain_openai import ChatOpenAI

from callback import AgentCallbackHandler

load_dotenv()


@tool
def get_length_of_text(text: str) -> int:
    """Return the length of a text by characters"""
    print(f"the input text is {text}")
    text = text.strip("'\n").strip('"')
    return len(text)


def find_the_tool_to_use(tools: [Tool], tool_name: str):
    tools = tools
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"No Tool with {tool_name} has been found")


def main():
    print("Hello from langchain-training!")
    llm = ChatOpenAI(
        temperature=0,
        callbacks=[AgentCallbackHandler()],
    )
    tools = [get_length_of_text]
    llm_with_tools = llm.bind_tools(tools)

    # Start conversation
    messages = [
        HumanMessage(content="What is the length in characters of the text: DOG?")
    ]

    while True:
        ai_message = llm_with_tools.invoke(messages)
        # if model decides to call tools , executes them and returns the results
        tool_calls = getattr(ai_message, "tool_calls", None) or []
        if len(tool_calls) > 0:
            messages.append(ai_message)
            for tool_call in tool_calls:
                # tool_call is typically a dictionary with keys: id, type, name, args
                tool_name = tool_call.get("name")
                tool_to_use = find_the_tool_to_use(tools, tool_name)
                tool_args = tool_call.get("args", {})
                tool_call_id = tool_call.get("id")
                observation = tool_to_use.invoke(tool_args)
                print(f"observation: {observation}")
                messages.append(
                    ToolMessage(content=str(observation), tool_call_id=tool_call_id)
                )
            # Continue the loop so that model uses the observation
            continue

        # print the final answer if model decides no tool_call
        print(ai_message.content)
        break


if __name__ == "__main__":
    main()
