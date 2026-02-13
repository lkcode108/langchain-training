from typing import Literal

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


class RouterQuery(BaseModel):
    """Route a user query to most relevant datasource"""

    datasource: Literal["web_search", "vectorstore"] = Field(
        ...,
        description="Given a user question choose to route it to websearch or vectorstore.",
    )


llm = ChatOpenAI(temperature=0)
structured_llm_router = llm.with_structured_output(RouterQuery)

system = """ You are an expert at routing a user question to vectorstore or web_search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions related to these topics. For all else, use web_search.
"""
route_prompt = ChatPromptTemplate.from_messages(
    [("system", system), ("human", "{question}")]
)

question_router = route_prompt | structured_llm_router
