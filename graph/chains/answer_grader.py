from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


class GradeAnswer(BaseModel):
    """Binary score to check that the answer addresses the question"""

    binary_score: bool = Field(
        description="Answer addresses the question. 'yes' or 'no'"
    )


llm = ChatOpenAI(temperature=0)

structured_llm_answer = llm.with_structured_output(GradeAnswer)

system = """You are a grader assessing whether an answer addresses / resolves a question \n 
Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_answer
