from langchain_classic import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(temperature=0)

generation_chain = prompt | llm | StrOutputParser()
