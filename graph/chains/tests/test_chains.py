from dotenv import load_dotenv
from pprint import pprint
from ingestion import retriever
from graph.chains.retrieval_grader import GradeDocuments,retrieval_grader
from graph.chains.generation import generation_chain
load_dotenv()

def test_retrieval_grader_answer_yes():
    question = "agent memory"
    docs = retriever.invoke(input=question)
    doctxt= docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke({"document":doctxt, "question":question})

    assert res.binary_score == "yes"

def test_retrieval_grader_answer_no():
    question = "agent memory"
    docs = retriever.invoke(input=question)
    doctxt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke({"document":doctxt,"question":"What is Pizza"})
    assert res.binary_score == "no"

def test_generation_chain():
    question = "agent memory"
    docs = retriever.invoke(input=question)
    generation = generation_chain.invoke({"context":docs,"question":question})
    pprint(generation)