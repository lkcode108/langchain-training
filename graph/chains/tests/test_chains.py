from pprint import pprint

from dotenv import load_dotenv

from graph.chains.answer_grader import answer_grader
from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.router import question_router
from ingestion import retriever

load_dotenv()


def test_retrieval_grader_answer_yes():
    question = "agent memory"
    docs = retriever.invoke(input=question)
    doctxt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"document": doctxt, "question": question}
    )

    assert res.binary_score == "yes"


def test_retrieval_grader_answer_no():
    question = "agent memory"
    docs = retriever.invoke(input=question)
    doctxt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"document": doctxt, "question": "What is Pizza"}
    )
    assert res.binary_score == "no"


def test_generation_chain():
    question = "agent memory"
    docs = retriever.invoke(input=question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)


def test_hallucination_Grade_yes():
    question = "agent memory"
    docs = retriever.invoke(input=question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    score = hallucination_grader.invoke({"documents": docs, "generation": generation})
    assert score.binary_score


def test_hallucination_grade_no():
    question = "agent memory"
    docs = retriever.invoke(input=question)
    score = hallucination_grader.invoke(
        {"documents": docs, "generation": "sdsdjshdshdjhdhjsdh"}
    )
    assert not score.binary_score


def test_answer_grade_yes():
    question = "agent memory"
    docs = retriever.invoke(input=question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    score = answer_grader.invoke({"question": question, "generation": generation})
    assert score.binary_score


def test_answer_grade_no():
    question = "agent memory"
    score = answer_grader.invoke(
        {"question": question, "generation": "Sunday is last day of the week"}
    )
    assert not score.binary_score


def test_route_to_vectorstore():
    question = "agent memory"
    output = question_router.invoke({"question": question})
    assert output.datasource == "vectorstore"


def test_route_to_websearch():
    question = "pizza making"
    output = question_router.invoke({"question": question})
    assert output.datasource == "web_search"
