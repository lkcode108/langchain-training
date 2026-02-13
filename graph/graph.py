from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import question_router
from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEB_SEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState

load_dotenv()


def decide_route_to_start(state):
    question = state["question"]
    route = question_router.invoke({"question": question})
    if route.datasource == WEB_SEARCH:
        print("ROUTE TO WEBSEARCH")
        return WEB_SEARCH
    else:
        print("ROUTE TO VETORSTORE")
        return RETRIEVE


def decide_to_generate(state):
    print("------ASSESS GRADED DOCUMENTS---")
    if state["web_search"]:
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return WEB_SEARCH
    else:
        return GENERATE


def decide_based_on_hallucination_answer_grades(state):
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_grade := score.binary_score:
        print("ANSWER GROUNDED IN / SUPPORTED BY RETRIEVED DOCUMENTS")
        answerscore = answer_grader.invoke(
            {"question": question, "generation": generation}
        )

        if answer_grade := answerscore.binary_score:
            print("ANSWER ADDRESSES THE QUESTION")
            return "useful"
        else:
            return "not useful"

    else:
        return "not supported"


workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(WEB_SEARCH, web_search)
workflow.add_node(GENERATE, generate)

workflow.set_conditional_entry_point(
    decide_route_to_start, path_map={WEB_SEARCH: WEB_SEARCH, RETRIEVE: RETRIEVE}
)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS, decide_to_generate, path_map=[WEB_SEARCH, GENERATE]
)
workflow.add_conditional_edges(
    GENERATE,
    decide_based_on_hallucination_answer_grades,
    path_map={"useful": END, "not useful": WEB_SEARCH, "not supported": GENERATE},
)

workflow.add_edge(WEB_SEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()
app.get_graph().draw_mermaid_png(output_file_path="workflow.png")
