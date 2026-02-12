from langgraph.graph import END,StateGraph
from graph.consts import GENERATE,RETRIEVE,GRADE_DOCUMENTS,WEB_SEARCH
from graph.nodes import generate,grade_documents,retrieve,web_search
from graph.state import GraphState

from dotenv import load_dotenv

load_dotenv()

def decide_to_generate(state):
    print("------ASSESS GRADED DOCUMENTS---")
    if state["web_search"]:
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return WEB_SEARCH
    else:
        return GENERATE

workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(WEB_SEARCH,web_search)
workflow.add_node(GENERATE,generate)

workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE,GRADE_DOCUMENTS)
workflow.add_conditional_edges(GRADE_DOCUMENTS,decide_to_generate,path_map=[WEB_SEARCH,GENERATE])

workflow.add_edge(WEB_SEARCH,GENERATE)
workflow.add_edge(GENERATE,END)

app = workflow.compile()
app.get_graph().draw_mermaid_png(output_file_path="workflow.png")

