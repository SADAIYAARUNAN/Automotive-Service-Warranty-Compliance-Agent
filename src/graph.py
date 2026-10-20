import sqlite3
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver


from src.state import AgentState
from src.nodes.sanitize import sanitize_data
from src.nodes.retrieve import query_compliance_db
from src.nodes.draft import claim_assessment


def error_counting(state: AgentState) -> str:
    """
    Checks if tool failures have occured.
    If the threshold is exceeded routes to END , otherwise continue.
    """

    if state.error_count >=3:
        return "end"
    return "continue"