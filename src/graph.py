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


# Initilaize Graph
workflow  = StateGraph(AgentState)


# Add Nodes
workflow.add_node("sanitize", sanitize_data)
workflow.add_node("retrieve",query_compliance_db)
workflow.add_node("draft",claim_assessment)


# Add Edge Pipeline
workflow.set_entry_point("sanitize")
workflow.add_edge("sanitize","retrieve")
workflow.add_edge("retrieve","draft")
workflow.add_edge("draft",END)


# SetUp SQLite for persistent state storage and recovery
db_conn = sqlite3.connect("aswca_checkpoints.sqlite", check_same_thread=False)
memory=SqliteSaver(db_conn)

app = workflow.compile(checkpointer=memory)


if __name__ == "__main__":

    config = {"configurable": {"thread_id":"test-session-001"}}

    input = {
        "raw_service_report" : (
            "Service Ticket #SRV-99421-EU. Customer Sarah Van Dijk dropped off her vehicle at Amsterdam Central Motors located at Keizersgracht 424," 
            "1016 GC Amsterdam. Contact phone: +31-20-555-0199. Vehicle VIN: WBA5SER7890123456, License Plate: 45-HZ-KJ. Customer complaint: Infotainment screen went completely blank during driving." 
            "Mechanic initial notes: Technician inspected the dashboard wiring harness and discovered aftermarket third-party amplifier spliced directly into the main CAN bus data lines. Customer stated an independent garage installed custom audio equipment last month." 
            "Requesting full warranty coverage replacement for the central display unit and infotainment module."

        )
    }


    final_state = app.invoke(input, config=config)


    print("\n[RESULT] Sanitized Report:")
    print(final_state["sanitized_report"])

    print(f"\n[RESULT] Policies Retrieved: {len(final_state['retrieved_warranty_policies'])}")

    print("\n[RESULT] Claim Assessment Summary:")

    print(final_state["warrant_claim_assessment"])