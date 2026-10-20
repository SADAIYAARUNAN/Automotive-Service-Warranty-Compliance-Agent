import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv


load_dotenv()


from src.graph import app as compiled_graph


app = FastAPI(
    title="Automotive Service & Warranty Compliance Agent (ASWCA)",
    description="production grade AI Agent API for legal service report sanitization and compliance evaluation.",
    version="1.0.0"
)


class ClaimRequest(BaseModel):
    raw_service_report: str
    thread_id: str = "default_session"

@app.post("/run-compliance-review")
async def run_compliance_review(request: ClaimRequest):
    """
    Triggers the ASWCA LangGraph pipeline:
    1. Sanitizes raw data (Presidio)
    2. Queries compliance policies via hybrid search (Qdrant + RRF)
    3. Drafts structured claim assessment (LLM)
    """

    try:
        config = {"configurable": {"thread_id": request.thread_id}}
        input = {"raw_service_report": request.raw_service_report}

        final_state = compiled_graph.invoke(input, config=config)

        return{
            "status":"success",
            "thread_id": request.thread_id,
            "raw_service_report": final_state.get("raw_service_report"),
            "retrieved_warranty_policies": final_state.get("retrieved_warranty_policies"),
            "warrant_claim_assessment": final_state.get("warrant_claim_assessment"),
            "error_count": final_state.get("error_count")
        }
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))


@app.get("/thread/{thread_id}")
async def get_thread_state(thread_id: str):
    """
    Retrieves the current saved checkpoint state for a given thread ID,
    demonstrating production state management and recovery.
    """
    try:
        config = {"configurable":{"thread_id": thread_id}}
        state_snapshot = compiled_graph.get_state(config)
        if not state_snapshot.values:
            return {"thread_id": thread_id,"status":"No active status found."}
        return{
            "thread_is":thread_id,
            "values":state_snapshot.values,
            "next_steps":state_snapshot.next
        }
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app",host="0.0.0.0",port=8000,reload=True)