from typing import Annotated, List, Any, Dict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(BaseModel):

    # Ensures new messages are appended and not overwritten
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)

    # Raw Input from FastAPI Endpoint
    raw_service_report: str = Field(default="")

    # Scrubbed text
    sanitized_report: str  = Field(default="")

    # Claims pulled for Hybrid Search
    retrieved_warranty_policies: List[Dict[str,Any]] = Field(default_factory=list)

    # Tracks Tool Failures to avoid infinite Loop
    error_count: int = Field(default=0)

    # The Final LLm Output
    warrant_claim_assessment: str = Field(default="")