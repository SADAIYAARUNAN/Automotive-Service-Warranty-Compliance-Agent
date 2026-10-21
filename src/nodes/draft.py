import os
from dotenv import load_dotenv
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from src.state import AgentState


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
llm=ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=api_key
)


def claim_assessment(state: AgentState) -> Dict[str, Any]:
    """
    Combines the sanitized data and Retrieved Qdrant Compliance policies
    to generate a structured risk assessment summary. 
    """

    sanitized_text = state.sanitized_report
    policies = state.retrieved_warranty_policies


    if not sanitized_text:
        return{
            "warrant_claim_assessment": "Error: No Sanitized data available for assessment."
        }

    formatted_policies = "\n".join(
        f"-[{p['policy_id']}] (Severity: {p['severity']}: {p['content']})"
        for p in policies
    )if policies else "No Matching Policies Found."


    prompt=f"""
    You are an expert AI compliance and Legal Research Agent for analyzing vehicle service logs , reports and claims.
    Review the sanitized data log against the retrieved warranty claim policies.


    === SANITIZED DATA ===
    {sanitized_text}


    === RETRIEVED POLICIES ===
    {formatted_policies}

    === INSTRUCTIONS ===
    Provide a professional compliance risk assessment containing:
    1. Violation Status (Compliant / Non-Compliant / Review Required)
    2. Applicable Policy Risks identified from the retrieved database
    3. Recommended remediation steps for the engineering team.
    """

    try:
        response = llm.invoke(
            [
                SystemMessage(
                    content="You are a precise legal and regulatory compliance engineering assistant."
                ),HumanMessage(content=prompt)
            ]
        )
        return{
            "warrant_claim_assessment":response.content,
            "message": [response]
        }
    except Exception as e:
        print(f"[ERROR] LLM generation Failed: {e}")
        return{
            "warrant_claim_assessment": f"failed to generate assessment due to API error: {e}",
            "error_count": state.error_count
        }