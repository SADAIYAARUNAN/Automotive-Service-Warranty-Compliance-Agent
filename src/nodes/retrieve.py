import os
import atexit
from typing import Dict, Any, List
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client.grpc import Idf

from src.state import AgentState


# Initiating Embedding Models
DENSE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL_NAME = "Qdrant/bm25"
COLLECTION_NAME = "compliance_policies"

dense_embedder = TextEmbedding(model_name=DENSE_MODEL_NAME)
sparse_embedder = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)


# Initialize Qdrant client
_client = None
def get_client() -> QdrantClient:
    global _client
    if _client is None:
        qdrant_url = os.getenv("QDRANT_URL")
        _client = QdrantClient(url=qdrant_url) if qdrant_url else QdrantClient(path="./qdrant_data")
        atexit.register(_client.close)
    return _client


# Mock Data
MOCK_DATA_CLAIMS: List[Dict[str, Any]] = [
    {
      "id": 1,
      "policy_id": "WARRANTY-MOD-01",
      "category": "Modifications & Alterations",
      "content": "Any electrical or software modifications to the CAN bus network or main wiring harness by unauthorized third-party workshops immediately voids warranty coverage for infotainment and electronic control units.",
      "access_level": "Tier-1",
      "severity": "Critical"
    },
    {
      "id": 2,
      "policy_id": "TSB-INF-EU-04",
      "category": "Hardware Defects",
      "content": "Standard factory-defect display failures are fully covered under the 3-year European manufacturer warranty, provided no physical tampering or liquid ingress is detected.",
      "access_level": "Tier-1",
      "severity": "Medium"
    },
    {
      "id": 3,
      "policy_id": "INTERNAL-PRICING-SENSITIVE",
      "category": "Internal Labor Rates",
      "content": "Dealership internal warranty reimbursement labor rates are fixed at €85/hour for Tier-1 certified technicians.",
      "access_level": "Restricted-Admin",
      "severity": "Low"
    }
]



def embed_dense(text: str) -> List[float]:
    """
    Generate a 384 dimensional dense vector
    """
    return next(dense_embedder.embed([text])).tolist()


def embed_sparse(text: str) -> models.SparseVector:
    """
    Generate a BM25 Sparse Vector
    """
    sparse_vector = next(sparse_embedder.embed([text]))
    return models.SparseVector(
        indices=sparse_vector.indices.tolist(),
        values=sparse_vector.values.tolist()
    )


def seed_knowledge_base() -> None:
    """
    Creates Hybrid Collection and Indexes
    """
    db = get_client()
    if db.collection_exists(COLLECTION_NAME):
        return

    db.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense":models.VectorParams(
                size=384,
                distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            "sparse":models.SparseVectorParams(
                modifier=models.Modifier.IDF
            )
        }
    )

    if db.count(collection_name=COLLECTION_NAME).count > 0:
        return


    points = []
    for claims in MOCK_DATA_CLAIMS:
        content = claims["content"]
        points.append(
            models.PointStruct(
                id=claims["id"],
                vector={
                    "dense":embed_dense(content),
                    "sparse":embed_sparse(content)
                },
                payload={
                    "policy_id": claims["policy_id"],
                    "category": claims["category"],
                    "content": content,
                    "severity": claims["severity"],
                    "access_level": claims["access_level"]
                },
            )
        )

    db.upsert(collection_name=COLLECTION_NAME,points=points)


def query_compliance_db(state: AgentState) -> Dict[str, Any]:
    """
    Executes a Hybrid Search (Dense + Sparse BM25 with RRF Fusion) against Qdrant
    using the sanitized data enforce filtering.
    """

    query_text = state.sanitized_report

    if not query_text:
        return{
            "retrieved_claims":[],
            "error_count":state.error_count + 1
        }

    
    try:
        seed_knowledge_base()
        db = get_client()


        dense_query = embed_dense(query_text)
        sparse_query = embed_sparse(query_text)


        # Access Filter
        access_filter = models.Filter(
            must = [
                models.FieldCondition(
                    key="access_level",
                    match=models.MatchValue(value="Tier-1")
                )
            ]
        )


        # Execute Hybris Search
        response = db.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=dense_query,
                    using="dense",
                    filter=access_filter,
                    limit=5
                ),
                models.Prefetch(
                    query=sparse_query,
                    using="sparse",
                    filter=access_filter,
                    limit=5
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=2,
            with_payload=True
        )



        claims = [
            {
                "policy_id": point.payload.get("policy_id"),
                "category": point.payload.get("category"),
                "severity": point.payload.get("severity"),
                "content": point.payload.get("content"),
                "score": round(point.score,4)
            }
            for point in response.points
            if point.payload
        ]


        return {"retrieved_warranty_policies":claims}


    except Exception as e:
        print(f"[Error] Qdrant retrievel failed: {e}")
        return{
            "retreived_policies": [],
            "error_count": state.error_count + 1
        }


if __name__ == "__main__":
    mock_data = AgentState(
        sanitized_report=(
          "Service Ticket #SRV-99421-EU. Customer <PERSON_REDACTED> dropped off her vehicle at Amsterdam Central Motors located at <LOCATION_REDACTED>," 
          "1016 GC <LOCATION_REDACTED>. Contact phone: <PHONE_REDCATED>. Vehicle VIN: <VIN_REDACTED>, License Plate: 45-HZ-KJ." 
          "Customer complaint: Infotainment screen went completely blank during driving. Mechanic initial notes: Technician inspected the dashboard wiring harness and discovered" 
          "aftermarket third-party amplifier spliced directly into the main CAN bus data lines. Customer stated an independent garage installed custom audio equipment last month." 
          "Requesting full warranty coverage replacement for the central display unit and infotainment module."
        )
    )


    output = query_compliance_db(mock_data)

    print(f"\n--- Retrieved Policies (count: {len(output['retrieved_warranty_policies'])})---")
    for idx, doc in enumerate(output['retrieved_warranty_policies'], start=1):
        print(f"\n{idx}. [{doc['policy_id']}] (Severity: {doc['severity']} | Score: {doc['score']})")
        print(f"  Category: {doc['category']}")
        print(f" Rule: {doc['content']}")
        







