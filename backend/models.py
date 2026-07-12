from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    """Request model for a user query against the RAG pipeline.

    Attributes:
        query: the user's natural language question
        top_k: how many top documents to retrieve (hybrid retriever)
        use_semantic: whether to include semantic retrieval
    """
    query: str = Field(..., description="User question")
    top_k: int = Field(5, description="Number of top documents to retrieve")
    use_semantic: bool = Field(True, description="Include semantic retrieval")

class SourceItem(BaseModel):
    """Information about a retrieved source document."""
    id: str
    score: float
    text: Optional[str]
    metadata: Optional[dict]

class QueryResponse(BaseModel):
    """Response model returned by the API.

    Attributes:
        query: original query string
        answer: generated answer text
        sources: list of SourceItem describing retrieved documents
    """
    query: str
    answer: str
    sources: List[SourceItem] = []
