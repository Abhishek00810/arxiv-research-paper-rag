from typing import TypedDict, List


class AgentState(TypedDict):
    query: str
    original_querya: str
    retrieved_chunks: List[dict]
