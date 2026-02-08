"""State definitions for the LangGraph workflow."""

from typing import TypedDict, List, Optional, Annotated
from operator import add
from pydantic import BaseModel, Field


class ResearchTask(BaseModel):
    task_id: str
    description: str
    task_type: str  # "search", "code", or "analysis"
    dependencies: List[str] = Field(default_factory=list)
    status: str = "pending"
    result: Optional[str] = None


class SearchResult(BaseModel):
    source: str
    title: str
    content: str
    relevance_score: float


class CodeExecutionResult(BaseModel):
    code: str
    output: str
    success: bool
    error: Optional[str] = None


class AgentState(TypedDict):
    query: str
    research_plan: List[ResearchTask]
    current_task_index: int
    search_results: Annotated[List[dict], add]
    code_results: Annotated[List[dict], add]
    final_report: str
    messages: Annotated[List[dict], add]
    iteration_count: int
    error: Optional[str]
    next_agent: str


def create_initial_state(query: str) -> AgentState:
    return AgentState(
        query=query,
        research_plan=[],
        current_task_index=0,
        search_results=[],
        code_results=[],
        final_report="",
        messages=[{"role": "user", "content": query}],
        iteration_count=0,
        error=None,
        next_agent="planner",
    )
