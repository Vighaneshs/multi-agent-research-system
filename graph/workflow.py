"""LangGraph workflow — wires the agents into a state graph."""

from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from graph.state import AgentState
from agents.planner_agent import PlannerAgent
from agents.search_agent import SearchAgent
from agents.code_agent import CodeAgent
from agents.synthesis_agent import SynthesisAgent
from config import settings


planner = PlannerAgent()
searcher = SearchAgent()
coder = CodeAgent()
synthesizer = SynthesisAgent()


# ── node functions ──────────────────────────────────────────────

def planner_node(state: AgentState) -> AgentState:
    print("Planner: creating research plan...")
    result = planner.run(state["query"])
    return {
        "research_plan": result["tasks"],
        "messages": [{"role": "assistant", "content": f"Created {len(result['tasks'])} tasks"}],
        "next_agent": "search" if result["tasks"] else "synthesize",
    }


def search_node(state: AgentState) -> AgentState:
    print("Search: retrieving information...")
    search_tasks = [t for t in state["research_plan"] if t.task_type == "search"]

    results = []
    for task in search_tasks:
        if task.status == "pending":
            results.extend(searcher.run(task.description))
            task.status = "completed"

    has_code = any(t.task_type == "code" for t in state["research_plan"])
    return {
        "search_results": results,
        "messages": [{"role": "assistant", "content": f"Found {len(results)} search results"}],
        "next_agent": "code" if has_code else "synthesize",
    }


def code_node(state: AgentState) -> AgentState:
    print("Code: running analysis...")
    code_tasks = [t for t in state["research_plan"] if t.task_type == "code"]

    results = []
    for task in code_tasks:
        if task.status == "pending":
            results.append(coder.run(task.description, context=state["search_results"]))
            task.status = "completed"

    return {
        "code_results": results,
        "messages": [{"role": "assistant", "content": f"Executed {len(results)} code tasks"}],
        "next_agent": "synthesize",
    }


def synthesis_node(state: AgentState) -> AgentState:
    print("Synthesis: writing report...")
    report = synthesizer.run(
        query=state["query"],
        search_results=state["search_results"],
        code_results=state["code_results"],
    )
    return {
        "final_report": report,
        "messages": [{"role": "assistant", "content": "Research complete!"}],
        "next_agent": "end",
    }


# ── routing ─────────────────────────────────────────────────────

def should_continue(state: AgentState) -> Literal["search", "code", "synthesize", "end"]:
    if state.get("error"):
        return "end"

    if state.get("iteration_count", 0) > settings.max_iterations:
        print("Max iterations reached — forcing synthesis")
        return "synthesize"

    mapping = {"search": "search", "code": "code", "synthesize": "synthesize"}
    return mapping.get(state.get("next_agent", "end"), "end")


# ── graph construction ──────────────────────────────────────────

def create_research_workflow() -> StateGraph:
    wf = StateGraph(AgentState)

    wf.add_node("planner", planner_node)
    wf.add_node("search", search_node)
    wf.add_node("code", code_node)
    wf.add_node("synthesize", synthesis_node)

    wf.set_entry_point("planner")

    wf.add_conditional_edges(
        "planner", should_continue,
        {"search": "search", "code": "code", "synthesize": "synthesize", "end": END},
    )
    wf.add_conditional_edges(
        "search", should_continue,
        {"code": "code", "synthesize": "synthesize", "end": END},
    )
    wf.add_conditional_edges(
        "code", should_continue,
        {"synthesize": "synthesize", "end": END},
    )
    wf.add_edge("synthesize", END)

    return wf


def compile_workflow(checkpointing: bool = True):
    wf = create_research_workflow()
    if checkpointing:
        return wf.compile(checkpointer=MemorySaver())
    return wf.compile()
