"""LangGraph 구성: 노드·엣지·조건부 라우팅. (담당: 공통·통합)"""
from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from agents.domain_evaluation import domain_evaluation_agent
from agents.market_evaluation import market_evaluation_agent
from agents.stakeholder_evaluation import stakeholder_evaluation_agent
from agents.synthesis import report_generation_agent, synthesis_agent, validation_judge
from agents.technical_research import technical_research_agent
from agents.technology_selection import technology_selection_agent
from state import AgentState


def route_after_validation(state: AgentState) -> Literal["retry", "report"]:
    return "retry" if state["validation_result"] == "retry" else "report"


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("technology_selection", technology_selection_agent)
    builder.add_node("technical_research", technical_research_agent)
    builder.add_node("market_evaluation", market_evaluation_agent)
    builder.add_node("stakeholder_evaluation", stakeholder_evaluation_agent)
    builder.add_node("domain_evaluation", domain_evaluation_agent)
    builder.add_node("synthesis", synthesis_agent)
    builder.add_node("validation", validation_judge)
    builder.add_node("report_generation", report_generation_agent)

    builder.add_edge(START, "technology_selection")
    builder.add_edge("technology_selection", "technical_research")

    # Fan-out
    builder.add_edge("technical_research", "market_evaluation")
    builder.add_edge("technical_research", "stakeholder_evaluation")
    builder.add_edge("technical_research", "domain_evaluation")

    # Fan-in
    builder.add_edge(
        ["market_evaluation", "stakeholder_evaluation", "domain_evaluation"],
        "synthesis",
    )
    builder.add_edge("synthesis", "validation")
    builder.add_conditional_edges(
        "validation",
        route_after_validation,
        {"retry": "technical_research", "report": "report_generation"},
    )
    builder.add_edge("report_generation", END)

    return builder.compile()
