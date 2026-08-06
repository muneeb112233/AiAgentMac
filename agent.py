"""
agent.py — builds the LangGraph agent. Streamlit caches the result of
build_agent() so it only runs ONCE per app session (see app.py).
"""

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from config import llm_google, SYS_PROMPT
from tools import tools_registry


def build_agent():
    """Creates a fresh agent + in-memory checkpointer. Call once, then reuse the returned graph."""
    memory = InMemorySaver()
    graph = create_agent(
        name="Email Writing Agent",
        model=llm_google,
        tools=tools_registry,
        system_prompt=SYS_PROMPT,
        checkpointer=memory,
    )
    return graph
