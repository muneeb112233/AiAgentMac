"""
app.py — Streamlit UI for Frankenstein, the Email Assistant agent.
Run with: streamlit run app.py
"""

import json
import uuid

import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from agent import build_agent
from tools import tools_registry, send_email_tool

st.set_page_config(page_title="Frankenstein — Loyal Assistant", page_icon="✉️", layout="wide")

# ------------------------------------------------------------------
# CACHED AGENT + SESSION STATE
# ------------------------------------------------------------------
# @st.cache_resource runs build_agent() ONCE for the app process, not on
# every rerun. Streamlit reruns this entire script top-to-bottom on every
# click/input — without caching, InMemorySaver() would be recreated every
# time and all chat memory would be wiped instantly.
@st.cache_resource
def get_agent():
    return build_agent()


graph = get_agent()

# Each browser session gets its own thread_id so LangGraph's checkpointer
# keeps separate memory per user. Stored in session_state so it's stable
# across reruns within the same session, but unique per session.
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []  # display-only history: [{"role", "content"}]

if "pending_draft" not in st.session_state:
    st.session_state.pending_draft = None  # last unsent draft dict, or None

config: RunnableConfig = {"configurable": {"thread_id": st.session_state.thread_id}}

# ------------------------------------------------------------------
# SIDEBAR — agent status, model, tools
# ------------------------------------------------------------------
with st.sidebar:
    st.title(" Frankenstein")
    st.caption("Your loyal email assistant")

    st.markdown("### Status")
    st.success("● Agent Online")

    st.markdown("### Model")
    st.code("gemini-flash-lite-latest", language=None)

    st.markdown("### Available Tools")
    for t in tools_registry:
        st.markdown(f"- `{t.name}`")

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_draft = None
        st.session_state.thread_id = str(uuid.uuid4())  # fresh memory thread too
        st.rerun()

# ------------------------------------------------------------------
# HELPERS — pull useful data out of graph.stream() chunks
# ------------------------------------------------------------------
def extract_draft_from_chunk(chunk: dict):
    """If this chunk contains a draft_email_tool result, return it as a dict."""
    tool_data = chunk.get("tools")
    if not tool_data:
        return None
    for msg in tool_data.get("messages", []):
        if getattr(msg, "name", None) == "draft_email_tool":
            content = msg.content
            try:
                return json.loads(content) if isinstance(content, str) else content
            except (json.JSONDecodeError, TypeError):
                return None
    return None


def extract_final_reply(chunk: dict):
    """If this chunk contains the agent's latest text reply, return it."""
    agent_data = chunk.get("agent")
    if not agent_data:
        return None
    for msg in agent_data.get("messages", []):
        content = getattr(msg, "content", None)
        if content:
            return content
    return None


# ------------------------------------------------------------------
# MAIN CHAT AREA
# ------------------------------------------------------------------
st.header("✉️ Chat with Frankenstein")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Draft review card — appears when the agent has produced an unsent draft
if st.session_state.pending_draft:
    draft = st.session_state.pending_draft
    with st.container(border=True):
        st.markdown("### 📝 Draft Ready for Review")
        st.markdown(f"**To:** {draft.get('to', '—')}")
        st.markdown(f"**Subject:** {draft.get('subject', '—')}")
        st.markdown("**Body:**")
        st.info(draft.get("body", ""))

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Approve & Send", use_container_width=True, type="primary"):
                result = send_email_tool.invoke({
                    "recipient": draft.get("to", ""),
                    "subject": draft.get("subject", ""),
                    "contents": draft.get("body", ""),
                })
                st.session_state.messages.append({"role": "assistant", "content": f"📤 {result}"})
                st.session_state.pending_draft = None
                st.rerun()
        with col2:
            if st.button("❌ Discard", use_container_width=True):
                st.session_state.messages.append({"role": "assistant", "content": "🗑️ Draft discarded."})
                st.session_state.pending_draft = None
                st.rerun()

# Chat input
user_input = st.chat_input("Tell Frankenstein who to email and what to say...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_reply = ""

        with st.spinner("Frankenstein is working..."):
            for chunk in graph.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="updates",
            ):
                draft = extract_draft_from_chunk(chunk)
                if draft and "error" not in draft:
                    st.session_state.pending_draft = draft

                reply = extract_final_reply(chunk)
                if reply:
                    full_reply = reply
                    placeholder.markdown(full_reply)

        if not full_reply:
            full_reply = "Done."
            placeholder.markdown(full_reply)

    st.session_state.messages.append({"role": "assistant", "content": full_reply})
    st.rerun()  # rerun so the draft card (if any) renders in the right place
