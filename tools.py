"""
tools.py — every @tool the agent can call. Pulled straight from the notebook,
just re-homed here so app.py / agent.py can import them cleanly.
"""

from datetime import datetime
from typing import Any

from zoneinfo import ZoneInfo
from langchain_core.tools import tool
from serpapi import GoogleSearch
from imap_tools import MailBox, AND

from config import yag, EMAIL, APP_PASSWORD, SERP_API_KEY, CONTACTS, llm_google


@tool
def send_email_tool(recipient: str, subject: str, contents: str) -> str:
    """Sends an email to the recipient with the specified subject and contents."""
    try:
        yag.send(to=recipient, subject=subject, contents=contents)
        return "Email sent!"
    except Exception as e:
        return f"Error sending email: {e}"


@tool
def serpapi_search(query: str) -> list[dict[str, Any]]:
    """Searches for a query using the SerpAPI on Google."""
    try:
        if not SERP_API_KEY:
            return [{"error": "SERP_API_KEY is missing or empty"}]

        params = {"q": query, "hl": "en", "gl": "us", "api_key": SERP_API_KEY}
        search = GoogleSearch(params)
        results = search.get_dict()

        if "organic_results" in results and results["organic_results"]:
            return [
                {"title": r["title"], "link": r["link"], "snippet": r.get("snippet", "")}
                for r in results["organic_results"][:5]
            ]
        return [{"error": "No results found"}]
    except Exception as e:
        return [{"error": f"Failed to perform search: {e}"}]


@tool
def read_inbox_tool(count: int = 5) -> list:
    """Reads the most recent emails from the inbox using imap-tools."""
    try:
        if not EMAIL or not APP_PASSWORD:
            return [{"error": "EMAIL or APP_PASSWORD environment variable is not set"}]
        results = []
        with MailBox("imap.gmail.com").login(EMAIL, APP_PASSWORD) as mailbox:
            for msg in mailbox.fetch(AND(all=True), reverse=True, limit=count):
                results.append({
                    "from": msg.from_,
                    "subject": msg.subject,
                    "snippet": (msg.text or msg.html or "")[:200],
                })
        return results
    except Exception as e:
        return [{"error": f"Failed to read inbox: {e}"}]


@tool
def draft_email_tool(recipient: str, subject: str, contents: str) -> dict:
    """Creates a draft email (does NOT send it) so it can be reviewed before sending."""
    return {
        "to": recipient,
        "subject": subject,
        "body": contents,
        "status": "draft - not sent yet",
    }


@tool
def send_email_with_attachment_tool(recipient: str, subject: str, contents: str, file_path: str) -> str:
    """Sends an email with a single file attached, using yagmail."""
    try:
        yag.send(to=recipient, subject=subject, contents=contents, attachments=file_path)
        return "Email with attachment sent!"
    except Exception as e:
        return f"Error sending email with attachment: {e}"


@tool
def get_current_datetime_tool(timezone: str = "Asia/Karachi") -> str:
    """Returns the current date and time, so the agent doesn't have to guess it."""
    try:
        now = datetime.now(ZoneInfo(timezone))
        return now.strftime("%A, %d %B %Y, %I:%M %p")
    except Exception as e:
        return f"Error getting current datetime: {e}"


@tool
def summarize_thread_tool(thread_text: str) -> str:
    """Summarizes a long email thread into a short paragraph using the LLM."""
    try:
        prompt = (
            "Summarize this email thread in 3-4 short sentences, "
            f"focusing on key points and any action items:\n\n{thread_text}"
        )
        response = llm_google.invoke(prompt)
        return str(response.content) if hasattr(response, "content") else str(response)
    except Exception as e:
        return f"Error summarizing thread: {e}"


@tool
def contact_lookup_tool(name: str) -> str:
    """Finds an email address by contact name from the contact book."""
    clean_name = name.strip().lower()
    if clean_name in CONTACTS:
        return CONTACTS[clean_name]
    return f"Contact '{name}' not found. Available contacts: {', '.join(CONTACTS.keys())}"


tools_registry = [
    draft_email_tool,
    send_email_tool,
    send_email_with_attachment_tool,
    serpapi_search,
    read_inbox_tool,
    get_current_datetime_tool,
    summarize_thread_tool,
    contact_lookup_tool,
]
