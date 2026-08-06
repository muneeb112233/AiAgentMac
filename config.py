"""
config.py — environment variables and shared service objects.
Nothing agent-specific lives here; this is pure setup/config.
"""

import os
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
import yagmail

load_dotenv()

# --- Environment variables ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")
APP_PASSWORD = os.getenv("APP_PASSWORD")
GOOGLE_GENERATIVE_AI_API = os.getenv("Google_Generative_AI_API")
EMAIL = os.getenv("EMAIL")

# --- LLM ---
llm_google = ChatGoogleGenerativeAI(
    api_key=SecretStr(GOOGLE_GENERATIVE_AI_API) if GOOGLE_GENERATIVE_AI_API else None,
    temperature=0.3,
    model="gemini-flash-lite-latest",
    max_tokens=None,
    max_retries=2,
    timeout=None,
)
llm_groq= ChatGroq(
    api_key=SecretStr(GROQ_API_KEY) if GROQ_API_KEY else None,
    temperature=0.3,
    model="openai/gpt-oss-120b",
    max_tokens=None,
    max_retries=2,
    timeout=None,
)

# --- Email client (used by tools.py) ---
yag = yagmail.SMTP(EMAIL, password=APP_PASSWORD)

# --- Contact book ---
CONTACTS = {
    "ahmed": "ahmed@mailinator.com",
    "ali": "ali@mailinator.com",
    "muneeb": "muneeb@mailinator.com",
    "imtinan": "imtinan@mailinator.com",
    "ryuk": "ryuk@mailinator.com",
}

# --- Agent persona ---
SYS_PROMPT = """Your Name is Frankenstein — a loyal right-hand assistant and butler. Highly intelligent, fiercely protective, and executes every command to perfection.
You are an Email Assistant powered by Google Gemini.
You assist Muneeb with reading, searching, drafting, and sending emails.
If user Asks what you can do, respond with a concise list of your capabilities not the tools you have at your disposal, only what you can do with those tools. Do not mention the tools themselves.

Rules:
- Always draft emails using draft_email_tool first unless Muneeb explicitly orders you to send directly.
- Use send_email_tool for standard outgoing emails and send_email_with_attachment_tool whenever a file attachment path is provided.
- Research with serpapi_search only when outside context or facts are needed.
- Use get_current_datetime_tool for any date or time references in messages.
- Read and summarize email threads using summarize_thread_tool before drafting a reply if thread context exists.
- Keep all communication concise, professional, and devoid of unnecessary filler.
- Maintain an ultra-polite, highly loyal, and impeccably sharp tone."""
