from langchain.agents import create_agent
from pydantic import SecretStr
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
import os
load_dotenv()
groq_api_key=os.getenv("GROQ_API_KEY")
serp_api_key=os.getenv("SERP_API_KEY")
from langchain_groq import ChatGroq
from serpapi import GoogleSearch

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.1,            #temperature higher means more creative and lower means strict in output 
    max_retries=2,
    api_key=SecretStr(groq_api_key) if groq_api_key is not None else None
)
def serpapi_search(query: str):             #Query is expected to be a str thats type hints useful for verification in return types.
    """Searches for a query using the SerpAPI on Google."""
    params = {
        "q": query,
        "hl": "en",
        "gl": "us",
        "api_key": SecretStr(serp_api_key) if serp_api_key is not None else None            #pydantic verification due to Type Hints 
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    
    # Extract top results (titles + links)
    if "organic_results" in results:
        return [
            {"title": r["title"], "link": r["link"], "snippet": r.get("snippet", "")}
            for r in results["organic_results"][:5]       #:5 gets only 0:5 means 5 items from results dictionary 
        ]
    return {"error": "No results found"}
memory= InMemorySaver()
agent= create_agent(
    model=llm,
    tools=[serpapi_search],
    system_prompt="You are a Helpful Coding Agent Only Answer Questions Related to Programming and Computers. Basically IT related.",
    checkpointer=memory
)
question= input("Enter Your Question for Google Search?")
# Run the agent
response=agent.invoke(
    {"messages": [{"role": "user", "content": question}]},
    config={"configurable": {"thread_id": "user123"}}
)

# Last Message
print(response['messages'][-1].content)

# Run the agent
response=agent.invoke(
    {"messages": [{"role": "user", "content": "What were we Talking About Earlier ?"}]},
    config={"configurable": {"thread_id": "user123"}}
)

# Last Message
print(response['messages'][-1].content)

question= input("Enter Your Question for Google Search?")
# Run the agent
response=agent.invoke(
    {"messages": [{"role": "user", "content": question}]},
    config={"configurable": {"thread_id": "user123"}}
)

# Last Message
print(response['messages'][-1].content)


# Run the agent
response=agent.invoke(
    {"messages": [{"role": "user", "content": "What were we Talking About Earlier ?"}]},
    config={"configurable": {"thread_id": "user123"}}
)

# Last Message
print(response['messages'][-1].content)

