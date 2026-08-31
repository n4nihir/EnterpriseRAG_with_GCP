from langchain_openai import ChatOpenAI
from app.agents.state import AgentState
from app.config import settings
import logfire

# Initialize the OpenAI reasoning model (o3-mini)
llm = ChatOpenAI(
    api_key=settings.OPENAI_API_KEY,
    model=settings.OPENAI_MODEL
)

def planner_node(state: AgentState):
    """
    The Planner determines if a search is needed based on the ENTIRE conversation.
    """
    # Get the conversation history (excluding the latest message)
    history = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"
    
    user_message = state["messages"][-1]["content"] if state["messages"] else ""
    
    prompt = f"""
    You are an intelligent Assistant Planner & Query Optimizer for an Enterprise Knowledge Base.
    Analyze the conversation history and the latest user message.
    
    CONVERSATION HISTORY:
    {history}
    
    LATEST MESSAGE:
    "{user_message}"
    
    Task:
    1. If the message is a casual greeting (hi, hello, thanks, goodbye), pleasantry, or asks only about conversation context (e.g., "what did I say earlier?", "what is my name?"), respond with ONLY: CONVERSATIONAL
    2. For ALL technical, conceptual, procedural, factual, or enterprise documentation questions:
       - Rewrite the question into a standalone, concise, keyword-rich search query optimized for vector database retrieval.
       - Resolve any ambiguous pronouns ("it", "this", "that", "the previous method") using the conversation history.
    
    Output ONLY 'CONVERSATIONAL' or the standalone search query without any explanation, markdown, or quotes.
    """
    
    with logfire.span("🧠 Planner Decision"):
        decision = llm.invoke(prompt).content.strip()
        logfire.info(f"Intent identified: {decision}")
    
    if decision == "CONVERSATIONAL":
        return {
            "current_query": "CONVERSATIONAL",
            "status": "Handling conversationally (using memory)...",
            "plan": ["Intent: Conversational/Memory", "Retrieval: Skipped"]
        }
    
    return {
        "current_query": decision,
        "status": f"Technical research needed. Searching for: {decision}",
        "plan": ["Intent: Technical", f"Search Term: {decision}"]
    }
