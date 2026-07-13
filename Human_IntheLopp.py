from langgraph.graph import START,END,StateGraph
from langchain_openai import ChatOpenAI
from typing import TypedDict ,Annotated,Literal,Optional
from dotenv import load_dotenv
import os
from langgraph.graph.message import add_messages, REMOVE_ALL_MESSAGES
from langchain_core.messages import RemoveMessage, HumanMessage
from langgraph.prebuilt import ToolNode
from langchain_tavily import TavilySearch
# FIX: `interrupt` (lowercase) is the function you call inside a node to
# pause execution. `Interrupt` (capital) is just the internal marker
# object LangGraph raises/returns for it -- not something you instantiate
# yourself and call .strip() on.
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

tavily_api_key = os.getenv("TAVILY_API_KEY") or os.getenv("TVILY_API_KEY")
if not tavily_api_key:
    raise RuntimeError("Missing Tavily API key. Set TAVILY_API_KEY or TVILY_API_KEY in your environment or .env file.")

search_tool = TavilySearch(max_results=5, tavily_api_key=tavily_api_key)

tools = [search_tool]

wllm = ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.7,
)
wllm_withtools=wllm.bind_tools(tools=tools)


class OlalalaState(TypedDict):
    Topic:str
    draft:str
    review:str
    feedback:str
    Final_post:str
    messages: Annotated[list, add_messages]


def build_post_writer_prompt(topic: str, feedback: str, previous_draft: str) -> str:
    if feedback and feedback.strip():
        return f"""You are a professional content writer. Revise the existing draft so it directly addresses the reviewer feedback.
Topic: {topic}
Reviewer feedback: {feedback}
Previous draft: {previous_draft or '(no draft yet)'}
Please rewrite the draft to reflect the feedback, improve clarity, and keep the topic intact."""

    return f"""You are a professional content writer. Write a fresh post on the given topic.
Topic: {topic}
Please write a polished, engaging post that covers the topic clearly and directly."""


def PostWriter(state:OlalalaState):
    topic = state["Topic"]
    feedback = state.get("feedback", "")
    previous_draft = state.get("draft", "")
    existing_messages = state.get("messages", [])

    if feedback.strip() or not existing_messages:
        prompt_messages = [HumanMessage(content=build_post_writer_prompt(topic, feedback, previous_draft))]
    else:
        prompt_messages = existing_messages

    results = wllm_withtools.invoke(prompt_messages)

    updates = {"messages": [results]}

    if not getattr(results, "tool_calls", None):
        print(f"""genreated post:{results.content}""")
        updates["draft"] = results.content

    return updates

from pydantic import BaseModel, Field

class ReviewResult(BaseModel):
    Review: Literal["Yes", "No"] = Field(description="Must be exactly 'Yes' or 'No'")
    FeedBack: str = Field(description="Your detailed feedback on the post")


def Reviewer(state:OlalalaState):
    # FIX: call interrupt() (not Interrupt(...)) -- this is what actually
    # pauses the graph. The dict you pass in is shown to whoever is
    # resuming it (e.g. surfaced in agent.get_state(...).tasks[0].interrupts).
    # Whatever value the graph is resumed with (Command(resume=...)) is
    # returned here.
    human_response = interrupt({
        "draft": state["draft"],
        "Topic": state["Topic"],
        "prompt": "Type 'approved' to accept, or write feedback for revision.",
    })

    responce = human_response.strip()

    # FIX: always return a dict on both branches. Previously the
    # non-approved path fell through and returned None, silently
    # dropping the human's feedback text.
    if responce.lower() in ["approved", "yes", "accept", "ok"]:
        return {
            "review": "Approved",
            "feedback": ""
        }
    else:
        return {
            "review": "No",
            "feedback": responce,
            # FIX: `add_messages` APPENDS, it doesn't replace -- returning
            # "messages": [] here would be a no-op and the old draft
            # message would still be sitting in state. RemoveMessage with
            # the REMOVE_ALL_MESSAGES sentinel is the actual way to clear
            # history under this reducer. Without a real clear, PostWriter's
            # "if not existing_messages" branch never re-fires, so Feedback
            # never gets woven into a new prompt.
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]
        }

def assignFinalPost(state:OlalalaState):
    final_state=state["draft"]
    return{
        "Final_post":final_state
    }

def route_after_writer(state: OlalalaState) -> Literal["tools", "reviewer"]:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "reviewer"

def router(state:OlalalaState)->Literal["post_writer","assign_final_post"]:
    review=state["review"]
    if review=="Approved":
        return "assign_final_post"
    else:
        return "post_writer"
    
builder=StateGraph(OlalalaState)
builder.add_node("post_writer",PostWriter)
builder.add_node("assign_final_post",assignFinalPost)
builder.add_node("reviewer",Reviewer)
builder.add_node("tools",ToolNode(tools=tools))

builder.add_edge(START,"post_writer")

builder.add_conditional_edges(
    "post_writer",
    route_after_writer,
    {"tools": "tools", "reviewer": "reviewer"}
)
builder.add_edge("tools","post_writer")

builder.add_conditional_edges(
    "reviewer",
    router,
    {"post_writer": "post_writer", "assign_final_post": "assign_final_post"}
)
builder.add_edge("assign_final_post",END)

# FIX: interrupt() requires a checkpointer to persist state across the
# pause -- without this, resuming with Command(resume=...) has nothing
# to resume from.
checkpointer = MemorySaver()
agent = builder.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    # FIX: a graph with interrupt() can't be driven with a single
    # .invoke() call. The first call runs until it hits interrupt() and
    # returns with "__interrupt__" in the result instead of blocking.
    # You then resume it with Command(resume=<human input>) using the
    # SAME thread_id, and repeat until there's no interrupt left.
    config = {"configurable": {"thread_id": "demo-thread-1"}}

    result = agent.invoke({
        "Topic":"The impact of AI on the future of work",
        "draft":"",
        "review":"",
        "feedback":"",
        "Final_post":"",
        "messages":[]
    }, config=config)

    while "__interrupt__" in result:
        interrupt_payload = result["__interrupt__"][0].value
        print("\n--- Draft for review ---")
        print(interrupt_payload["draft"])
        print(interrupt_payload["prompt"])

        human_input = input("> ")
        result = agent.invoke(Command(resume=human_input), config=config)

    print("=== FINAL POST ===")
    print(result["Final_post"])