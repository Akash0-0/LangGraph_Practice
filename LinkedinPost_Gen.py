from langgraph.graph import START,END,StateGraph
from langchain_openai import ChatOpenAI
from typing import TypedDict ,Annotated,Literal,Optional
from dotenv import load_dotenv
import os
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_tavily import TavilySearch
#just for commits
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

rllm = ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.5
)

class OlalalaState(TypedDict):
    Topic:str
    draft:str
    review:str
    feedback:str
    Final_post:str
    # FIX: ToolNode reads tool_calls off a messages list, not off `draft`.
    # This key is required for the tools node to work at all.
    messages: Annotated[list, add_messages]


def PostWriter(state:OlalalaState):
    topic=state["Topic"]
    Feedback=state["feedback"]
    existing_messages = state.get("messages", [])

    # FIX: only build the initial prompt once per writing cycle.
    # If we already have messages (e.g. coming back from "tools"),
    # keep the conversation going instead of restarting it.
    if not existing_messages:
        messages=f""" You are a professional content writer your job is to write up a post on givven topic 
                    You can use the given tool which searchtools The topic is Topic: {topic} 
                    Ypu can also use the feedback given by reviewer if its available
                      if its empty you can ignore it Feedback:{Feedback}"""
        existing_messages = [{"role": "user", "content": messages}]

    results=wllm_withtools.invoke(existing_messages)

    updates = {"messages": [results]}

    # FIX: only treat this as a finished draft once the model stops
    # asking for tool calls. Otherwise "draft" gets set prematurely
    # to an empty/partial response while a tool call is pending.
    if not getattr(results, "tool_calls", None):
        print(f"""genreated post:{results.content}""")
        updates["draft"] = results.content

    return updates

from pydantic import BaseModel, Field

class ReviewResult(BaseModel):
    # FIX: constrain to Literal so with_structured_output can't return
    # something other than exactly "Yes"/"No"
    Review: Literal["Yes", "No"] = Field(description="Must be exactly 'Yes' or 'No'")
    FeedBack: str = Field(description="Your detailed feedback on the post")



def Reviewer(state:OlalalaState):
    topic=state["Topic"]
    draft=state["draft"]

    structured_llm = rllm.with_structured_output(ReviewResult)
    
    messagess = f"Review this LinkedIn post. Topic: {topic}. Draft: {draft}."
    
    result = structured_llm.invoke(messagess)
    
    return {
        "review": result.Review,
        "feedback": result.FeedBack,
        # FIX: clear messages so the next PostWriter pass starts a
        # fresh tool-calling conversation instead of appending forever
        "messages": []
    }

def assignFinalPost(state:OlalalaState):
    final_state=state["draft"]
    return{
        "Final_post":final_state
    }

# FIX: route straight after PostWriter based on whether the last
# message has tool_calls, so we go to "tools" only when actually needed
# instead of unconditionally fanning out to both tools and reviewer.
def route_after_writer(state: OlalalaState) -> Literal["tools", "reviewer"]:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "reviewer"

def router(state:OlalalaState)->Literal["post_writer","assign_final_post"]:
    review=state["review"]
    if review=="No":
        return "post_writer"
    else:  # FIX: always return something valid instead of falling through to None
        return "assign_final_post"
    
builder=StateGraph(OlalalaState)
builder.add_node("post_writer",PostWriter)
builder.add_node("assign_final_post",assignFinalPost)
builder.add_node("reviewer",Reviewer)
builder.add_node("tools",ToolNode(tools=tools))

builder.add_edge(START,"post_writer")

# FIX: this used to be two unconditional edges firing every time
# (post_writer -> tools AND post_writer -> reviewer in parallel).
# Now it's one conditional edge that picks exactly one path.
builder.add_conditional_edges(
    "post_writer",
    route_after_writer,
    {"tools": "tools", "reviewer": "reviewer"}
)
# FIX: tools had no outgoing edge before, so a tool result never made
# it back to the writer. This closes the ReAct loop.
builder.add_edge("tools","post_writer")

builder.add_conditional_edges(
    "reviewer",
    router,
    # FIX: explicit mapping so keys match the actual node names
    # ("post_writer"/"assign_final_post"), not "PostWriter"/"assignFinalPost"
    {"post_writer": "post_writer", "assign_final_post": "assign_final_post"}
)
builder.add_edge("assign_final_post",END)

agent=builder.compile()

if __name__ == "__main__":
    agent.invoke({
        "Topic":"The impact of AI on the future of work",
        "draft":"",
        "review":"",
        "feedback":"",
        "Final_post":"",
        "messages":[]
    })
