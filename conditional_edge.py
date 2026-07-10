from langchain_openai import ChatOpenAI
from langgraph.graph import START,END,StateGraph
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFMinerLoader
load_dotenv()
presistent_directory = "db/chroma_db"
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  

from typing import TypedDict
class OlalalState(TypedDict):
    query:str
    category:str
    chatbot:str
    answer:str

llm = ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)


def Router(state:OlalalState)->dict:
    text=state["query"]
    messages=f""" YOu are a professional categorises from 
    the given text you have to vategorise if the query is about academy or fees
    only type one word academic or fees nothing else here is the query : {text}
    """
    result=llm.invoke(messages)
    cat=result.content
    if cat=="academic" or cat=="Academic":
        return "academic"
    elif cat=="fees" or "Fees":
        return "fees"

fees_dir="db\chroma_dbfee_structure.pdf"

def fees(state:OlalalState):
    query=state["query"]
    db = Chroma(
    persist_directory=fees_dir,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"})

    retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3,     "fetch_k": 10,"lambda_mult": 0.5})   
    docs = retriever.invoke(query)
    messages=f""" Ypu are a chatbot that helps students with query
             With given contenxt here is the context context: {docs} 
             and here is the query query: {query}"""
    result=llm.invoke(messages)

    return{
        "answer":result.content
    }

academic_dir="db\chroma_dbacademics_handbook.pdf"

def academic(state:OlalalState):
    query=state["query"]
    db = Chroma(
    persist_directory=academic_dir,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"})

    retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3,     "fetch_k": 10,"lambda_mult": 0.5})   
    docs = retriever.invoke(query)
    messages=f""" Ypu are a chatbot that helps students with query
             With given contenxt here is the context context: {docs} 
             and here is the query query: {query}"""
    result=llm.invoke(messages)

    return{
        "answer":result.content
    }

builder=StateGraph(OlalalState)


builder.add_node("fees",fees)
builder.add_node("academic",academic)

builder.add_conditional_edges(START,Router)
builder.add_edge("fees",END)
builder.add_edge("academic",END)

agent=builder.compile()

run=agent.invoke({"query":input("GAYBOYY WHAT YOU WANNA KNOWW")})
print(run)
