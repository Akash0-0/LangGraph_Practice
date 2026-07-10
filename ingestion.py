from langchain_openai import ChatOpenAI
from langgraph.graph import START,END,StateGraph
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFMinerLoader

presistent_directory = "db/chroma_db"
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  

def doc_loading(path:str):
    loader=PDFMinerLoader(
        file_path=path
    )
    documents=loader.load()

    return documents

def split_documents(docs, chunk_size=1000, chunk_overlap=0):
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into chunks...")

    # RecursiveCharacterTextSplitter cascades through separators (\n\n -> \n -> " ")
    # so chunks never exceed chunk_size, unlike the basic CharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    chunk = text_splitter.split_documents(docs)
    return chunk

paths=["academics_handbook.pdf","fee_structure.pdf"]

for i in paths:
    docs=doc_loading(path=i)
    chunks=split_documents(docs)


    vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=presistent_directory+i,
            collection_metadata={"hnsw:space": "cosine"}
        )
    print("--- Finished creating vector store ---")