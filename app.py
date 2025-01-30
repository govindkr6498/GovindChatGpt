from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
import os

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change for better security)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Set OpenAI API Key
# os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"  # Replace with your actual API key

os.environ["OPENAI_API_KEY"] = "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"
print(os.getenv("OPENAI_API_KEY"))

# Load PDF and create vector store
def get_vectorstore_from_static_pdf(pdf_path="C:/Users/lenovo/Downloads/ApexDeveloperGuidea.pdf"):
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""  # Handle None values

    # Split text into chunks
    from langchain.text_splitter import CharacterTextSplitter
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)

    # Create a vectorstore from the chunks
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(chunks, embeddings)

    return vector_store

# Load the vector store at startup
vector_store = get_vectorstore_from_static_pdf()

# Chat request model
class ChatRequest(BaseModel):
    message: str

# Function to get response from OpenAI
def get_response(user_input):
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()
    
    response = llm.invoke(user_input)  # Directly using LLM for response
    return response.content

# API Endpoint
@app.post("/api/chat")
async def chat_endpoint(chat_request: ChatRequest):
    response = get_response(chat_request.message)
    return {"answer": response}

# Run the server using: python -m uvicorn app:app --reload
