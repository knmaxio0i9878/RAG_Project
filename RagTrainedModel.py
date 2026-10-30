# 1 Imports

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY is missing. Add it to your .env file.")


# 2 API_Configuration

MODEL = "nvidia/nemotron-3.5-lightning:free"
FILE_PATH = str(Path(__file__).resolve().parent / "TechNova_Employee_Handbook_Detailed.pdf")


# 3 Load PDF

loader = PyPDFLoader(FILE_PATH)
documents = loader.load()
print("Total Pages:",len(documents))

# 4. Text Chunking

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(documents)
print("Total Chunks:",len(chunks))

# 5. Create Embedding Model

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Embedding Model Ready !")

# 6. Create Vector Database

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./technova_rag_db"
)

print("Vector Database Created !")

# 7. Create Retriever

retriever = vector_db.as_retriever(
    search_kwargs={"k":3}
)
print("Retriever Body Ready !")

# 8. Create LLM
llm = ChatOpenAI(
    model = MODEL,
    openai_api_key=api_key,
    openai_api_base="https://openrouter.ai/api/v1"
)
print("LLM Ready")

# 9 Create RAG Prompt

prompt = ChatPromptTemplate.from_template("""
  You are an AI assistant for Technova employee policies.
  Answer the user's question using only provided context

  If the answer is not available in the context,
  say:

  "I could not find this information in the provided document."
  Do not make up information.

  Context:
  {context}

  Give a clear concise answer.

""")

# Give  a Clear and concise answer
# Answer in maximum 50 words
# Do not make up information

# 10 Create Rag Chain

rag_chain = prompt | llm

# 11 Ask user question
question = input("\n Ask a question about Technova Policy:")

# 12 Retriever Relevent Chunks
results = retriever.invoke(question)

print("------------------------")
print("Retriever Context")
print("------------------------")

context = " "

for i, document in enumerate(results):
  print(f"\n --- Retrieved Chunk {i+1} ---")
  print("Source:",document.metadata)
  print(document.page_content)
  context += "\n" + document.page_content

# 13 Send Context + Question to LLM

response = rag_chain.invoke({
    "context":context,
    "question":question
})

# 14 Final Answer
print("------------------------")
print("Final AI Answer")
print("------------------------")

print(response.content)
