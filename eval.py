import os
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

with open("test_question.json") as f:
    test_cases=json.load(f)


PDF_PATH = "OS_Units1-2.pdf"

suffix = os.path.splitext(PDF_PATH)[1]  

loader = PyMuPDFLoader(PDF_PATH)
texts = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(texts)

embedding = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=gemini_api_key
)
db = Chroma.from_documents(chunks, embedding)


correct_count = 0
total = len(test_cases)

for case in test_cases:
    question = case["question"]
    expected_page = case["expected_page"]

    results = db.similarity_search(question, k=1)

    retrieved_pages = []
    for doc in results:
        p = doc.metadata.get("page")
        if isinstance(p, int):
            retrieved_pages.append(p + 1)

    is_correct = expected_page in retrieved_pages
    if is_correct:
        correct_count += 1

    print(f"Q: {question}")
    print(f"Expected page: {expected_page} | Retrieved pages: {retrieved_pages} | {'✅' if is_correct else '❌'}")
    print()

accuracy = correct_count / total * 100
print(f"Retrieval accuracy: {correct_count}/{total} ({accuracy:.1f}%)")