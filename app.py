from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyMuPDFLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")


# embedding model - fetch once, cache for future use
@st.cache_resource
def get_embedding_model():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=gemini_api_key
    )


def format_history(history, limit=3):
    """Turn the last `limit` (question, answer) pairs into a plain text block for the prompt."""
    recent = history[-limit:]
    return "\n".join(f"Q: {q}\nA: {a}" for q, a in recent)


st.set_page_config(page_title="StudyBuddy", page_icon="📚")

st.title("📚 StudyBuddy")
st.markdown("I'm Ur AI study Assistant")


#handling functions for diiferent files
def load_pdf(file_name):
    loader=PyMuPDFLoader(file_name)
    return loader.load()

def load_text(file_name):
    loader=TextLoader(file_name)
    return loader.load()


# ---- sidebar: file upload, separate from the chat flow ----
with st.sidebar:
    st.header("Upload material")
    uploaded_file = st.file_uploader("Upload file", type=["pdf","txt"])

    if uploaded_file:
        if "db" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
            with st.spinner("Reading and indexing PDF..."):
                try:
                    suffix=os.path.splitext(uploaded_file.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = tmp.name

                    def load_file(file):
                        ext=os.path.splitext(file)[1].lower()
                        if ext==".pdf":
                            return load_pdf(file)
                        if ext==".txt":
                            return load_text(file)
                        raise ValueError(f"Unsupported file:{ext}")    
                    texts = load_file(tmp_path)

                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    chunks = splitter.split_documents(texts)

                    if not chunks:
                        st.error("Couldn't extract any text from this PDF. It may be scanned or image-based.")
                        st.stop()

                    embedding = get_embedding_model()
                    st.session_state.db = Chroma.from_documents(chunks, embedding)
                    st.session_state.file_name = uploaded_file.name

                except Exception as e:
                    st.error(f"Failed to process the PDF: {e}")
                    st.stop()

        st.success(f"Indexed: {uploaded_file.name}")


# ---- prompt template includes conversation history ----
prompt = ChatPromptTemplate([
    (
        "system",
        "Act as a student assistant. If document context is provided, prioritize it "
        "and avoid ungrounded claims about the document. If no document context is "
        "provided, answer using your general knowledge while maintaining the assistant persona."
    ),
    ("user", "Previous conversation:\n{history}\n\nContext:\n{context}\n\nQuery:\n{query}")
])


llm = ChatGroq(model="openai/gpt-oss-120b", groq_api_key=groq_api_key)
output_parser = StrOutputParser()
chain = prompt | llm | output_parser


# ---- chat state ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (role, message) tuples, role = "user"/"assistant"
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []  # list of (question, answer) tuples, used for prompt memory


# ---- redraw all previous messages (Streamlit reruns the whole script each time) ----
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)


# ---- new input at the bottom, chat-style ----
query = st.chat_input("Ask something...")

if query:
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.chat_history.append(("user", query))

    history_text = format_history(st.session_state.qa_history)

    if uploaded_file and "db" in st.session_state:
        try:
            context_docs = st.session_state.db.similarity_search(query)
            context = "\n\n".join(doc.page_content for doc in context_docs)

            res = chain.invoke({"context": context, "query": query, "history": history_text})

            with st.chat_message("assistant"):
                st.markdown(res)
                with st.expander("📄 Sources used for this answer"):
                    for i, doc in enumerate(context_docs, start=1):
                        page_num = doc.metadata.get("page", "unknown")
                        if isinstance(page_num, int):
                            page_num += 1  # PyMuPDF pages are 0-indexed
                        st.markdown(f"**Source {i} — Page {page_num}**")
                        st.caption(doc.page_content[:300] + "...")

            st.session_state.chat_history.append(("assistant", res))
            st.session_state.qa_history.append((query, res))

        except Exception as e:
            error_msg = f"Something went wrong getting a response: {e}"
            with st.chat_message("assistant"):
                st.error(error_msg)
            st.session_state.chat_history.append(("assistant", error_msg))

    else:
        # no PDF uploaded - plain LLM fallback
        try:
            res = chain.invoke({"context": "", "query": query, "history": history_text})
            with st.chat_message("assistant"):
                st.markdown(res)
            st.session_state.chat_history.append(("assistant", res))
            st.session_state.qa_history.append((query, res))
        except Exception as e:
            error_msg = f"Something went wrong getting a response: {e}"
            with st.chat_message("assistant"):
                st.error(error_msg)
            st.session_state.chat_history.append(("assistant", error_msg))