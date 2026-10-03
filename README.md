# 📚 StudyBuddy

An AI-powered study assistant that lets you chat with your own PDFs — textbooks, notes, or question papers — and get answers grounded in the actual document, with page-level source citations so you can verify every answer.

**Live demo:** https://studybuddychat.streamlit.app/

---

## What it does

- Upload a PDF (textbook chapter, notes, PYQs) from the sidebar.
- Ask questions about it in a chat interface.
- Answers are generated using Retrieval-Augmented Generation (RAG) — the app retrieves the most relevant sections of your PDF and passes them to the LLM as context, instead of relying on the model's general knowledge.
- Every answer shows which page(s) it was sourced from, so you can go verify it directly in your material.
- Supports natural follow-up questions (e.g. "explain that more simply") using short-term conversation memory.

## Why I built it

Most "chat with PDF" demos are shallow wrappers around an LLM call. I wanted something I'd actually use before exams — and to build a project that forced me to deal with real engineering problems: deployment constraints, caching expensive operations, error handling, and making answers verifiable instead of just trusting the model.

## Tech stack

| Layer | Tool |
|---|---|
| UI | Streamlit (chat interface) |
| LLM | Groq (`openai/gpt-oss-120b`) |
| Embeddings | Google Gemini (`gemini-embedding-001`) |
| Vector store | Chroma |
| Orchestration | LangChain (LCEL) |
| PDF parsing | PyMuPDF |

## Architecture

```
PDF Upload (sidebar)
   → PyMuPDFLoader (extract text + page metadata)
   → RecursiveCharacterTextSplitter (chunk_size=500, overlap=50)
   → Gemini Embeddings
   → Chroma vector store (cached in session state)

User Question (chat input)
   → similarity_search() on Chroma
   → top-k relevant chunks + recent Q&A history → prompt
   → Groq LLM
   → Answer + source citations (page numbers) displayed in chat
```

## Key engineering decisions

- **Session-scoped caching** — the vector index is built once per uploaded file and cached in `st.session_state`, so follow-up questions don't re-embed the document on every query (each embedding call is a network request, so this matters for both speed and cost).
- **Deployability over local-only tooling** — originally used Ollama for embeddings during development, but switched to a hosted embedding model (Gemini) since Ollama requires a locally running server and can't run on free hosting platforms like Streamlit Cloud.
- **Source citation** — chunk-level metadata (page number) is preserved from ingestion through to the UI, so answers are traceable back to the original document rather than being opaque LLM output.
- **Scoped conversation memory** — only the last 3 Q&A turns are included in the prompt, rather than full history, to keep the model focused on the current thread and avoid excessive token usage.
- **Error handling** — empty queries, unparseable/scanned PDFs, and LLM API failures are all handled explicitly with user-facing messages instead of raw stack traces.
- **Measuring retrieval instead of assuming it works** — a small evaluation script checks whether the right page is retrieved for known questions (see [Evaluation](#evaluation)).

## Evaluation

`eval.py` tests the retrieval step only. For each question in a test set, it runs the top-k vector search and checks whether any retrieved chunk comes from the page that contains the answer.

**Test set:** 21 questions written from a 26-page Operating Systems notes PDF (two complete units), mixing direct lookups, reworded questions, and numeric questions from worked examples.

**Setup:** `chunk_size=1000`, `chunk_overlap=100`, Gemini embeddings, Chroma.

| Top-k | Correct page retrieved |
|---|---|
| k=1 | 20/21 (95.2%) |
| k=3 | 21/21 (100%) |

**The one miss at k=1:** a worked scheduling example that is split across a page boundary. The problem statement is on page 20 and the solution is on page 21. Retrieval returned the problem statement first, which is a reasonable match for the question, and the solution page appeared within the top 3.

**Caveats:** the test set is small, uses a single document, and scores retrieval at page level only. It does not evaluate the quality of the generated answers.

To run it, place the PDF and question file next to `eval.py` and run `python eval.py`. Delete the `chroma_db` folder whenever you change the PDF or the chunking settings, otherwise the script reuses the old saved index.

## Known limitations

- No handling for scanned/image-only PDFs (no OCR).
- Single-document search only — can't currently query across multiple uploaded files at once.
- Retrieval is pure vector similarity — no reranking or hybrid (keyword + vector) search yet.
- Fixed-size chunking can separate related content, such as a worked example and its solution on the next page.
- The free Gemini embedding tier allows about 100 requests per minute, so very large PDFs can hit rate-limit errors during indexing. Batching with pauses or retries would be needed for big documents.
- The evaluation is small and retrieval-only; generated answers are not scored.

## Running locally

```bash
git clone https://github.com/yashwanthc16/StudyBuddy.git
cd StudyBuddy
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
```

Then run:
```bash
streamlit run app.py
```

## Future improvements

- Multi-document search
- Hybrid (keyword + vector) retrieval and reranking
- OCR support for scanned PDFs
- Batched embedding with retry/backoff for large PDFs
- Larger, multi-document evaluation set, plus scoring of answer quality