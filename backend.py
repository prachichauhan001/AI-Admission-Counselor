"""
AI Admission Counselor for MIET — Backend
-------------------------------------------
Agentic RAG chatbot backend built with LangGraph + Groq LLM.

Exposes what app.py (the Streamlit UI) needs:
  - COLLEGE_INFO     : single source of truth for name / address / tagline
  - PROGRAMME_GROUPS : dict of programme categories -> list of programme names
                        (used to render the selection screen buttons)
  - build_app()       : builds (once, then reuses) the compiled LangGraph app
"""

import os
from typing import TypedDict, Annotated

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

_GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not _GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY not found. Create a '.env' file in the same folder as "
        "backend.py with a line like: GROQ_API_KEY=gsk_your_key_here "
        "(get a free key at https://console.groq.com/keys)."
    )

# ---------------------------------------------------------------------------
# Single source of truth for the college's identity. app.py imports this too,
# so the name/address shown on screen and the name/address the LLM uses in
# its answers can never drift apart or get hallucinated.
# ---------------------------------------------------------------------------

COLLEGE_INFO = {
    "name": "Meerut Institute of Engineering & Technology",
    "short_name": "MIET",
    "tagline": "Innovate. Learn. Lead the Future.",
    "established": "1997",
    "address": "N.H. 58, Delhi-Roorkee Highway, Baghpat Bypass Road, Meerut, Uttar Pradesh, India",
    "affiliation": "Dr. A.P.J. Abdul Kalam Technical University (AKTU), Lucknow",
    "logo_url": "https://www.miet.ac.in/images/newimages/logo.png",
}

_IDENTITY_BLOCK = (
    f"You are the official AI Admission Counselor for {COLLEGE_INFO['name']} "
    f"({COLLEGE_INFO['short_name']}), established {COLLEGE_INFO['established']}, "
    f"located at {COLLEGE_INFO['address']}, affiliated to {COLLEGE_INFO['affiliation']}. "
    f"Always use this exact college name, campus and location whenever asked — "
    f"never invent, guess, or use a different college name or city.\n\n"
)

# ---------------------------------------------------------------------------
# Programme groups (used by the Streamlit selection screen)
# ---------------------------------------------------------------------------

PROGRAMME_GROUPS = {
    "B.Tech (Engineering Branches, 4 Years)": [
        "B.Tech Biotechnology",
        "B.Tech Civil Engineering",
        "B.Tech Computer Science",
        "B.Tech CSE",
        "B.Tech CSE - AI",
        "B.Tech CSE - Data Science",
        "B.Tech CSE - AI & ML",
        "B.Tech Electrical Engineering",
        "B.Tech ECE",
        "B.Tech Mechanical Engineering",
        "B.Tech ECE (VLSI Design & Technology)",
    ],
    "Management & Computer Applications (2 Years)": [
        "MBA",
        "MCA",
    ],
    "M.Tech (2 Years)": [
        "M.Tech - ECE",
        "M.Tech - Biotechnology",
        "M.Tech - CSE",
        "M.Tech - CSE (AI & ML)",
    ],
    "Pharmacy Courses": [
        "B.Pharm",
        "M.Pharm - Pharmaceutics",
        "M.Pharm - Pharmacology",
    ],
    "New Courses for Working Professionals": [
        "B.Tech Mechanical Engineering (Working Professionals)",
        "B.Tech Electrical Engineering (Working Professionals)",
        "M.Tech CSE (Working Professionals)",
    ],
    "Basic Sciences (Affiliated to CCS University, Meerut)": [
        "B.Sc. Biotechnology",
        "B.Sc. Microbiology",
        "B.Sc. (Hons.) Biotechnology",
        "B.Sc. (Hons.) Microbiology",
        "M.Sc. Biotechnology",
        "M.Sc. Microbiology",
    ],
}

# ---------------------------------------------------------------------------
# Lazy singletons — embeddings / retrievers / llm / compiled graph.
# Built on first use and cached in-process, so Streamlit's own
# @st.cache_resource wrapper around build_app() only ever triggers this once.
# ---------------------------------------------------------------------------

_embeddings = None
_retrievers = None
_llm = None
_compiled_app = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _embeddings


def _build_retriever(pdf_path: str, index_name: str):
    """Builds a FAISS retriever for a PDF, caching the index on disk so
    repeated runs load instantly instead of re-embedding every time.
    Returns None (instead of crashing) if the PDF is missing or has no
    extractable text, so one bad PDF doesn't take down the whole app."""

    embeddings = _get_embeddings()
    index_path = f"faiss_index_{index_name}"

    if os.path.exists(index_path):
        # Index already built earlier -> just load it (fast)
        vectorstore = FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )
        return vectorstore.as_retriever(search_kwargs={"k": 4})

    # First run for this PDF -> build the index and save it
    if not os.path.exists(pdf_path):
        print(f"[WARNING] '{pdf_path}' not found — '{index_name}' topic will "
              f"fall back to general knowledge answers.")
        return None

    loader = PyPDFLoader(pdf_path)
    document = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(document)

    if not chunks:
        print(f"[WARNING] '{pdf_path}' produced no extractable text (likely "
              f"scanned images with no OCR) — '{index_name}' topic will fall "
              f"back to general knowledge answers until this PDF is fixed.")
        return None

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)

    return vectorstore.as_retriever(search_kwargs={"k": 4})


def _build_retrievers():
    """Builds all 8 PDF retrievers once and caches them in-process."""
    global _retrievers
    if _retrievers is None:
        _retrievers = {
            "scholarship": _build_retriever("scholership_pdf.pdf", "scholarship"),
            "admission": _build_retriever("Admission_documents.pdf", "admission"),
            "transport": _build_retriever("transport.pdf", "transport"),
            "fee": _build_retriever("fee_structure.pdf", "fee"),
            "hostel": _build_retriever("hostel_booklet.pdf", "hostel"),
            "rules": _build_retriever("rule_book.pdf", "rules"),
            "courses": _build_retriever("academic_boucher.pdf", "courses"),
            "syllabus": _build_retriever("B.Tech_1styear.pdf", "syllabus"),
        }
    return _retrievers


def _get_llm():
    global _llm
    if _llm is None:
        # llama-3.3-70b-versatile has been retired by Groq (deprecated June 2026).
        # openai/gpt-oss-120b is Groq's recommended replacement for equivalent quality.
        #
        # temperature dropped from 0.4 -> 0.1 and a fixed seed added: at 0.4 the
        # same question ("what's the MBA fee", "tell me about placements") could
        # come back worded/numbered differently on every ask. Low temperature +
        # a fixed seed makes answers to the same question consistent instead of
        # re-generated from scratch each time.
        _llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0.1,
            model_kwargs={"seed": 42},
            timeout=30,
            max_retries=3,
        )
    return _llm


def _invoke_llm_safely(prompt: str):
    """Calls the Groq LLM and turns low-level connection errors into a clear,
    user-facing message instead of a raw traceback."""
    llm = _get_llm()
    try:
        return llm.invoke(prompt)
    except Exception as exc:
        raise RuntimeError(
            "Could not reach Groq's servers (api.groq.com). This is usually caused "
            "by one of: no internet connection, a firewall/antivirus/VPN blocking "
            "the request, or an invalid GROQ_API_KEY. Please check your connection "
            "and your .env file, then try again.\n\n"
            f"Original error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class State(TypedDict):
    programme: str
    messages: Annotated[list, add_messages]
    query_type: str
    retrieved_context: str


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def _classifier_node(state: State) -> dict:
    """Look at the latest user message and decide which path to take."""

    last_message = state["messages"][-1].content

    prompt = (
        "Classify the following student query into exactly one category: "
        "'scholarship', 'admission', 'transport', 'fee', 'hostel', 'rules', "
        "'courses', 'syllabus', or 'general'.\n\n"
        "Use 'scholarship' for questions about scholarships, financial aid, merit waivers, "
        "or fee concessions based on merit/category.\n"
        "Use 'admission' for questions about the admission process, eligibility criteria, "
        "entrance exams, counselling, seat allotment, or documents required for admission.\n"
        "Use 'transport' for questions about bus routes, transport fee, or pickup points.\n"
        "Use 'fee' for questions about tuition fee, hostel fee, payment schedule, refund, "
        "or late charges.\n"
        "Use 'hostel' for questions about hostel rooms, mess, or hostel facilities "
        "(NOT hostel fee amount, that goes under 'fee').\n"
        "Use 'rules' for questions about college discipline, code of conduct, attendance "
        "rules, or hostel/campus rules.\n"
        "Use 'courses' for questions about which courses/branches are offered, their "
        "duration, or university affiliation.\n"
        "Use 'syllabus' for questions about first-year subjects, curriculum, or B.Tech "
        "1st year academic structure.\n"
        "Use 'general' for greetings, casual talk, placements, rankings, campus location, "
        "or anything not related to the above topics.\n\n"
        f"Query: {last_message}\n\n"
        "Return only one word: scholarship, admission, transport, fee, hostel, rules, "
        "courses, syllabus, or general."
    )

    response = _invoke_llm_safely(prompt)
    category = response.content.strip().lower()

    resolved = "general"
    for key in ["scholarship", "admission", "transport", "fee", "hostel",
                "rules", "syllabus", "courses"]:
        if key in category:
            resolved = key
            break

    return {"query_type": resolved}


def _retrieve_context(retriever, query: str) -> str:
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)


def _make_rag_node(retriever_key: str):
    """Factory that builds a RAG node bound to one of the 8 retrievers.
    If that PDF failed to load (missing/no extractable text), falls back
    to answering from general knowledge instead of crashing."""

    def _node(state: State) -> dict:
        retrievers = _build_retrievers()
        retriever = retrievers.get(retriever_key)
        if retriever is None:
            return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}

        query = state["messages"][-1].content
        programme = state.get("programme", "")

        # Bias the similarity search towards the student's own programme so
        # a "fee" or "scholarship" PDF that lists many programmes in one
        # table doesn't pull in chunks about *other* programmes (e.g. asking
        # about MBA fee no longer drags in the B.Pharm fee table).
        search_query = f"{programme} {query}".strip() if programme else query

        return {"retrieved_context": _retrieve_context(retriever, search_query)}

    return _node


def _general_node(state: State) -> dict:
    """Answers directly using the LLM's own knowledge, no retrieval needed."""
    return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}


def _response_node(state: State) -> dict:
    """Generates the final answer, personalized using the student's programme."""

    query = state["messages"][-1].content
    programme = state.get("programme", "Unknown")
    context = state["retrieved_context"]

    if context == "NO_RETRIEVAL_NEEDED":
        prompt = (
            _IDENTITY_BLOCK
            + f"You are talking to a {programme} student. Answer the question below "
              f"in 3-5 short sentences, friendly and precise.\n\n"
              f"If the question asks for a specific number or statistic you don't have "
              f"confirmed data for (e.g. exact placement percentage, package figures, "
              f"rankings, exact dates), do NOT invent a number — say this isn't "
              f"confirmed and suggest checking the official MIET website or contacting "
              f"the admissions/placement office for the latest figures.\n\n"
              f"Question: {query}"
        )
    else:
        prompt = (
            _IDENTITY_BLOCK
            + f"You are helping a {programme} student. Use ONLY the context below, "
              f"taken from official college documents.\n\n"
              f"IMPORTANT: this context may contain information about OTHER programmes "
              f"too. Ignore anything not relevant to {programme} and answer ONLY for "
              f"{programme}, unless the student explicitly asks you to compare "
              f"programmes. If the exact figure/detail for {programme} isn't present "
              f"in the context, say so honestly instead of guessing or borrowing "
              f"another programme's figure.\n\n"
              f"Context:\n{context}\n\n"
              f"Question: {query}\n\n"
              f"Answer in 3-6 short, clear sentences. Be concise and accurate, no filler."
        )

    response = _invoke_llm_safely(prompt)
    return {"messages": [("ai", response.content.strip())]}


def _route_query(state: State):
    mapping = {
        "scholarship": "scholarship_rag",
        "admission": "admission_rag",
        "transport": "transport_rag",
        "fee": "fee_rag",
        "hostel": "hostel_rag",
        "rules": "rules_rag",
        "courses": "courses_rag",
        "syllabus": "syllabus_rag",
    }
    return mapping.get(state["query_type"], "general")


# ---------------------------------------------------------------------------
# Graph builder — this is what app.py calls (wrapped in st.cache_resource)
# ---------------------------------------------------------------------------

def build_app():
    """Builds (once) and returns the compiled LangGraph app."""

    global _compiled_app
    if _compiled_app is not None:
        return _compiled_app

    graph = StateGraph(State)

    graph.add_node("classifier", _classifier_node)
    graph.add_node("scholarship_rag", _make_rag_node("scholarship"))
    graph.add_node("admission_rag", _make_rag_node("admission"))
    graph.add_node("transport_rag", _make_rag_node("transport"))
    graph.add_node("fee_rag", _make_rag_node("fee"))
    graph.add_node("hostel_rag", _make_rag_node("hostel"))
    graph.add_node("rules_rag", _make_rag_node("rules"))
    graph.add_node("courses_rag", _make_rag_node("courses"))
    graph.add_node("syllabus_rag", _make_rag_node("syllabus"))
    graph.add_node("general", _general_node)
    graph.add_node("response", _response_node)

    graph.add_edge(START, "classifier")
    graph.add_conditional_edges("classifier", _route_query)

    graph.add_edge("scholarship_rag", "response")
    graph.add_edge("admission_rag", "response")
    graph.add_edge("transport_rag", "response")
    graph.add_edge("fee_rag", "response")
    graph.add_edge("hostel_rag", "response")
    graph.add_edge("rules_rag", "response")
    graph.add_edge("courses_rag", "response")
    graph.add_edge("syllabus_rag", "response")
    graph.add_edge("general", "response")

    graph.add_edge("response", END)

    _compiled_app = graph.compile()
    return _compiled_app


# ---------------------------------------------------------------------------
# Optional: run `python backend.py` for a quick terminal test.
# This block is guarded so importing backend.py from Streamlit (app.py)
# never triggers the input() loop or blocks app startup.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli_app = build_app()

    print("=========================================")
    print(f"   AI Admission Counselor for {COLLEGE_INFO['short_name']}")
    print("=========================================\n")
    print("Which programme are you in?\n")

    flat_programmes = []
    for group, options in PROGRAMME_GROUPS.items():
        print(f"--- {group} ---")
        for opt in options:
            flat_programmes.append(opt)
            print(f"{len(flat_programmes)}. {opt}")
        print()

    choice = input("Enter your choice number: ")
    try:
        student_programme = flat_programmes[int(choice) - 1]
    except (ValueError, IndexError):
        student_programme = "B.Tech CSE"

    print(f"\nGreat! You're set as a {student_programme} student.")

    while True:
        user_query = input("You:  ")
        if user_query.lower() in ["exit", "quit"]:
            break

        result = cli_app.invoke({
            "programme": student_programme,
            "messages": [("human", user_query)],
        })

        print(f"Assistant : {result['messages'][-1].content}")