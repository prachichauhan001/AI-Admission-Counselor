"""
AI Admission Counselor for MIET
--------------------------------
An agentic RAG chatbot that answers student queries about scholarships,
admission, transport, document requirements, fee structure, hostel
facilities, college rules, and syllabus using LangGraph + Groq LLM,
with a fallback general-knowledge path.
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


# Step 1 - Building the RAG retrievers

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_retriever(pdf_path: str, index_name: str):
    """Builds a FAISS retriever for a PDF, caching the index on disk so
    repeated runs load instantly instead of re-embedding every time."""

    index_path = f"faiss_index_{index_name}"

    if os.path.exists(index_path):
        # Index already built earlier -> just load it (fast)
        vectorstore = FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )
    else:
        # First run for this PDF -> build the index and save it
        loader = PyPDFLoader(pdf_path)
        document = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=800,
                                                   chunk_overlap=100)

        chunks = splitter.split_documents(document)

        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(index_path)

    return vectorstore.as_retriever(search_kwargs={"k": 4})


# 8 PDFs -> 8 retrievers (each cached under its own faiss_index_<name> folder)
# NOTE: file names below match exactly what is present in the project folder
scholarship_retriever = build_retriever("scholership_pdf.pdf", "scholarship")
admission_retriever = build_retriever("Admission_documents.pdf", "admission")
transport_retriever = build_retriever("transport.pdf", "transport")
fee_retriever = build_retriever("fee_structure.pdf", "fee")
hostel_retriever = build_retriever("hostel_booklet.pdf", "hostel")
rules_retriever = build_retriever("rule_book.pdf", "rules")
courses_retriever = build_retriever("academic_boucher.pdf", "courses")
syllabus_retriever = build_retriever("B.Tech_1styear.pdf", "syllabus")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.4)


# Step 2 - State

class State(TypedDict):
    programme: str
    messages: Annotated[list, add_messages]
    query_type: str
    retrieved_context: str


# Step 3 - Nodes generation

def classifier_node(state: State) -> dict:
    """Look at the latest user message and decide which path to take."""

    last_message = state['messages'][-1].content

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
        "Use 'general' for greetings, casual talk, or anything not related to the above topics.\n\n"
        f"Query: {last_message}\n\n"
        "Return only one word: scholarship, admission, transport, fee, hostel, rules, "
        "courses, syllabus, or general."
    )

    response = llm.invoke(prompt)
    category = response.content.strip().lower()

    if "scholarship" in category:
        category = "scholarship"
    elif "admission" in category:
        category = "admission"
    elif "transport" in category:
        category = "transport"
    elif "fee" in category:
        category = "fee"
    elif "hostel" in category:
        category = "hostel"
    elif "rules" in category:
        category = "rules"
    elif "syllabus" in category:
        category = "syllabus"
    elif "courses" in category:
        category = "courses"
    else:
        category = "general"

    return {"query_type": category}


def _retrieve_context(retriever, query: str) -> str:
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])


def scholarship_rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the scholarship PDF."""
    query = state["messages"][-1].content
    return {"retrieved_context": _retrieve_context(scholarship_retriever, query)}


def admission_rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the admission/document-requirement PDF."""
    query = state["messages"][-1].content
    return {"retrieved_context": _retrieve_context(admission_retriever, query)}


def transport_rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the transport PDF."""
    query = state["messages"][-1].content
    return {"retrieved_context": _retrieve_context(transport_retriever, query)}


def fee_rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the fee structure PDF (course + hostel)."""
    query = state["messages"][-1].content
    return {"retrieved_context": _retrieve_context(fee_retriever, query)}


def hostel_rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the hostel booklet PDF."""
    query = state["messages"][-1].content
    return {"retrieved_context": _retrieve_context(hostel_retriever, query)}


def rules_rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the rule book PDF."""
    query = state["messages"][-1].content
    return {"retrieved_context": _retrieve_context(rules_retriever, query)}


def courses_rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the academic brochure PDF."""
    query = state["messages"][-1].content
    return {"retrieved_context": _retrieve_context(courses_retriever, query)}


def syllabus_rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the B.Tech 1st year PDF."""
    query = state["messages"][-1].content
    return {"retrieved_context": _retrieve_context(syllabus_retriever, query)}


def general_node(state: State) -> dict:
    """Answers directly using the LLM's own knowledge, no retrieval needed."""
    return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}


def response_node(state: State) -> dict:
    """Generates the final answer, personalized using the student's programme."""
    query = state["messages"][-1].content
    programme = state.get("programme", "Unknown")
    context = state["retrieved_context"]

    if context == "NO_RETRIEVAL_NEEDED":
        prompt = (
            f"You are a friendly college assistant talking to a {programme} student. "
            f"Answer this question using your own general knowledge:\n\n{query}"
        )
    else:
        prompt = (
            f"You are a college assistant helping a {programme} student. "
            f"Use the following context from the official college documents to answer "
            f"the question accurately. If the context mentions specific figures for "
            f"different programmes, highlight the one relevant to {programme} if possible.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Give a clear, friendly, and precise answer."
        )

    response = llm.invoke(prompt)
    return {"messages": [("ai", response.content.strip())]}


# Step 4 - router function

def route_query(state: State):
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
    return mapping.get(state['query_type'], "general")


# Step 5 - Building the graph

graph = StateGraph(State)

graph.add_node("classifier", classifier_node)
graph.add_node("scholarship_rag", scholarship_rag_node)
graph.add_node("admission_rag", admission_rag_node)
graph.add_node("transport_rag", transport_rag_node)
graph.add_node("fee_rag", fee_rag_node)
graph.add_node("hostel_rag", hostel_rag_node)
graph.add_node("rules_rag", rules_rag_node)
graph.add_node("courses_rag", courses_rag_node)
graph.add_node("syllabus_rag", syllabus_rag_node)
graph.add_node("general", general_node)
graph.add_node("response", response_node)

# edges

graph.add_edge(START, "classifier")

graph.add_conditional_edges(
    "classifier", route_query
)

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

app = graph.compile()

# Step 6 - Run the code

print("=========================================")
print("   AI Admission Counselor for MIET")
print("=========================================\n")

print("Which programme are you in?\n")

print("--- B.Tech (Engineering Branches, 4 Years) ---")
print("1.  Biotechnology")
print("2.  Civil Engineering")
print("3.  Computer Science")
print("4.  Computer Science and Engineering (CSE)")
print("5.  CSE - Artificial Intelligence (AI)")
print("6.  CSE - Data Science")
print("7.  CSE - Artificial Intelligence & Machine Learning (AI & ML)")
print("8.  Electrical Engineering (EE)")
print("9.  Electronics & Communication Engineering (ECE)")
print("10. Mechanical Engineering (Minor in AIML / Hons. Robotics & AI)")
print("11. ECE (VLSI Design & Technology)")

print("\n--- Management & Computer Applications (2 Years) ---")
print("12. MBA")
print("13. MCA")

print("\n--- M.Tech (2 Years) ---")
print("14. M.Tech - Electronics & Communication Engineering")
print("15. M.Tech - Biotechnology")
print("16. M.Tech - Computer Science and Engineering")
print("17. M.Tech - CSE (Artificial Intelligence & Machine Learning)")

print("\n--- Pharmacy Courses ---")
print("18. B.Pharm (4 Years)")
print("19. M.Pharm - Pharmaceutics (2 Years)")
print("20. M.Pharm - Pharmacology (2 Years)")

print("\n--- New Courses for Working Professionals ---")
print("21. B.Tech - Mechanical Engineering (3 Years)")
print("22. B.Tech - Electrical Engineering (3 Years)")
print("23. M.Tech - Computer Science and Engineering (2 Years)")

print("\n--- Basic Sciences (Affiliated to CCS University, Meerut) ---")
print("24. B.Sc. Biotechnology (3 Years)")
print("25. B.Sc. Microbiology (3 Years)")
print("26. B.Sc. (Hons.) Biotechnology (3 Years)")
print("27. B.Sc. (Hons.) Microbiology (2 Years)")
print("28. M.Sc. Biotechnology (2 Years)")
print("29. M.Sc. Microbiology (2 Years)")

choice = input("\nEnter your choice number: ")

programme_map = {
    "1": "B.Tech Biotechnology",
    "2": "B.Tech Civil Engineering",
    "3": "B.Tech Computer Science",
    "4": "B.Tech CSE",
    "5": "B.Tech CSE - AI",
    "6": "B.Tech CSE - Data Science",
    "7": "B.Tech CSE - AI & ML",
    "8": "B.Tech Electrical Engineering",
    "9": "B.Tech ECE",
    "10": "B.Tech Mechanical Engineering",
    "11": "B.Tech ECE (VLSI Design & Technology)",
    "12": "MBA",
    "13": "MCA",
    "14": "M.Tech - ECE",
    "15": "M.Tech - Biotechnology",
    "16": "M.Tech - CSE",
    "17": "M.Tech - CSE (AI & ML)",
    "18": "B.Pharm",
    "19": "M.Pharm - Pharmaceutics",
    "20": "M.Pharm - Pharmacology",
    "21": "B.Tech Mechanical Engineering (Working Professionals)",
    "22": "B.Tech Electrical Engineering (Working Professionals)",
    "23": "M.Tech CSE (Working Professionals)",
    "24": "B.Sc. Biotechnology",
    "25": "B.Sc. Microbiology",
    "26": "B.Sc. (Hons.) Biotechnology",
    "27": "B.Sc. (Hons.) Microbiology",
    "28": "M.Sc. Biotechnology",
    "29": "M.Sc. Microbiology",
}
student_programme = programme_map.get(choice, "B.Tech CSE")

print(f"\nGreat! You're set as a {student_programme} student.")

while True:
    user_query = input("You:  ")

    if user_query.lower() in ["exit", "quit"]:
        break

    result = app.invoke({
        "programme": student_programme,
        "messages": [("human", user_query)]
    })

    print(f"Assistant : {result['messages'][-1].content}")