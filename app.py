"""
AI Admission Counselor MIET — Streamlit UI
--------------------------------------------
Run with:  streamlit run app.py

Flow:
  1. Landing / selection screen -> student picks their programme
  2. Screen switches to a full chat interface, personalized to that programme
"""

import streamlit as st
import streamlit.components.v1 as components
import os
import re
import html as html_lib
from backend import build_app, PROGRAMME_GROUPS, COLLEGE_INFO

MIET_LOGO_URL = COLLEGE_INFO["logo_url"]
SCHOLARSHIP_TOOL_PATH = os.path.join(os.path.dirname(__file__), "scholarship_tool.html")

st.set_page_config(
    page_title=f"AI Admission Counselor | {COLLEGE_INFO['short_name']}",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Global styling — MIET brand colors (red #DD322B + white + charcoal)
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Playfair+Display:wght@700;800&display=swap');

        :root {
            --miet-red: #DD322B;
            --miet-red-dark: #A8241F;
            --miet-charcoal: #222222;
        }

        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
        }

        #MainMenu, footer, header {visibility: hidden;}

        .stApp {
            background: #FAFAFA;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1020px;
        }

        /* ---------- Hero (selection page) ---------- */
        .hero {
            position: relative;
            overflow: hidden;
            background: linear-gradient(120deg, var(--miet-red-dark) 0%, var(--miet-red) 55%, #EF4B44 100%);
            border-radius: 24px;
            padding: 2.8rem 2.2rem 2.4rem 2.2rem;
            text-align: center;
            color: white;
            margin-bottom: 2.2rem;
            box-shadow: 0 16px 36px rgba(221, 50, 43, 0.28);
        }

        .hero::before {
            content: "";
            position: absolute;
            top: -60px;
            right: -60px;
            width: 220px;
            height: 220px;
            background: rgba(255,255,255,0.08);
            border-radius: 50%;
        }

        .hero::after {
            content: "";
            position: absolute;
            bottom: -80px;
            left: -40px;
            width: 180px;
            height: 180px;
            background: rgba(255,255,255,0.06);
            border-radius: 50%;
        }

        .hero-logo {
            width: 92px;
            height: 92px;
            border-radius: 20px;
            background: white;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 16px auto;
            box-shadow: 0 8px 22px rgba(0,0,0,0.2);
            padding: 10px;
            position: relative;
            z-index: 1;
        }

        .hero-logo img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        .hero .college-name {
            font-family: 'Playfair Display', serif;
            font-size: 1.75rem;
            font-weight: 800;
            letter-spacing: 0.2px;
            margin: 0 0 2px 0;
            position: relative;
            z-index: 1;
        }

        .hero .address {
            font-size: 0.78rem;
            color: #ffe3e1;
            margin: 0 0 6px 0;
            position: relative;
            z-index: 1;
        }

        .hero .est {
            font-size: 0.72rem;
            letter-spacing: 2px;
            text-transform: uppercase;
            color: #ffd9d6;
            margin-bottom: 16px;
            position: relative;
            z-index: 1;
        }

        .hero h1 {
            font-size: 1.9rem;
            font-weight: 800;
            margin: 0.3rem 0 0.5rem 0;
            letter-spacing: -0.5px;
            position: relative;
            z-index: 1;
        }

        .hero .tagline {
            font-size: 0.98rem;
            color: #ffe3e1;
            font-style: italic;
            margin: 0 0 0.3rem 0;
            position: relative;
            z-index: 1;
        }

        .hero .desc {
            font-size: 0.9rem;
            color: #ffd9d6;
            margin: 0;
            position: relative;
            z-index: 1;
        }

        .group-title {
            font-size: 1.02rem;
            font-weight: 700;
            color: var(--miet-charcoal);
            margin: 1.4rem 0 0.6rem 0;
            padding-left: 10px;
            border-left: 4px solid var(--miet-red);
        }

        div[data-testid="stButton"] > button {
            width: 100%;
            border-radius: 12px;
            border: 1.5px solid #ececec;
            background: white;
            color: var(--miet-charcoal);
            font-weight: 600;
            padding: 0.6rem 0.75rem;
            transition: all 0.15s ease;
            font-size: 0.9rem;
        }

        div[data-testid="stButton"] > button:hover {
            border-color: var(--miet-red);
            background: #fff2f1;
            color: var(--miet-red-dark);
            transform: translateY(-1px);
            box-shadow: 0 4px 10px rgba(221,50,43,0.12);
        }

        /* Back button gets its own compact style so it doesn't look like a
           full-width programme button */
        div[data-testid="stButton"].back-btn > button {
            width: auto;
            padding: 0.4rem 1rem;
            font-size: 0.85rem;
        }

        /* ---------- Mobile: keep programme buttons side-by-side in equal
           boxes instead of Streamlit's default single-column stacking ---------- */
        @media (max-width: 640px) {
            div[data-testid="stHorizontalBlock"] {
                flex-wrap: wrap !important;
                gap: 8px !important;
            }
            div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
                flex: 1 1 47% !important;
                width: 47% !important;
                min-width: 47% !important;
            }
            div[data-testid="stButton"] > button {
                font-size: 0.76rem;
                padding: 0.55rem 0.5rem;
                white-space: normal;
                min-height: 3rem;
                line-height: 1.25;
            }
        }

        /* ---------- Chat page header ---------- */
        .chat-header {
            background: linear-gradient(120deg, var(--miet-red-dark) 0%, var(--miet-red) 100%);
            border-radius: 18px;
            padding: 1rem 1.5rem;
            color: white;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.4rem;
            box-shadow: 0 8px 20px rgba(221, 50, 43, 0.22);
        }

        .chat-header .left {
            display: flex;
            align-items: center;
            gap: 13px;
        }

        .chat-header .logo-mini {
            width: 46px;
            height: 46px;
            border-radius: 12px;
            background: white;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 6px;
            flex-shrink: 0;
        }

        .chat-header .logo-mini img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        .chat-header .title {
            font-weight: 700;
            font-size: 1.08rem;
            line-height: 1.25;
        }

        .chat-header .subtitle {
            font-size: 0.8rem;
            color: #ffe3e1;
        }

        .chat-header .address {
            font-size: 0.72rem;
            color: #ffd9d6;
            margin-top: 1px;
        }

        .pill {
            background: rgba(255, 255, 255, 0.18);
            border: 1px solid rgba(255,255,255,0.35);
            padding: 5px 14px;
            border-radius: 20px;
            font-size: 0.78rem;
            font-weight: 600;
            white-space: nowrap;
        }

        /* ---------- Chat bubbles ---------- */
        div[data-testid="stChatMessage"] {
            background: transparent;
            padding: 0;
        }

        .msg-row {
            display: flex;
            margin-bottom: 14px;
        }

        .msg-row.user { justify-content: flex-end; }
        .msg-row.bot { justify-content: flex-start; }

        .bubble {
            max-width: 78%;
            padding: 0.75rem 1rem;
            border-radius: 16px;
            font-size: 0.94rem;
            line-height: 1.55;
        }

        .bubble.user {
            background: var(--miet-red);
            color: white;
            border-bottom-right-radius: 4px;
        }

        .bubble.bot {
            background: white;
            color: var(--miet-charcoal);
            border: 1px solid #eee;
            border-bottom-left-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        }

        .empty-state {
            text-align: center;
            padding: 2.2rem 1rem 1rem 1rem;
            color: #999;
        }

        .empty-state .icon {
            font-size: 2.2rem;
            margin-bottom: 8px;
        }

        .footer-note {
            text-align: center;
            color: #b8b8b8;
            font-size: 0.78rem;
            margin-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "select"
if "programme" not in st.session_state:
    st.session_state.programme = None
if "show_scholarship_tool" not in st.session_state:
    st.session_state.show_scholarship_tool = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


@st.cache_resource(show_spinner=False)
def load_graph():
    return build_app()


def render_message_text(text: str) -> str:
    """Safely prepares chat text for the custom HTML bubble.

    The bubble is inserted as raw HTML, so plain markdown (**bold**) doesn't
    render on its own — this escapes any stray HTML first (so the layout
    never breaks), then converts basic markdown (**bold**, *italics*) into
    real HTML and turns newlines into line breaks.
    """
    escaped = html_lib.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", escaped)
    escaped = escaped.replace("\n", "<br>")
    return escaped


def render_scholarship_tool_embed():
    """Renders the embedded UP State Scholarship Eligibility Checking Tool."""
    if os.path.exists(SCHOLARSHIP_TOOL_PATH):
        with open(SCHOLARSHIP_TOOL_PATH, "r", encoding="utf-8") as f:
            scholarship_tool_html = f.read()
        components.html(scholarship_tool_html, height=850, scrolling=True)
    else:
        st.warning("Scholarship tool file not found. Make sure 'scholarship_tool.html' is in the same folder as app.py.")


def _go_back_to_selection():
    st.session_state.page = "select"
    st.session_state.programme = None
    st.session_state.chat_history = []


# ---------------------------------------------------------------------------
# PAGE 1 — Programme selection
# ---------------------------------------------------------------------------

def render_selection_page():
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-logo">
                <img src="{MIET_LOGO_URL}" alt="{COLLEGE_INFO['short_name']} Logo" />
            </div>
            <div class="college-name">{COLLEGE_INFO['name']}</div>
            <div class="address">{COLLEGE_INFO['address']}</div>
            <div class="est">Shaping Futures Since {COLLEGE_INFO['established']}</div>
            <h1>AI Admission Counselor</h1>
            <p class="tagline">{COLLEGE_INFO['tagline']}</p>
            <p class="desc">Your instant guide to admissions, fees, scholarships, hostel &amp; more.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Select your programme to get started")

    for group, options in PROGRAMME_GROUPS.items():
        st.markdown(f'<div class="group-title">{group}</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        for i, option in enumerate(options):
            with cols[i % 3]:
                if st.button(option, key=f"prog_{option}"):
                    st.session_state.programme = option
                    st.session_state.page = "chat"
                    st.session_state.chat_history = []
                    st.rerun()

    st.markdown(
        f'<div class="footer-note">© {COLLEGE_INFO["short_name"]} Group of Institutions — {COLLEGE_INFO["address"]}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# PAGE 2 — Chat screen
# ---------------------------------------------------------------------------

SUGGESTIONS = [
    "What documents do I need for admission?",
    "Tell me about hostel facilities",
    "What scholarships are available?",
    "How does the transport service work?",
]


def render_chat_page():
    programme = st.session_state.programme

    # Visible back button, right above the header, on the main page itself —
    # NOT tucked inside the sidebar, which starts collapsed and was easy to miss.
    back_col, _spacer = st.columns([1, 5])
    with back_col:
        st.markdown('<div class="back-btn">', unsafe_allow_html=True)
        if st.button("⬅ Back to programmes", key="back_top"):
            _go_back_to_selection()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="chat-header">
            <div class="left">
                <div class="logo-mini">
                    <img src="{MIET_LOGO_URL}" alt="{COLLEGE_INFO['short_name']} Logo" />
                </div>
                <div>
                    <div class="title">AI Admission Counselor — {COLLEGE_INFO['name']}</div>
                    <div class="subtitle">{programme}</div>
                    <div class="address">{COLLEGE_INFO['address']}</div>
                </div>
            </div>
            <div class="pill">● Online</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.image(MIET_LOGO_URL, width=140)
        st.markdown("### Your Session")
        st.write(f"**Programme:** {programme}")
        if st.button("🔄 Change Programme"):
            _go_back_to_selection()
            st.rerun()
        st.markdown("---")
        st.caption("Ask about scholarships, admission, transport, required documents, fees, or hostel life.")

        st.markdown("---")
        with st.expander("🎓 Check Scholarship Eligibility"):
            st.caption("UP State Scholarship Eligibility Checking Tool (2026-27)")
            render_scholarship_tool_embed()

    with st.spinner("Setting things up..."):
        app = load_graph()

    # Empty state / suggestion chips
    clicked_suggestion = None
    if not st.session_state.chat_history:
        st.markdown(
            """
            <div class="empty-state">
                <div class="icon">💬</div>
                <div>Ask me anything about MIET admissions, fees, or hostel life.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        chip_cols = st.columns(len(SUGGESTIONS))
        for i, s in enumerate(SUGGESTIONS):
            with chip_cols[i]:
                if st.button(s, key=f"chip_{i}"):
                    clicked_suggestion = s

    # Render chat history as custom bubbles
    for i, (role, content) in enumerate(st.session_state.chat_history):
        css_class = "user" if role == "human" else "bot"
        safe_content = render_message_text(content)
        st.markdown(
            f"""
            <div class="msg-row {css_class}">
                <div class="bubble {css_class}">{safe_content}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # If this bot message points the student to the scholarship tool,
        # show a real, clickable button right under it that jumps straight
        # to the tool's UI instead of leaving it as plain, unclickable text.
        if role == "ai" and "scholarship eligibility" in content.lower():
            if st.button("🎓 Open Scholarship Eligibility Checker", key=f"open_schol_tool_{i}"):
                st.session_state.show_scholarship_tool = True
                st.rerun()

    if st.session_state.show_scholarship_tool:
        st.markdown("---")
        tool_header_col, tool_close_col = st.columns([5, 1])
        with tool_header_col:
            st.markdown("#### 🎓 Scholarship Eligibility Checker")
            st.caption("UP State Scholarship Eligibility Checking Tool (2026-27)")
        with tool_close_col:
            if st.button("✖ Close", key="close_schol_tool"):
                st.session_state.show_scholarship_tool = False
                st.rerun()
        render_scholarship_tool_embed()
        st.markdown("---")

    # Chat input
    user_query = st.chat_input("Type your question here...")
    final_query = user_query or clicked_suggestion

    if final_query:
        st.session_state.chat_history.append(("human", final_query))

        try:
            with st.spinner("Thinking..."):
                result = app.invoke({
                    "programme": programme,
                    "messages": [("human", final_query)],
                })
                answer = result["messages"][-1].content
        except RuntimeError as e:
            answer = f"⚠️ {e}"

        st.session_state.chat_history.append(("ai", answer))
        st.rerun()


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if st.session_state.page == "select":
    render_selection_page()
else:
    render_chat_page()