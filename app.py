import streamlit as st
import tempfile
import time
import re
from PIL import Image

from transformers import pipeline

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="REV AI PDF Assistant", layout="wide")

# -------------------------
# CUSTOM UI (LIGHT THEME)
# -------------------------
st.markdown("""
<style>
.stApp {
    background-color: #f7f9fc;
    color: #000000;
}

h1 {
    color: #1F618D;
    text-align: center;
    font-weight: bold;
}

[data-testid="stChatMessage"] {
    border-radius: 15px;
    padding: 12px;
    margin-bottom: 10px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
}

[data-testid="stChatMessage"]:has(div[aria-label="user"]) {
    background-color: #D6EAF8;
}

[data-testid="stChatMessage"]:has(div[aria-label="assistant"]) {
    background-color: #EBF5FB;
}

section[data-testid="stSidebar"] {
    background-color: #F2F4F4;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# LOGO + TITLE
# -------------------------
col1, col2, col3 = st.columns([1,2,1])
with col2:
    logo = Image.open("logo.png")  # place your image in project folder
    st.image(logo, width=200)

st.markdown("<h1>REV's AI PDF Assistant</h1>", unsafe_allow_html=True)

# -------------------------
# SIDEBAR
# -------------------------
with st.sidebar:
    st.header("⚙️ Controls")

    if st.button("🗑 Reset Chat"):
        st.session_state.messages = []
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        st.success("Chat Reset!")

    show_memory = st.checkbox("🧠 Show Memory")

# -------------------------
# SESSION STATE
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

# -------------------------
# FILE UPLOAD
# -------------------------
uploaded_files = st.file_uploader(
    "📂 Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

# -------------------------
# STREAMING FUNCTION
# -------------------------
def stream_text(text):
    placeholder = st.empty()
    streamed = ""
    for word in text.split():
        streamed += word + " "
        placeholder.markdown(streamed)
        time.sleep(0.02)
    return streamed

# -------------------------
# HIGHLIGHT FUNCTION
# -------------------------
def highlight_text(context, answer):
    words = answer.split()[:10]
    pattern = "|".join(words)
    return re.sub(pattern, r"🔴 **\\g<0>**", context, flags=re.IGNORECASE)

# -------------------------
# PROCESS PDFs
# -------------------------
def process_pdfs(files):
    documents = []

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            loader = PyPDFLoader(tmp.name)
            documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)


    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=512
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=st.session_state.memory,
        return_source_documents=True
    )

    return qa_chain

# -------------------------
# PROCESS BUTTON
# -------------------------
if uploaded_files:
    if st.button("⚡ Process PDFs"):
        with st.spinner("Processing PDFs..."):
            st.session_state.qa_chain = process_pdfs(uploaded_files)
        st.success("✅ PDFs Ready!")

# -------------------------
# CHAT DISPLAY
# -------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------
# USER INPUT
# -------------------------
query = st.chat_input("Ask something about your PDFs...")

if query:
    if st.session_state.qa_chain is None:
        st.warning("⚠️ Upload and process PDFs first.")
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.qa_chain({"question": query})

                answer = result["answer"]
                sources = result["source_documents"]

                # Streaming effect
                final_answer = stream_text(answer)

                # Highlighted sources
                with st.expander("📄 Source Context (Highlighted)"):
                    for i, doc in enumerate(sources):
                        st.markdown(f"**Source {i+1}:** {doc.metadata.get('source')}")
                        highlighted = highlight_text(doc.page_content[:500], answer)
                        st.markdown(highlighted + "...")

        st.session_state.messages.append({"role": "assistant", "content": final_answer})

# -------------------------
# MEMORY VIEW
# -------------------------
if show_memory:
    st.sidebar.subheader("🧠 Memory")
    for msg in st.session_state.memory.chat_memory.messages:
        st.sidebar.write(msg.content)