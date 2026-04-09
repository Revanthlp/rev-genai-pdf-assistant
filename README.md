📄 REV AI PDF Assistant (RAG Chatbot)
🚀 Project Overview

REV AI PDF Assistant is an AI-powered Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and interact with them using natural language questions.

The system extracts content from PDFs, splits it into chunks, converts it into embeddings, stores them in a vector database, and retrieves relevant context to generate accurate answers using a Large Language Model.

🧠 Key Features
📂 Upload multiple PDF files
🔍 Intelligent document chunking & embedding
🧾 Context-aware question answering (RAG pipeline)
💬 Conversational memory (chat history support)
⚡ Real-time streaming responses
📌 Source document highlighting
🎨 Clean Streamlit UI with custom styling
🏗️ System Architecture
PDF Files
   ↓
Document Loader (PyPDFLoader)
   ↓
Text Splitter (RecursiveCharacterTextSplitter)
   ↓
Embeddings (HuggingFace MiniLM)
   ↓
Vector Store (Chroma DB)
   ↓
Retriever
   ↓
LLM (Flan-T5 via HuggingFace)
   ↓
Conversational Retrieval Chain
   ↓
Streamlit Chat UI
🛠️ Tech Stack
Frontend: Streamlit
Backend: Python
LLM: google/flan-t5-base (HuggingFace Transformers)
Embeddings: sentence-transformers/all-MiniLM-L6-v2
Vector Database: ChromaDB
Framework: LangChain
PDF Processing: PyPDFLoader
📦 Installation & Setup
1️⃣ Clone Repository
git clone https://github.com/Revanthlp/rev-genai-pdf-assistant
cd rev-genai-pdf-assistant
2️⃣ Create Virtual Environment (Optional)
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run Application
streamlit run app.py
📁 Project Structure
rev-genai-pdf-assistant/
│
├── app.py                  # Main Streamlit application
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
├── logo.png               # UI logo
└── chroma_db/             # Vector database (auto-generated)
🧪 How It Works
User uploads PDF files
PDFs are parsed into text
Text is split into chunks
Each chunk is converted into embeddings
Embeddings stored in Chroma vector DB
User asks a question
Most relevant chunks are retrieved
LLM generates final answer using context
Response is displayed in chat UI
💡 Example Use Cases
Academic research assistant 📚
Resume / document Q&A system 📄
Legal document analysis ⚖️
Notes summarizer 📑
⚠️ Limitations
Performance depends on model size (Flan-T5 base)
Large PDFs may slow processing
Requires internet for HuggingFace models
Not optimized for production-scale traffic
🚀 Future Improvements
Upgrade to LLaMA / Mistral models
Add authentication system
Deploy API backend (FastAPI)
Add multi-user chat history database
Improve UI with React frontend
👨‍💻 Author

Gullapalli Revanth Lakshmi Prasad
AI/ML Developer | GenAI Enthusiast

⭐ If you like this project

Give a ⭐ on the repository and feel free to contribute!
