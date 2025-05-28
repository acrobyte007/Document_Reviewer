# Document Review Tool

A Streamlit-based AI tool for automated scoring and review of technical documents (up to 100 pages) using a Retrieval-Augmented Generation (RAG) approach. It compares document sections against a user-provided JSON template, evaluates content quality, detects placeholders, enforces configurable rules, and generates feedback using Groq's LLM via LangChain.

## Features
- Upload JSON template and PDF/DOCX document.
- Uses SBERT (`all-MiniLM-L6-v2`) for embeddings and FAISS for vector storage.
- Splits long sections with RecursiveCharacterTextSplitter for Groq LLM feedback.
- Configurable rules (e.g., mandatory sections, minimum word count).
- Provides section scores, feedback, and approval/rejection decision.

## Requirements
- Python 3.8+
- Dependencies: `streamlit`, `sentence-transformers`, `faiss-cpu`, `PyPDF2`, `python-docx`, `numpy`, `langchain`, `langchain-groq`, `langchain-text-splitters`, `python-dotenv`
- Groq API key (set in `.env` file as `GROQ_API_KEY`)

## Setup
1. Install dependencies:
   ```bash
   pip install streamlit sentence-transformers faiss-cpu PyPDF2 python-docx numpy langchain langchain-groq langchain-text-splitters python-dotenv
   ```
2. Create a `.env` file with:
   ```bash
   GROQ_API_KEY=your-groq-api-key
   ```
3. Save `app.py` and `backend.py` in the same directory.

## Usage
1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Open the provided URL (e.g., `http://localhost:8501`) in a browser.
3. Upload a JSON template (e.g.):
   ```json
   {
     "sections": [
       {"name": "System Architecture", "instructions": "Describe the system components, including hardware and software, with a diagram."},
       {"name": "Functional Requirements", "instructions": "List all functional requirements with priority levels and acceptance criteria."}
     ]
   }
   ```
4. Upload a PDF/DOCX document with matching section headers.
5. Configure rules (mandatory sections, minimum words).
6. Click "Review Document" to view scores, feedback, and decision.

## Example Output
For a document with a missing section and placeholders:
```
Overall Score: 30.3%
Decision: Rejected
---
Section: System Architecture (Score: 0%)
Feedback: Section 'System Architecture' is missing but is mandatory.
---
Section: Functional Requirements (Score: 60.5%)
Feedback: Section processed in 2 chunks. The content has a similarity score of 0.81 but contains placeholders like '[Insert requirements]'. Replace placeholders with specific functional requirements.
```