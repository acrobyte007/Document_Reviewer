import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from backend import parse_template, parse_document, store_template_in_vector_db, detect_placeholders, generate_feedback
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="Document Review Tool", layout="wide")
model = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384
index = faiss.IndexFlatL2(dimension)
section_names = []
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY not found. Please set it in .env file.")
    st.stop()
llm = ChatGroq(api_key=groq_api_key, model_name="mixtral-8x7b-32768")

def main():
    st.title("AI-Powered Document Review Tool")
    st.markdown("Upload a JSON template and a PDF/DOCX document to review. Configure rules and get automated feedback.")
    st.header("Upload Files")
    col1, col2 = st.columns(2)
    with col1:
        template_file = st.file_uploader("Template (JSON)", type=["json"])
    with col2:
        document_file = st.file_uploader("Document (PDF/DOCX)", type=["pdf", "docx"])
    st.header("Configure Rules")
    mandatory_sections = st.text_input("Mandatory Sections (comma-separated)", placeholder="e.g., System Architecture, Functional Requirements")
    min_words = st.number_input("Minimum Words per Section", min_value=0, value=100, step=10)
    if st.button("Review Document", disabled=not (template_file and document_file)):
        if not template_file or not document_file:
            st.error("Please upload both a template and a document.")
            return
        with st.spinner("Processing..."):
            try:
                rules = {
                    "mandatorySections": [s.strip() for s in mandatory_sections.split(",")] if mandatory_sections else [],
                    "minWords": int(min_words)
                }
                template_sections = parse_template(template_file)
                store_template_in_vector_db(template_sections, model, index)
                document_sections = parse_document(document_file, [s['name'] for s in template_sections])
                results = {'sections': [], 'overall_score': 0, 'decision': 'Needs Revision'}
                total_score = 0
                section_count = 0
                for section in template_sections:
                    section_name = section['name']
                    instruction = section['instructions']
                    content = document_sections.get(section_name, "")
                    if not content and section_name in rules['mandatorySections']:
                        results['sections'].append({
                            'name': section_name,
                            'score': 0,
                            'feedback': f"Section '{section_name}' is missing but is mandatory."
                        })
                        continue
                    word_count = len(content.split())
                    if word_count < rules['minWords']:
                        results['sections'].append({
                            'name': section_name,
                            'score': 0,
                            'feedback': f"Section '{section_name}' has {word_count} words, below the minimum of {rules['minWords']}."
                        })
                        continue
                    content_embedding = model.encode([content])[0]
                    instruction_embedding = model.encode([instruction])[0]
                    similarity = np.dot(content_embedding, instruction_embedding) / (np.linalg.norm(content_embedding) * np.linalg.norm(instruction_embedding))
                    score = similarity * 100
                    if detect_placeholders(content):
                        score -= 20
                    score = max(0, min(score, 100))
                    feedback = generate_feedback(section_name, content, instruction, similarity, llm)
                    results['sections'].append({
                        'name': section_name,
                        'score': round(score, 1),
                        'feedback': feedback
                    })
                    total_score += score
                    section_count += 1
                if section_count > 0:
                    results['overall_score'] = round(total_score / section_count, 1)
                if results['overall_score'] >= 80 and all(s['score'] > 0 for s in results['sections']):
                    results['decision'] = 'Approved'
                elif any(s['score'] == 0 for s in results['sections']):
                    results['decision'] = 'Rejected'
                st.header("Review Results")
                st.subheader(f"Overall Score: {results['overall_score']}%")
                st.subheader(f"Decision: {results['decision']}")
                st.write("---")
                for section in results['sections']:
                    with st.expander(f"Section: {section['name']} (Score: {section['score']}%)"):
                        st.markdown(f"**Feedback:** {section['feedback']}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()