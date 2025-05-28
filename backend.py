import json
import re
import PyPDF2
from docx import Document
from typing import Dict, List
import numpy as np
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

def parse_template(template_file) -> List[Dict]:
    """Parse JSON template file."""
    try:
        template_data = json.load(template_file)
        sections = template_data.get('sections', [])
        if not sections:
            raise ValueError("Template has no sections.")
        return sections
    except Exception as e:
        raise ValueError(f"Invalid template format: {str(e)}")

def parse_document(document_file, section_names: List[str]) -> Dict[str, str]:
    """Parse PDF or DOCX document and extract sections."""
    sections = {}
    try:
        if document_file.name.endswith('.pdf'):
            reader = PyPDF2.PdfReader(document_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            lines = text.split('\n')
            current_section = None
            for line in lines:
                if line.strip() in section_names:
                    current_section = line.strip()
                    sections[current_section] = ""
                elif current_section:
                    sections[current_section] += line + " "
        elif document_file.name.endswith('.docx'):
            doc = Document(document_file)
            current_section = None
            for para in doc.paragraphs:
                if para.text.strip() in section_names:
                    current_section = para.text.strip()
                    sections[current_section] = ""
                elif current_section:
                    sections[current_section] += para.text + " "
        else:
            raise ValueError("Unsupported document format. Use PDF or DOCX.")
        return sections
    except Exception as e:
        raise ValueError(f"Error parsing document: {str(e)}")

def store_template_in_vector_db(sections: List[Dict], model, index):
    """Store template instructions in FAISS."""
    global section_names
    section_names = sections
    embeddings = model.encode([s['instructions'] for s in sections])
    index.reset()  # Clear previous index
    index.add(np.array(embeddings, dtype=np.float32))

def detect_placeholders(text: str) -> bool:
    """Detect placeholders like [Insert text here]."""
    placeholder_pattern = r'\[.*?(insert|enter|add|describe).*?\]'
    return bool(re.search(placeholder_pattern, text, re.IGNORECASE))

def generate_feedback(section_name: str, content: str, instruction: str, similarity: float, llm) -> str:
    """Generate feedback using Groq via LangChain with RecursiveCharacterTextSplitter."""
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Adjust based on Groq's context window (e.g., ~4000 tokens for Mixtral)
        chunk_overlap=100,  # Small overlap to maintain context
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    # Split content if too long
    content_chunks = text_splitter.split_text(content)
    feedback_parts = []

    # Define prompt template
    prompt_template = PromptTemplate(
        input_variables=["section_name", "content", "instruction", "similarity"],
        template="""
        You are an AI reviewer evaluating a technical document section. Provide concise, actionable feedback based on the following:
        - Section Name: {section_name}
        - Template Instruction: {instruction}
        - Submitted Content: {content}
        - Semantic Similarity Score: {similarity:.2f} (0 to 1 scale)
        - Placeholder Detection: {placeholders}

        Feedback should include:
        - Whether the content aligns with the template instructions.
        - Issues like missing details, placeholders, or irrelevance.
        - Suggestions for improvement.

        Keep feedback clear, professional, and under 100 words per chunk.
        """
    )

    placeholders = "Detected" if detect_placeholders(content) else "Not detected"

    # Process each chunk
    for chunk in content_chunks:
        prompt = prompt_template.format(
            section_name=section_name,
            content=chunk[:500],  # Limit chunk size further if needed
            instruction=instruction,
            similarity=similarity,
            placeholders=placeholders
        )

        try:
            response = llm.invoke(prompt)
            feedback_parts.append(response.content.strip())
        except Exception as e:
            feedback_parts.append(f"Error generating feedback for chunk: {str(e)}")

    # Combine feedback from all chunks
    combined_feedback = " ".join(feedback_parts)
    if len(feedback_parts) > 1:
        combined_feedback = f"Section processed in {len(feedback_parts)} chunks. {combined_feedback}"

    # Truncate to 300 words to avoid overly long feedback
    words = combined_feedback.split()
    if len(words) > 300:
        combined_feedback = " ".join(words[:300]) + "..."

    return combined_feedback