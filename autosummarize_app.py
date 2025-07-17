# autosummarize_app.py

import streamlit as st
from transformers import pipeline
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import fitz
import docx
import nltk

nltk.download('punkt')

# ----------- File Reading -----------

def read_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "".join(page.get_text() for page in doc)

def read_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def read_txt(file):
    return file.read().decode("utf-8")

# ----------- Summarization -----------

def abstractive_summary(text, max_len=150, min_len=40):
    cleaned = " ".join(text.strip().split())
    if len(cleaned.split()) < 100:
        raise ValueError("Text too short for abstractive summarization.")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(cleaned, max_length=max_len, min_length=min_len, do_sample=False)
    return summary[0]['summary_text']

def extractive_summary(text, num_sentences=5):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)

# ----------- Streamlit UI -----------

st.set_page_config(page_title="AutoSummarize", layout="centered")
st.title("📄 AutoSummarize: Document Summarization Tool")

uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt", "docx"])
summary_type = st.selectbox("Choose summarization type", ["Abstractive", "Extractive"])
num_sentences = st.slider("Number of sentences (for extractive)", 3, 10, 5)

if uploaded_file:
    file_type = uploaded_file.type
    if file_type == "application/pdf":
        text = read_pdf(uploaded_file)
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = read_docx(uploaded_file)
    elif file_type == "text/plain":
        text = read_txt(uploaded_file)
    else:
        st.error("Unsupported file type.")
        text = ""

    if text:
        st.subheader("📜 Original Text Preview")
        st.text_area("Document Text", text[:3000], height=300)

        if st.button("Generate Summary"):
            with st.spinner("Summarizing..."):
                try:
                    if summary_type == "Abstractive":
                        summary = abstractive_summary(text)
                    else:
                        summary = extractive_summary(text, num_sentences)
                    st.subheader("📝 Summary")
                    st.write(summary)
                except Exception as e:
                    st.error(f"Error: {e}")