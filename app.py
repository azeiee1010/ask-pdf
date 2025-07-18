import streamlit as st
from dotenv import load_dotenv
import os

from utils.pdf_utils import extract_text_from_pdf
from utils.qa_chain import build_chain

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Ask PDF", layout="centered")
st.title("Ask Questions from PDF")

pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])

if pdf_file:
    pdf_text = extract_text_from_pdf(pdf_file)

    if pdf_text.strip():
        st.success("PDF loaded and text extracted.")

        user_question = st.text_input("Ask a question about the PDF:")

        if user_question.strip():
            chain, docs = build_chain(pdf_text, OPENAI_API_KEY)

            with st.spinner("Processing your question..."):
                answer = chain.run(input_documents=docs, question=user_question)

            st.subheader("Answer")
            st.write(answer)
    else:
        st.warning("No text could be extracted from the uploaded PDF.")
