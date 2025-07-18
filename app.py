import streamlit as st
import PyPDF2
from dotenv import load_dotenv
import os

from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.chat_models import ChatOpenAI

# Load environment variables (for OpenAI API key)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Ask PDF", layout="centered")
st.title("Ask Questions from PDF")

# Step 1: Upload PDF
pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])

# Step 2: Extract text from PDF
def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text
    except Exception as e:
        st.error(f"Failed to read PDF: {e}")
        return ""

if pdf_file:
    pdf_text = extract_text_from_pdf(pdf_file)

    if pdf_text.strip():
        st.success("PDF loaded and text extracted.")

        # Step 3: Ask a question
        user_question = st.text_input("Ask a question about the PDF:")

        if user_question.strip():
            # Step 4: Split text for better performance
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = text_splitter.create_documents([pdf_text])

            # Step 5: Initialize OpenAI model with LangChain
            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0,
                openai_api_key=OPENAI_API_KEY
            )

            # Step 6: Load QA chain
            chain = load_qa_chain(llm, chain_type="stuff")

            # Step 7: Run the QA chain
            with st.spinner("Processing your question..."):
                answer = chain.run(input_documents=docs, question=user_question)

            st.subheader("Answer")
            st.write(answer)
    else:
        st.warning("No text could be extracted from the uploaded PDF.")
