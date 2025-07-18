from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

def build_chain(pdf_text, openai_api_key, model="gpt-3.5-turbo", temperature=0):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents([pdf_text])

    llm = ChatOpenAI(
        model_name=model,
        temperature=temperature,
        openai_api_key=openai_api_key
    )

    chain = load_qa_chain(llm, chain_type="stuff")
    return chain, docs
