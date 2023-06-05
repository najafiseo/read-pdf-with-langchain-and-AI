import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from deep_translator import GoogleTranslator

load_dotenv()
st.header("LangChain GPT App 😍")
language = st.selectbox("Choose your language :" , ("English" , "Persian"))
pdf = st.file_uploader("Choose your PDF File: " , type="pdf")
if pdf :
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text+= page.extract_text()
       

    # creating Chunks

    text_splitter = CharacterTextSplitter(        
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 200,
        length_function = len,
    )
    chunks = text_splitter.split_text(text)
    st.write(chunks)
    if language == "Persian":
        chunks = GoogleTranslator(source="fa" , target = "en").translate_batch(chunks)
    else :
        pass
    

    #ebedding
    embeddings = OpenAIEmbeddings()
    knowladge_base = FAISS.from_texts(chunks ,embeddings )

    user_question = st.text_input("Ask your question:")
    if language =="Persian":
        user_question = GoogleTranslator(source="fa", target = "en").translate(user_question)
    else:
        pass

    if user_question :
        docs = knowladge_base.similarity_search(user_question)
        chain = load_qa_chain(OpenAI(temperature=0.9), chain_type="stuff")
        response = chain.run(input_documents=docs, question=user_question)
        if language == "Persian":
            response=GoogleTranslator(ource="en", target = "fa").translate(response)
        st.write(response)
