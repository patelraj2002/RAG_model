import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

st.title("Gemma Model Document Q&A")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma-7b-it")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./data")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        
        # Ensure final_documents is not empty
        if not st.session_state.final_documents:
            st.error("No documents found. Please ensure there are PDF files in the './data' directory.")
            return
        
        # Generate embeddings for documents
        try:
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embedding)
            st.write("Vector Store DB is Ready")
        except Exception as e:
            st.error(f"Error generating vector store: {e}")

prompt1 = st.text_input("What do you want to ask from the documents?")
if st.button("Document Embedding"):
    vector_embedding()

import time

if prompt1:
    if "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write("Response Time:", time.process_time() - start)
        st.write(response['answer'])

        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
    else:
        st.error("Please generate the document embeddings first by clicking the 'Document Embedding' button.")

