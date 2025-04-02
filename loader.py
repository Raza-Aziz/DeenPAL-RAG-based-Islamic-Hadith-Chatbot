import re
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Load PDFs and Process Data (Only Runs on First App Launch)
@st.cache_resource
def load_and_prepare_data():
    print("1- Loading Hadith PDFs")
    
    folder_path = "data/"
    loader = PyPDFDirectoryLoader(folder_path)
    documents = loader.load()

    # Metadata processing
    for doc in documents:
        split_source = (doc.metadata['source'].split("/")[-1])
        exact_source_with_ext = split_source.split('_', maxsplit=1)[1]
        exact_source = exact_source_with_ext.split('.')[0]
        doc.metadata = {'source': exact_source}

    print("2- Documents loaded successfully.")

    # Splitting into chunks
    pattern = r"(?:Chapter\s\d+:)|(?:Book\s\d+,\sNumber\s\d+:)"
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators=[pattern], is_separator_regex=True)
    chunks = text_splitter.split_documents(documents)

    # Adding metadata
    for chunk in chunks:
        matches = re.search(pattern, chunk.page_content)
        if matches:
            hadith_number = "".join([word for word in matches.group(0) if word.isdigit()])
            chunk.metadata.update({'hadith_number': hadith_number})

    print("3- Documents split and metadata added.")

    # Generate embeddings
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    # Store embeddings in Chroma
    persist_directory = 'database/chroma_db'
    db = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_directory)

    print("4- Chroma vector store initialized.")
    return db, embeddings