import os
import tempfile

import chromadb
import ollama
import streamlit as st
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile

system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""


def process_document(uploaded_file: UploadedFile) -> list[Document]:
    with tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    loader = PyMuPDFLoader(temp_file_path)
    docs = loader.load()
    os.unlink(temp_file_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    return text_splitter.split_documents(docs)


def get_vector_collection() -> chromadb.Collection:
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )

    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )


def add_to_vector_collection(all_splits: list[Document], file_name: str):
    collection = get_vector_collection()
    
    # Query the collection to get the IDs of documents with the same file_name
    existing_docs = collection.get(where={"file_name": file_name})
    if existing_docs["ids"]:
        collection.delete(ids=existing_docs["ids"])
    
    documents, metadatas, ids = [], [], []
    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append({"file_name": file_name})
        ids.append(f"{file_name}_{idx}")
    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    st.success("Data added to the vector store! 🎉")


def query_collection(prompt: str, n_results: int = 10):
    collection = get_vector_collection()
    results = collection.query(query_texts=[prompt], n_results=n_results)
    return results


def call_llm(context: str, prompt: str):
    response = ollama.chat(
        model="llama3.2:3b",
        stream=True,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"Context: {context}, Question: {prompt}",
            },
        ],
    )
    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break


def re_rank_cross_encoders(documents: list[str], prompt:str) -> tuple[str, list[int]]:
    relevant_text = ""
    relevant_text_ids = []

    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ranks = encoder_model.rank(prompt, documents, top_k=3)
    for rank in ranks:
        relevant_text += documents[rank["corpus_id"]]
        relevant_text_ids.append(rank["corpus_id"])

    return relevant_text, relevant_text_ids


if __name__ == "__main__":
    st.set_page_config(page_title="BioGraphAI", page_icon="🧬")

    # Colorful sidebar with document upload
    with st.sidebar:
        st.title("🧬 BioGraphAI")
        st.markdown("---")
        st.markdown("### Upload Your Document")
        uploaded_file = st.file_uploader(
            "**Select a PDF file:**", type=["pdf"], accept_multiple_files=False,
            help="Upload a PDF document to begin."
        )

        if uploaded_file:
            if st.button("🚀 Process Document"):
                with st.spinner("Processing document..."):
                    normalize_uploaded_file_name = uploaded_file.name.translate(
                        str.maketrans({"-": "_", ".": "_", " ": "_"})
                    )
                    all_splits = process_document(uploaded_file)
                    add_to_vector_collection(all_splits, normalize_uploaded_file_name)
        st.markdown("---")
        st.markdown("### About")
        st.markdown("BioGraphAI uses advanced AI to answer questions from your uploaded documents.")

    # Main content area
    st.markdown("<h1 style='text-align: center; color: #4A90E2;'>Ask BioGraphAI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Enter your question below:</p>", unsafe_allow_html=True)

    prompt = st.text_area(
        "**Your Question:**",
        placeholder="Ask me anything about the document...",
        height=150,
    )

    if st.button("🔥 Get Answer"):
        if prompt:
            with st.spinner("Generating answer..."):
                prompt_modified = "imagine yourself as the main character of the uploaded document and answer likewise, make sure the answer is precise and make sure it is less than 100 words and also do not mention that you are a trained model. Behave as if and talk as if you are that person from the document(main character).You are the main character. Answer only the question not anything more." + prompt
                results = query_collection(prompt_modified)
                context = results.get("documents")[0]
                relevant_text, relevant_text_ids = re_rank_cross_encoders(context, prompt_modified)
                response = call_llm(context=relevant_text, prompt=prompt_modified)
                st.markdown("---")
                st.markdown("<h3 style='color: #2E8B57;'>Answer:</h3>", unsafe_allow_html=True)
                st.write_stream(response)

        else:
            st.warning("Please enter a question.")
