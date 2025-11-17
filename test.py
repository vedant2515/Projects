import fitz  
import pandas as pd
from docx import Document
from openai import OpenAI
import chromadb
import os

os.environ["OPENAI_API_KEY"] = "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"  # Replace with your key
client = OpenAI()

chroma_client = chromadb.Client()
collection = chroma_client.create_collection("documents")

def extract_text(file_path):
    if file_path.endswith(".pdf"):
        text = ""
        with fitz.open(file_path) as pdf:
            for page in pdf:
                text += page.get_text()
        return text

    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])

    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        return df.to_string()

    else:
        raise ValueError("Unsupported file format! Upload PDF, DOCX, or CSV.")

def chunk_text(text, chunk_size=1000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def store_embeddings(chunks):
    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[str(i)],
            documents=[chunk]
        )

def retrieve_context(query, top_k=3):
    results = collection.query(query_texts=[query], n_results=top_k)
    contexts = [doc for doc in results['documents'][0]]
    return "\n".join(contexts)

import ollama

def ask_question(query):
    context = retrieve_context(query)
    prompt = f"Answer based on the following document:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    response = ollama.chat(model='mistral', messages=[{"role": "user", "content": prompt}])
    return response['message']['content']


if __name__ == "__main__":
    path = "Demo.pdf"
    text = extract_text(path)
    chunks = chunk_text(text)
    store_embeddings(chunks)

    print("\Document loaded successfully! You can now ask questions.\n")

    while True:
        q = input("Ask a question (or type 'exit'): ")
        if q.lower() == "exit":
            print("👋 Exiting Q&A Bot.")
            break
        ans = ask_question(q)
        print("\nAnswer:", ans, "\n")
