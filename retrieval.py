# retrieval.py
from embed import get_embedding, create_index
import numpy as np

def retrieve_documents(query, index, embeddings, texts, model, tokenizer, top_k=3):
    query_embedding = get_embedding(query, model, tokenizer)
    _, indices = index.search(query_embedding, top_k)
    return [texts[i] for i in indices[0]]

if __name__ == "__main__":
    texts = ["This is the first document.", "This is the second document."]
    index, embeddings = create_index(texts)

    # Model and tokenizer initialization
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    query = "What is artificial intelligence?"
    relevant_docs = retrieve_documents(query, index, embeddings, texts, model, tokenizer)
    print("Relevant Documents:", relevant_docs)
