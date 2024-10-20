# embedding_and_indexing.py
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np

def get_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def create_index(texts):
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # Compute embeddings for each text
    embeddings = [get_embedding(text, model, tokenizer) for text in texts]
    
    # Convert the list of embeddings to a 2D numpy array
    embeddings_array = np.vstack(embeddings).astype(np.float32)

    # Get the dimensionality of the embeddings
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    
    # Add the embeddings to the index
    index.add(embeddings_array)
    return index, embeddings_array

if __name__ == "__main__":
    # Example texts (replace with texts extracted from PDFs)
    texts = ["This is the first document.", "This is the second document."]
    index, embeddings = create_index(texts)
    print("Index and embeddings created.")
