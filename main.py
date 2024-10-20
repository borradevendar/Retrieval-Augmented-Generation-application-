# main.py
from extract_text import extract_text_from_pdfs
from embed import create_index
from retrieval import retrieve_documents
from question_generation import generate_questions
from transformers import AutoTokenizer, AutoModel

def generate_questions_from_pdfs(pdf_files, query):
    # Extract text from PDFs
    texts = extract_text_from_pdfs(pdf_files)
    
    # Create embeddings and index
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    index, embeddings = create_index(texts)
    
    # Retrieve relevant documents
    relevant_docs = retrieve_documents(query, index, embeddings, texts, model, tokenizer)
    
    # Generate questions
    questions = generate_questions(relevant_docs)
    return questions

if __name__ == "__main__":
    pdf_files = ["./pdf_folder/girlmeetsboy1.pdf",
                 "./pdf_folder/girlmeetsboy.pdf",
                 "./pdf_folder/chess1.pdf",
                 ]  # Replace with your PDF file paths
    query = "Explain the concept of deep learning."
    questions = generate_questions_from_pdfs(pdf_files, query)

    # Generate questions from documents
    generated_questions = generate_questions(questions)

    for q in generated_questions:
        print("Generated Question:", q)  # Directly print q since it is a string
