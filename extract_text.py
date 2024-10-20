# extract_text.py
import pdfplumber

def extract_text_from_pdfs(pdf_files):
    all_texts = []
    for file in pdf_files:
        with pdfplumber.open(file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
            all_texts.append(text)
    return all_texts

if __name__ == "__main__":
    pdf_files = ["C:/Users/borra/OneDrive/Desktop/pdf_folder/Tales of Alonica.pdf"]  # Replace with your PDF file paths
    texts = extract_text_from_pdfs(pdf_files)
    for i, text in enumerate(texts):
        print(f"PDF {i+1} Text:\n", text[:500])  # Print first 500 characters of each PDF
