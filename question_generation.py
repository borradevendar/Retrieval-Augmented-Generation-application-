from transformers import pipeline

def generate_questions(documents):
    # Use the text2text-generation pipeline for question generation
    question_generator = pipeline("text2text-generation", model="valhalla/t5-small-qg-hl")
    questions = []
    for doc in documents:
        # The model expects input in the format: "generate question: <text>"
        input_text = f"generate question: {doc}"
        results = question_generator(input_text)
        for result in results:
            questions.append(result['generated_text'])  # Correctly access 'generated_text'
    print(questions)
    return questions

if __name__ == "__main__":
    documents = [
        "Deep learning is a subset of machine learning involving neural networks.", 
        "Artificial intelligence is the simulation of human intelligence in machines."
    ]
    questions = generate_questions(documents)
    for q in questions:
        print("Generated Question:", q)  # Directly print q since it is a string
