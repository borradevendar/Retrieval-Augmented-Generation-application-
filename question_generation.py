from transformers import pipeline

def generate_questions(documents):
    # Use the text2text-generation pipeline for question generation
    question_generator = pipeline("text2text-generation", model="valhalla/t5-small-qg-hl")
    questions = set()  # Use a set to store unique questions

    for doc in documents:
        # The model expects input in the format: "generate question: <text>"
        input_text = f"generate question: {doc}"
        results = question_generator(input_text)
        
        for result in results:
            generated_question = result['generated_text'].strip()
            if generated_question:  # Check if the question is not empty
                questions.add(generated_question)  # Add to set for uniqueness

    return list(questions)  # Convert back to list
