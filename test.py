import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_and_tokenizer(model_path='./trained_model'):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return tokenizer, model

def generate_response(tokenizer, model, question, max_length=128, num_beams=5, temperature=0.7, top_k=50, top_p=0.95):
    inputs = tokenizer.encode(question, return_tensors='pt')
    
    # Generate response
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_beams=num_beams,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=1
    )
    
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def evaluate_model(tokenizer, model, questions_and_answers):
    correct_count = 0
    for item in questions_and_answers:
        question = item['question']
        correct_answer = item['answer']
        model_response = generate_response(tokenizer, model, question)
        print(f"Question: {question}")
        print(f"Model Response: {model_response}")
        print(f"Correct Answer: {correct_answer}")
        # Compare the model response with the correct answer
        if correct_answer.lower() in model_response.lower():
            correct_count += 1
        print()
    
    accuracy = (correct_count / len(questions_and_answers)) * 100
    print(f"Accuracy: {accuracy:.2f}%")

def main():
    tokenizer, model = load_model_and_tokenizer()
    
    # Define questions and answers for internal testing
    questions_and_answers = [
        {"question": "Apa ibu kota Prancis?", "answer": "Paris"},
        {"question": "Siapa presiden Amerika Serikat?", "answer": "Joe Biden"}
    ]
    
    while True:
        mode = input("Masukkan 'test' untuk pertanyaan atau 'evaluate' untuk evaluasi akurasi (atau ketik 'exit' untuk keluar): ")
        if mode.lower() == 'exit':
            break
        elif mode.lower() == 'test':
            question = input("Masukkan pertanyaan: ")
            response = generate_response(tokenizer, model, question)
            print(f"Answer: {response}\n")
        elif mode.lower() == 'evaluate':
            evaluate_model(tokenizer, model, questions_and_answers)
        else:
            print("Pilihan tidak valid. Ketik 'test' untuk pertanyaan, 'evaluate' untuk evaluasi, atau 'exit' untuk keluar.")

if __name__ == "__main__":
    main()
