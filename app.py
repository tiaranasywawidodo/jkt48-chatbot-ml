import joblib
from utils.preprocessing import preprocess_text
from utils.groq_client import generate_response

vectorizer = joblib.load("model/vectorizer.pkl")
model = joblib.load("model/classifier.pkl")

print("Chatbot JKT48 siap! (ketik 'exit' untuk keluar)")

while True:
    user_input = input("Kamu: ")
    if user_input.lower() == "exit":
        break

    clean_text = preprocess_text(user_input)
    vector = vectorizer.transform([clean_text])
    intent = model.predict(vector)[0]

    response = generate_response(intent, user_input)
    print("Bot:", response)
