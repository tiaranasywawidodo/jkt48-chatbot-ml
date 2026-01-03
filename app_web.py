from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from utils.preprocessing import preprocess_text
import os
from groq import Groq
from dotenv import load_dotenv

# Load Groq API Key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# Load model & vectorizer
intent_model = joblib.load("model/intent_model.pkl")
tfidf_vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

app = FastAPI()

# Model request
class UserMessage(BaseModel):
    message: str

@app.post("/chat")
def chat(user_message: UserMessage):
    clean_text = preprocess_text(user_message.message)
    X = tfidf_vectorizer.transform([clean_text])
    intent = intent_model.predict(X)[0]

    prompt = f"Jawab pertanyaan ini sesuai intent '{intent}' tentang JKT48: {user_message.message}"

    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            max_output_tokens=150
        )
        answer = response.output_text
    except Exception as e:
        answer = f"Maaf, terjadi error saat memanggil API: {e}"

    return {"intent": intent, "answer": answer}
