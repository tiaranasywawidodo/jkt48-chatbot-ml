from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import urllib.request
from utils.preprocessing import preprocess_text
from groq import Groq
from dotenv import load_dotenv

# Load API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# Download model jika belum ada
if not os.path.exists("model/intent_model.pkl"):
    os.makedirs("model", exist_ok=True)
    # Ganti URL berikut dengan link file model yang kamu simpan di GitHub/Drive
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/tiaranasywawidodo/repo/main/model/intent_model.pkl",
        "model/intent_model.pkl"
    )
if not os.path.exists("model/tfidf_vectorizer.pkl"):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/tiaranasywawidodo/repo/main/model/tfidf_vectorizer.pkl",
        "model/tfidf_vectorizer.pkl"
    )

# Load model & vectorizer
intent_model = joblib.load("model/intent_model.pkl")
tfidf_vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

app = FastAPI()

class UserMessage(BaseModel):
    message: str

@app.post("/chat")
def chat(user_message: UserMessage):
    clean_text = preprocess_text(user_message.message)
    X = tfidf_vectorizer.transform([clean_text])
    intent = intent_model.predict(X)[0]

    # Groq API response
    prompt = f"Jawab pertanyaan ini sesuai intent '{intent}' tentang JKT48: {user_message.message}"
    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            max_output_tokens=150
        )
        answer = response.output_text
    except Exception as e:
        answer = f"Maaf, terjadi error: {e}"

    return {"intent": intent, "answer": answer}
