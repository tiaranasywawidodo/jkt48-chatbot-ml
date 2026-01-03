import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_response(intent, user_input):
    prompt = f"""
Kamu adalah chatbot informasi JKT48.
User bertanya: "{user_input}"
Intent terdeteksi: {intent}
Jawab secara informatif dan sopan.
"""
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
