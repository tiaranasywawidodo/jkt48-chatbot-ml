import joblib
from utils.preprocessing import preprocess_text

# Load model & vectorizer
intent_model = joblib.load("model/intent_model.pkl")
tfidf_vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

print("🤖 Chatbot Informasi JKT48")
print("Ketik 'exit' untuk keluar\n")

while True:
    user_input = input("Kamu: ")

    if user_input.lower() == "exit":
        print("🤖 Terima kasih!")
        break

    # Preprocessing
    clean_text = preprocess_text(user_input)

    # Transform text
    X = tfidf_vectorizer.transform([clean_text])

    # Predict intent
    intent = intent_model.predict(X)[0]

    # Response (sementara statis, nanti pakai Groq API)
    response_map = {
        "informasi_umum": "JKT48 adalah idol group asal Indonesia yang berbasis di Jakarta.",
        "profil_jkt48": "JKT48 merupakan sister group resmi AKB48.",
        "jadwal_event": "Jadwal event JKT48 dapat dilihat di website resmi.",
        "member_jkt48": "JKT48 memiliki banyak member dari berbagai generasi.",
        "teater_jkt48": "Teater JKT48 berada di FX Sudirman Jakarta.",
        "jam_operasional": "Teater JKT48 biasanya buka sore hingga malam.",
        "harga_tiket": "Harga tiket teater JKT48 bervariasi tergantung event.",
        "cara_beli_tiket": "Tiket dapat dibeli melalui website resmi JKT48.",
        "promo_event": "Pantau media sosial resmi JKT48 untuk promo terbaru."
    }

    print(f"🤖 ({intent}) :", response_map.get(intent, "Maaf, saya belum paham pertanyaan itu."))
