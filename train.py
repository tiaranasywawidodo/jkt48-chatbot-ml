import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from utils.preprocessing import preprocess_text

# Load dataset
with open("data/intents.json", encoding="utf-8") as f:
    data = json.load(f)

texts = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        texts.append(preprocess_text(pattern))
        labels.append(intent["tag"])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_train_vec = tfidf_vectorizer.fit_transform(X_train)
X_test_vec = tfidf_vectorizer.transform(X_test)

# Model
intent_model = LogisticRegression(max_iter=1000)
intent_model.fit(X_train_vec, y_train)

# Evaluation
y_pred = intent_model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(intent_model, "model/intent_model.pkl")
joblib.dump(tfidf_vectorizer, "model/tfidf_vectorizer.pkl")
