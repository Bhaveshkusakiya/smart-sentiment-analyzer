import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# === 1. Load your labeled data ===
# CSV must have: 'Feedback', 'Sentiment' (1=positive, 0=negative)
df = pd.read_csv('feedback_data.csv')  # Replace with your real data path

# === 2. Preprocessing function ===
def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    return text.lower().strip()

df['Feedback'] = df['Feedback'].apply(preprocess)

# === 3. Train-test split ===
X = df['Feedback']
y = df['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 4. Build a simple pipeline ===
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression(max_iter=1000))
])

# === 5. Train the model ===
model.fit(X_train, y_train)

# === 6. Evaluate ===
y_pred = model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# === 7. Save model ===
joblib.dump(model, 'model/sentiment_model.pkl')
print("✅ Model saved to model/sentiment_model.pkl")
