import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

# --- 1. PREPARE DATASET ---

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load your dataset
# This dataset should have two columns: 'text' (the review) and 'deceptive' (the label)
# The label can be 'truthful' or 'deceptive'. We will convert it to 0 and 1.
df = pd.read_csv('deceptive-opinion.csv')

# Map labels to numbers (0 for genuine, 1 for fake/deceptive)
# Make sure to check the exact labels in your CSV file.
df['label'] = df['deceptive'].apply(lambda x: 1 if x == 'deceptive' else 0)
X = df['text']
y = df['label']

# --- 2. TEXT PREPROCESSING ---

# As mentioned in your presentation, NLP preprocessing is key [cite: 37]
stemmer = PorterStemmer()

def preprocess_text(text):
    """Cleans and prepares the text for modeling."""
    # Remove non-alphabetic characters and convert to lower case
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    # Tokenize
    words = text.split()
    # Remove stopwords and apply stemming
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

X_processed = X.apply(preprocess_text)

# --- 3. FEATURE EXTRACTION (TF-IDF) ---

# Using TfidfVectorizer as recommended in your presentation [cite: 41, 43]
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X_processed)

# --- 4. MODEL TRAINING ---

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Using Scikit-learn to build the classifier [cite: 36]
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# --- 5. SAVE THE MODEL AND VECTORIZER ---

# Create the model directory if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')

# Save the trained model and the vectorizer for later use in the web app
joblib.dump(model, 'model/model.joblib')
joblib.dump(vectorizer, 'model/vectorizer.joblib')

print("\nModel and vectorizer saved successfully in the 'model/' directory!")