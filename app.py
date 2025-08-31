from flask import Flask, request, jsonify, render_template
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Initialize Flask app
app = Flask(__name__)

# --- LOAD MODELS AND PREPROCESSING TOOLS ---

# Load the trained model and vectorizer
try:
    model = joblib.load('model/model.joblib')
    vectorizer = joblib.load('model/vectorizer.joblib')
except FileNotFoundError:
    model = None
    vectorizer = None

# Initialize text preprocessing tools from NLTK [cite: 38]
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    """Re-uses the same preprocessing function from training."""
    if not isinstance(text, str):
        return ""
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# --- DEFINE ROUTES ---

@app.route('/')
def home():
    """Renders the main page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyzes the review text from the request."""
    if model is None or vectorizer is None:
        return jsonify({'error': 'Model not loaded. Please train the model first by running train_model.py'}), 500

    data = request.get_json()
    review_text = data.get('review_text', '')

    if not review_text.strip():
        return jsonify({'error': 'Review text cannot be empty.'}), 400

    # 1. Preprocess the input text
    processed_text = preprocess_text(review_text)
    
    # 2. Vectorize the text using the loaded TF-IDF vectorizer [cite: 41]
    vectorized_text = vectorizer.transform([processed_text])
    
    # 3. Make a prediction
    prediction = model.predict(vectorized_text)
    probability = model.predict_proba(vectorized_text)

    # Interpret the prediction (0 = Genuine, 1 = Fake)
    result_text = "Likely Genuine ✅" if prediction[0] == 0 else "Potentially Fake ⚠️"
    confidence = (1 - probability[0][1]) if prediction[0] == 0 else probability[0][1]

    # Return the result as JSON
    return jsonify({
        'prediction': result_text,
        'confidence': f"{confidence:.2%}" 
    })

# --- RUN THE APP ---
if __name__ == '__main__':
    app.run(debug=True)