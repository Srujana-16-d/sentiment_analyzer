import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
import string
import joblib
from flask import Flask, request, render_template_string
import argparse

# Download necessary NLTK data
nltk.download('stopwords')

# Text preprocessing function
def preprocess_text(text):
    if pd.isna(text):
        return ''
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = ''.join([i for i in text if not i.isdigit()])
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    text = ' '.join([word for word in words if word not in stop_words])
    
    return text

class SentimentAnalyzer:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.classes = ['Negative', 'Neutral', 'Positive']
        try:
            self.pipeline = joblib.load('sentiment_model.pkl')
            print("Model loaded successfully")
        except Exception as e:
            print(f"No existing model found: {e}")
            self.pipeline = None

    def train(self, data_path):
        # Load and preprocess data
        df = pd.read_csv(data_path)
        df['text'] = df['text'].apply(preprocess_text)

        # Split data
        X = df['text']
        y = df['sentiment']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create pipeline
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('classifier', LinearSVC())
        ])

        # Train model
        self.pipeline.fit(X_train, y_train)

        # Evaluate model
        predictions = self.pipeline.predict(X_test)
        print("\nModel Performance:")
        print("Accuracy:", accuracy_score(y_test, predictions))
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))

        # Save model
        joblib.dump(self.pipeline, 'sentiment_model.pkl')

    def predict(self, text):
        if not self.pipeline:
            try:
                self.pipeline = joblib.load('sentiment_model.pkl')
            except Exception as e:
                return f"Error: Model not found. Please train the model first. Error: {str(e)}"
        
        try:
            processed_text = preprocess_text(text)
            prediction = self.pipeline.predict([processed_text])[0]
            return self.classes[prediction]
        except Exception as e:
            return f"Error: Failed to predict sentiment. Error: {str(e)}"

# Flask app
app = Flask(__name__)
analyzer = SentimentAnalyzer()

@app.route('/')
def home():
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Sentiment Analysis</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 20px auto; padding: 20px; }
                .container { background-color: #f9f9f9; padding: 20px; border-radius: 5px; }
                textarea { width: 100%; height: 150px; margin: 10px 0; }
                button { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
                button:hover { background-color: #45a049; }
                .result { margin-top: 20px; padding: 15px; background-color: #e3f2fd; border-radius: 5px; }
                .error { margin-top: 20px; padding: 15px; background-color: #ffebee; border-radius: 5px; color: #c62828; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Sentiment Analysis</h1>
                <form method="POST" action="/analyze">
                    <textarea name="text" placeholder="Enter text to analyze"></textarea>
                    <button type="submit">Analyze Sentiment</button>
                </form>
                {% if result %}
                    {% if result.startswith('Error:') %}
                        <div class="error">
                            <h3>Error:</h3>
                            <p>{{ result }}</p>
                        </div>
                    {% else %}
                        <div class="result">
                            <h3>Result:</h3>
                            <p>Sentiment: {{ result }}</p>
                        </div>
                    {% endif %}
                {% endif %}
            </div>
        </body>
        </html>
    ''', result=request.args.get('result'))

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form.get('text')
    if text:
        sentiment = analyzer.predict(text)
        return redirect(f"/?result={sentiment}")
    return redirect("/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sentiment Analysis Application')
    parser.add_argument('--train', help='Path to training data CSV file')
    parser.add_argument('--port', type=int, default=8505, help='Port to run the application on')
    args = parser.parse_args()

    if args.train:
        analyzer = SentimentAnalyzer()
        analyzer.train(args.train)
    else:
        print(f"Starting Flask app on port {args.port}...")
        print("You can access the application at:")
        print(f"http://localhost:{args.port}")
        print(f"http://127.0.0.1:{args.port}")
        app.run(host='0.0.0.0', port=args.port)