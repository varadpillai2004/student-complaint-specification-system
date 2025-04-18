from flask import Flask, request, render_template
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the trained model and vectorizer
model = pickle.load(open('student_complaint_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract complaint text
        report = request.form['report']

        # Debugging: print form data
        print(f"Report: {report}")

        # Text vectorization
        report_vectorized = vectorizer.transform([report])

        # Make prediction
        prediction = model.predict(report_vectorized)

        return render_template('result.html', prediction=prediction[0])

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
