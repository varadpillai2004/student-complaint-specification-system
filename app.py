from flask import Flask, request, render_template
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model and vectorizer
model = pickle.load(open('student_complaint_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        report = request.form['report']  # Complaint text
        age = float(request.form['age'])  # Age
        gpa = float(request.form['gpa'])  # GPA
        gender = request.form['gender']  # Gender (M/F)
        nationality = request.form['nationality']  # Nationality (e.g., Egypt)
        
        # Debugging: print form data
        print(f"Report: {report}, Age: {age}, GPA: {gpa}, Gender: {gender}, Nationality: {nationality}")
        
        # Text vectorization
        report_vectorized = vectorizer.transform([report])
        
        # Prepare features (scaling numerical features)
        numerical_features = scaler.transform([[age, gpa]])
        
        # Combine text and numerical features
        features = pd.concat([pd.DataFrame(report_vectorized.toarray()), pd.DataFrame(numerical_features)], axis=1)
        
        # Make prediction
        prediction = model.predict(features)
        
        return render_template('result.html', prediction=prediction[0])
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
