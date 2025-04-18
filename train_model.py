import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv('student_complaints.csv')

# Preprocess the data
X = df[['Reports', 'Age', 'Gpa', 'Gender', 'Nationality']]
y = df['Genre']

# Text vectorization for 'Reports' (complaint text)
vectorizer = TfidfVectorizer(max_features=5000)
X_report_vectorized = vectorizer.fit_transform(X['Reports'])

# Scaling numerical features: Age and GPA
scaler = StandardScaler()
X_numerical = scaler.fit_transform(X[['Age', 'Gpa']])

# Combine text features and numerical features
import scipy.sparse as sp
X_combined = sp.hstack([X_report_vectorized, X_numerical])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Train a model (RandomForest as an example)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model, vectorizer, and scaler
pickle.dump(model, open('student_complaint_model.pkl', 'wb'))
pickle.dump(vectorizer, open('tfidf_vectorizer.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

print("Model training and saving complete.")
