import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv('student_complaints_large.csv')

# Check the column names to ensure they are correct
print(df.columns)

# Preprocess the data (using 'Complaint' as input and 'Category' as target)
X = df['Complaint']  # Only 'Complaint' column as feature
y = df['Category']   # 'Category' column as the target (output)

# Text vectorization for 'Complaint' (text data)
vectorizer = TfidfVectorizer(max_features=5000)
X_complaint_vectorized = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_complaint_vectorized, y, test_size=0.2, random_state=42)

# Train a model (RandomForestClassifier in this case)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model, vectorizer
pickle.dump(model, open('student_complaint_model.pkl', 'wb'))
pickle.dump(vectorizer, open('tfidf_vectorizer.pkl', 'wb'))

print("Model training and saving complete.")
