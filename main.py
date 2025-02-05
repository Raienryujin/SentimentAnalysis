import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
import string

# Configuration
DATA_PATH = 'sentiment_data.csv'
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_ITER = 200

# Download NLTK data if not already available
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# Load dataset
data = pd.read_csv(DATA_PATH)

# Handle missing data
data.dropna(subset=['text', 'sentiment'], inplace=True)

# Preprocess text data
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

data['text'] = data['text'].apply(preprocess_text)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['sentiment'], test_size=TEST_SIZE, random_state=RANDOM_STATE)

# Create a pipeline for vectorization and classification
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', LogisticRegression(max_iter=MAX_ITER))
])

# Train the model
pipeline.fit(X_train, y_train)

# Predict sentiments
y_pred = pipeline.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Function to predict sentiment of new text
def predict_sentiment(text):
    text = preprocess_text(text)
    prediction = pipeline.predict([text])
    return prediction[0]

# Example usage
new_text = "I hate this movie, its horrible"
print(predict_sentiment(new_text))