import joblib
import cloudpickle

# Load with joblib (this part is fine locally)
model = joblib.load('models/sentiment_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Re-save using ONLY cloudpickle
with open('models/sentiment_model.cpkl', 'wb') as f:
    cloudpickle.dump(model, f)

with open('models/tfidf_vectorizer.cpkl', 'wb') as f:
    cloudpickle.dump(vectorizer, f)

print("Converted without joblib dependency")