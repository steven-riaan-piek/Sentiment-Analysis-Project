import joblib
import cloudpickle
 
# Load original models
model = joblib.load('models/sentiment_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
 
# Save as cloudpickle
cloudpickle.dump(model, open('models/sentiment_model.cpkl', 'wb'))
cloudpickle.dump(vectorizer, open('models/tfidf_vectorizer.cpkl', 'wb'))
 
print("Conversion complete!")