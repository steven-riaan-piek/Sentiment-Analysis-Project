# Sentiment Analysis Project

## Render Deployment

Build: `pip install -r requirements.txt`
Start: `gunicorn app:app`

**Issues fixed:**
- Python 3.14.3 Cython build fails for sklearn 1.3-1.5.

**Solution:**
1. Install cloudpickle: `pip install cloudpickle`
2. Re-save models:
```python
import cloudpickle
import joblib
model = joblib.load('models/sentiment_model.pkl') 
vectorizer = joblib.load('models/tfidf_vectorizer.pkl') 
cloudpickle.dump(model, open('models/sentiment_model.cpkl', 'wb'))
cloudpickle.dump(vectorizer, open('models/tfidf_vectorizer.cpkl', 'wb'))
```
3. Update app.py model loads to .cpkl files.
4. git add/commit/push.

No heavy deps - deploys instantly.

