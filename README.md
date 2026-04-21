# Sentiment-Analysis-Project

## 📂 Dataset

The dataset is too large to store on GitHub.

Download it here:
[IMDB Dataset](https://drive.google.com/file/d/1y-rCU8LabROle08NhxLq2FUZTihn_mO9/view?usp=sharing)
[Cleaned Data Files](https://drive.google.com/drive/folders/1mj5K6ESZ03ruf5nXdTET7NkEtjSNSsyY?usp=sharing)

## 🚀 Dash Web App (Step 6)

**Local Run**:
```
pip install -r requirements.txt
python app.py
```
Visit: http://127.0.0.1:8050

Enter review text, click Predict → Get sentiment (Positive/Negative) + probability bar.

**Deploy to Render** (Python 3.10+):
1. Push to GitHub repo.
2. New Web Service → Connect repo.
3. Build: `pip install -r requirements.txt`
4. Start: `gunicorn app: app`
5. Live URL provided.

Models auto-loaded from `models/`.

## 📋 TODO
See [TODO.md](TODO.md)

## Team Workflow
Follows info.text steps 1-6.
