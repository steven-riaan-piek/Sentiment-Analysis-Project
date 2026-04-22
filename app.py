import joblib
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
# import numpy as np  # removed to avoid build issues; use model methods directly

import plotly.graph_objects as go
# import plotly.express as px  # removed - px.bar requires pandas


# Load the trained model and vectorizer
print("Loading model and vectorizer...")
model = joblib.load('models/sentiment_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
print("Model and vectorizer loaded successfully!")



# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# App layout
app.layout = dbc.Container([
    html.H1("Sentiment Analysis Web App", 
            className="text-center mb-4 mt-4", 
            style={'color': '#2c3e50'}),
    
    dbc.Row([
        dbc.Col([
            dcc.Textarea(
                id='review-input',
                placeholder='Enter your review text here... (this is only an example of what will show on businesses side)',
                style={'width': '100%', 'height': 150},
                className="form-control"
            ),
            html.Br(),
            dbc.Button(
                'Predict Sentiment',
                id='predict-button',
                color='primary',
                className='btn-lg btn-block',
                n_clicks=0
            ),
            html.Br(),
            html.Div(id='prediction-output', className="mt-4")
        ], width=10)
    ], className="justify-content-center"),
    
    html.Div(id='probability-bar', className="mt-4")
], fluid=True)

@callback(
    [Output('prediction-output', 'children'),
     Output('probability-bar', 'children')],
    [Input('predict-button', 'n_clicks')],
    [dash.dependencies.State('review-input', 'value')]
)
def update_prediction(n_clicks, review_text):
    if n_clicks > 0 and review_text:
        # Preprocess the input text using the loaded vectorizer
        X_new = vectorizer.transform([review_text])
        
        # Predict sentiment and probability
        prediction = model.predict(X_new)[0]
        probabilities = model.predict_proba(X_new)[0]
        
        sentiment = 'Positive' if prediction == 1 else 'Negative'
        pos_prob = f"{probabilities[1]*100:.1f}%" if hasattr(probabilities, '__len__') else "N/A"
        neg_prob = f"{probabilities[0]*100:.1f}%" if hasattr(probabilities, '__len__') else "N/A"
        
        # Main output
        output = dbc.Alert([
            html.H4(f"Prediction: {sentiment}", className="alert-heading"),
            html.P(f"Positive Probability: {pos_prob} | Negative Probability: {neg_prob}")
        ], color='success' if sentiment == 'Positive' else 'danger', className="text-center")
        
        # Probability bar chart - use go.Bar to avoid pandas dep
        fig = go.Figure(data=[
            go.Bar(
                x=['Negative', 'Positive'],
                y=probabilities,
                marker_color=['#e74c3c', '#27ae60']
            )
        ])
        fig.update_layout(
            title="Sentiment Probabilities",
            xaxis_title="Sentiment",
            yaxis_title="Probability",
            showlegend=False,
            height=400
        )
        
        bar = dcc.Graph(figure=fig)

        
        return output, bar
    
    return "", ""

if __name__ == '__main__':
    print("Starting Dash app on http://127.0.0.1:8050")
    app.run_server(debug=True, host='127.0.0.1', port=8050)

