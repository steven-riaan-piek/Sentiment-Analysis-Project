# Model Development – Sentiment Analysis

## Objective

The objective of this phase was to build machine learning classification models to predict the sentiment of movie reviews (positive or negative) and to identify the best-performing model for this task.

## Approach

### Data Preparation

The dataset used was the IMDB movie reviews dataset.
The target variable was `sentiment`.
The feature variable was `review`.
Text data was cleaned by converting to lowercase, removing HTML tags, removing non-alphabetic characters, and eliminating unnecessary whitespace.

### Feature Extraction

Text data was transformed into numerical features using TF-IDF vectorization.
A maximum of 5000 features was used to represent the most important words in the dataset.

### Train-Test Split

The dataset was split into:

* 80% training data
* 20% testing data

A fixed random state was used to ensure reproducibility of results.

## Models Developed

### Logistic Regression

Logistic Regression was used as the primary classification model for sentiment prediction.

* Accuracy: ~0.88
* F1 Score: ~0.88

### Naive Bayes

The Multinomial Naive Bayes model was implemented as a secondary model for comparison.

* Accuracy: ~0.84–0.87
* F1 Score: ~0.84–0.87

## Model Selection

Logistic Regression was selected as the final model because it consistently achieved higher accuracy and F1 score compared to Naive Bayes, making it more suitable for sentiment classification.

## Challenges

* Text preprocessing required careful handling to remove HTML artifacts such as "br" tags from the dataset.
* File path issues initially prevented the dataset from loading and were resolved by correcting directory paths.
* Ensuring that TF-IDF vectorization was applied correctly was critical, as incorrect implementation led to errors during model training.
* Large dataset size increased computation time, requiring efficient feature selection using a limited number of features.

## Conclusion

The modeling phase successfully developed and evaluated two classification models for sentiment analysis. Logistic Regression was selected as the best-performing model, achieving strong and consistent results. The model is suitable for predicting sentiment in unseen text data and can be used in further stages such as evaluation, saving, and deployment.
