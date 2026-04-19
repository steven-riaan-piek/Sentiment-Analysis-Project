# Model Development – Sentiment Analysis

## Objective

The objective of this phase was to develop machine learning classification models to predict the sentiment of movie reviews (positive or negative) and to determine the better-performing model.

## Approach

### Data Loading

The dataset was loaded from a CSV file located at:
`mlg381/data/IMDBDataset.csv`

The dataset contains:

* `review`: textual movie reviews
* `sentiment`: target variable (positive or negative)

Initial checks were performed to inspect the dataset shape, columns, and class distribution.

### Data Cleaning

Text preprocessing was applied to prepare the data for machine learning. The following steps were performed:

* Conversion of all text to lowercase
* Removal of non-alphabetic characters using regular expressions

This ensured that the text data was standardized and noise was reduced.

### Feature Extraction

Text data was converted into numerical features using TF-IDF vectorization.
A maximum of 5000 features was used to represent the most important words in the dataset.

### Train-Test Split

The dataset was split into:

* 80% training data
* 20% testing data

A fixed random state (`random_state=42`) was used to ensure reproducibility.

## Models Developed

### Logistic Regression

Logistic Regression was implemented as the primary classification model.

* The model was trained using the training dataset
* Predictions were generated on the test dataset

### Naive Bayes

Multinomial Naive Bayes was implemented as a secondary model for comparison.

* The model was trained on the same training data
* Predictions were generated on the test dataset

## Model Evaluation

Model performance was evaluated using accuracy.

* Logistic Regression Accuracy: approximately 0.88
* Naive Bayes Accuracy: slightly lower than Logistic Regression

A comparison was performed to determine the better model.

## Model Selection

Logistic Regression was selected as the better-performing model because it achieved higher accuracy compared to Naive Bayes.

## Challenges

* Ensuring correct file paths for loading the dataset
* Applying proper text preprocessing before vectorization
* Avoiding errors caused by incorrect execution order in the notebook
* Handling large text data efficiently using feature limits

## Conclusion

The model development phase successfully trained and compared two classification models. Logistic Regression outperformed Naive Bayes and was selected as the final model for sentiment prediction. The model is capable of accurately classifying unseen movie reviews into positive or negative sentiments.
