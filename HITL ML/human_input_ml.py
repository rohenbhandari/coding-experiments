# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 15:09:59 2023

@author: rb
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics

# Assume df is your data frame and it has a column 'email' with text and 'label' with 1 for spam and 0 for not spam
df = pd.read_csv('emails.csv')
df = df.dropna(subset=['email'])
df = df.dropna(subset=['label'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['email'], df['label'], test_size=0.2, random_state=42)

# Feature extraction
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Predict
predictions = model.predict(X_test_vectorized)

# Create a dataframe to hold the test data and predictions
result_df = pd.DataFrame({'email': X_test, 'true_label': y_test, 'predicted_label': predictions})

# Initial model evaluation
initial_predictions = model.predict(X_test_vectorized)
print("Initial Model Performance:")
print(f"Accuracy: {metrics.accuracy_score(y_test, initial_predictions)}")
print(f"Precision: {metrics.precision_score(y_test, initial_predictions, average='weighted')}")
print(f"Recall: {metrics.recall_score(y_test, initial_predictions, average='weighted')}")
print(f"F1 Score: {metrics.f1_score(y_test, initial_predictions, average='weighted')}")


def collect_human_feedback(row):
    print(f'Email: {row["email"]}\nPredicted Label: {row["predicted_label"]}')
    user_input = input("Is the prediction correct? (yes/no): ")
    if user_input.lower() == 'no':
        corrected_label = (input("Enter the correct label (1 for spam, 0 for not spam): "))
        return corrected_label
    return row["predicted_label"]
# Collect feedback on 5 random predictions
feedback_df = result_df.sample(5).apply(collect_human_feedback, axis=1)
result_df.loc[feedback_df.index, 'human_reviewed_label'] = feedback_df

# Filter out rows with NaN values in 'human_reviewed_label'
updated_data = result_df.dropna(subset=['human_reviewed_label'])

# Update the model with the new labels
model.fit(vectorizer.transform(updated_data['email']), updated_data['human_reviewed_label'])

X_test_vectorized_updated = vectorizer.transform(updated_data['email'])
y_test_updated = updated_data['human_reviewed_label']
updated_predictions = model.predict(X_test_vectorized_updated)
print("\nUpdated Model Performance:")
print(f"Accuracy: {metrics.accuracy_score(y_test_updated, updated_predictions)}")
print(f"Precision: {metrics.precision_score(y_test_updated, updated_predictions, average='weighted')}")
print(f"Recall: {metrics.recall_score(y_test_updated, updated_predictions, average='weighted')}")
print(f"F1 Score: {metrics.f1_score(y_test_updated, updated_predictions, average='weighted')}")

# Now the model has been updated with human feedback and can be used for further predictions