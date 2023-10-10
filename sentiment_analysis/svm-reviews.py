# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 12:30:39 2023

@author: rb
"""

import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

from nltk.corpus import stopwords

# Load the data
data = pd.read_csv('reviews.csv')

# Prepare the data
data['full_text'] = data['title'] + ' ' + data['body']

# Prepare the data
X = data['full_text']  # Features (title + review text)
y = data['rating']  # Target variable (rating)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train the model
model = SVC(kernel='linear')  # You can experiment with other kernels like 'rbf'
model.fit(X_train_vectorized, y_train)

# Make predictions
y_pred = model.predict(X_test_vectorized)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output the evaluation metrics
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')


# Assume model is your trained SVM model and X_test_vectorized, y_test are your test data and labels
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.show()

y_probs = model.decision_function(X_test_vectorized)

# Plot histogram
plt.hist(y_probs, bins=10)
plt.xlim(0,1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability')
plt.ylabel('Frequency')
plt.show()

incorrect_predictions = y_test != y_pred
X_test_incorrect = X_test[incorrect_predictions]
y_test_incorrect = y_test[incorrect_predictions]
y_pred_incorrect = y_pred[incorrect_predictions]

# Display some examples of incorrect predictions
for i in range(10):  # Display 10 examples
    print(f'Text: {X_test_incorrect.iloc[i]}')
    print(f'Actual Rating: {y_test_incorrect.iloc[i]}')
    print(f'Predicted Rating: {y_pred_incorrect[i]}')
    print('---')
