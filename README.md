# Fraud-detection-using-ML

Fraud is one of the major issues we come up majorly in banks, life insurance, health insurance, and many others. These major frauds are dependent on the person who is trying to sell you the fake product or service, if you are matured enough to decide what is wrong then you will never get into any fraud transactions. But one such fraud that has been increasing a lot these days is fraud in making payments.

The dataset that I used is the transaction data for online purchases collected from an e-commerce retailer on internet. The dataset contains more than 39000 transactions, each transaction contains 5 features that will describe the nature of the transactions.

These are the required libraries required for fraud detetction using ML 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

As this is a problem of binary classification, I used a Logistic Regression algorithm, as it is one of the most powerful algorithms for a binary classification model.

let’s simply train the fraud detection model using logistic regression algorithm and have a look at the accuracy score that we will get by using this algorithm:
clf = LogisticRegression().fit(X_train, y_train)

# Make predictions on test set
y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred, y_test))
Code language: Python (python)

Now, let’s evaluate the performance of our model. I used the confusion matrix algorithm to evaluate the performance of our model. We can use the confusion matrix algorithm with a one-line code only:

# Compare test set predictions with ground truth labels
print(confusion_matrix(y_test, y_pred))

This is the output we get: 
[[12753 0]
[ 0 190]]

So out of all the transaction in the dataset,190 transactions are correctly recognized as fraud, and 12753 transactions are recognized as not fraudulent transactions.







