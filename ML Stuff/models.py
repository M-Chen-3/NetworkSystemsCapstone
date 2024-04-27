import os
import sys
import numpy as np
import csv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score

TEST_SIZE = 0.2
VAL_SIZE = 0.2

labels = ["Google", "Amazon"]
x = [(5*(i%2), i+i%2) for i in range(10)]
y = [[0, 1][i%2] for i in range(10)]
length = len(x)

x_scaler = StandardScaler()
x = x_scaler.fit_transform(x).tolist()

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=TEST_SIZE
)

x_valid = x_train[:int(length*VAL_SIZE)]
y_valid = y_train[:int(length*VAL_SIZE)]
x_train = x_train[int(length*VAL_SIZE):]
y_train = y_train[int(length*VAL_SIZE):]

support_vector_machine = svm.SVC(gamma="scale")
support_vector_machine.fit(x_train, y_train)

svm_valid = support_vector_machine.predict(x_valid)
svm_test = support_vector_machine.predict(x_test)

print(f"SVM Accuracy of Validation Set Predictions: {accuracy_score(y_valid, svm_valid)}")
print(f"SVM F1 Score of Validation Set Predictions: {f1_score(y_valid, svm_valid)}")
print(f"SVM Accuracy of Testing Set Predictions: {accuracy_score(y_test, svm_test)}")
print(f"SVM F1 Score of Validation Set Predictions: {f1_score(y_valid, svm_valid)}\n")

linear_svm = svm.LinearSVC(dual="auto")
linear_svm.fit(x_train, y_train)

linear_valid = linear_svm.predict(x_valid)
linear_test = linear_svm.predict(x_test)

print(f"Linear Accuracy of Validation Set Predictions: {accuracy_score(y_valid, linear_valid)}")
print(f"Linear F1 Score of Validation Set Predictions: {f1_score(y_valid, linear_valid)}")
print(f"Linear Accuracy of Testing Set Predictions: {accuracy_score(y_test, linear_test)}")
print(f"Linear F1 Score of Validation Set Predictions: {f1_score(y_valid, linear_valid)}")
