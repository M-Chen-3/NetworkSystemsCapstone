import os
from os import listdir
from os.path import isfile, join
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score

TEST_SIZE = 0.2
VAL_SIZE = 0.2

def load_data():
    # Get all the file names
    direct = "data" + os.sep
    data_files = [f for f in listdir(direct) if isfile(join(direct, f))]
    data_list = [pd.DataFrame(), pd.DataFrame()]

    for file in data_files:
        df = pd.read_csv(direct + file)

        # Get the labels
        data_list[1] = pd.concat([data_list[1], df["website"]], axis=0)
        df = df.drop(["website", "time", "description"], axis=1)

        ### Get the features ###
        # Standardize size column
        size_df = df["size"]
        df = df.drop("size", axis=1)
        size_array = size_df.values.reshape(-1, 1)
        scaler = MinMaxScaler()
        size_df = pd.DataFrame(scaler.fit_transform(size_array), columns=["size"])
        df = pd.concat([size_df, df], axis=1)

        # Select the columns we want to one-hot encode (as they aren't numeric)
        cat_columns = ["source", "dest", "protocol"]

        # One-hot encode those columns and remove old ones
        encoder = OneHotEncoder(sparse_output=False)
        encoded_parts = encoder.fit_transform(df[cat_columns])
        df = df.drop(cat_columns, axis=1)

        # Merge one-hot encoded columns with old dataframe
        encoded_df = pd.DataFrame(encoded_parts, columns=encoder.get_feature_names_out(cat_columns))
        new_df = pd.concat([df, encoded_df], axis=1)
        data_list[0] = pd.concat([data_list[0], new_df], axis=0, ignore_index=True)
        # print(data_list[0])

    return data_list

# Sample data for model testing
# labels = ["Google", "Amazon"]
# x = [(5*(i%2), i+i%2) for i in range(10)]
# y = [[0, 1][i%2] for i in range(10)]
# length = len(x)
# x_scaler = StandardScaler()
# x = x_scaler.fit_transform(x).tolist()

# Load in the data and replace NaNs with 0
# NaNs appear when a value in one file doesn't exist in the other file
x, y = load_data()
length = y.shape[0]
x = x.fillna(0)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=TEST_SIZE
)

# Split again into training set and validation set
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train, y_train, test_size=(1 - TEST_SIZE)*VAL_SIZE
)

support_vector_machine = svm.SVC(kernel="poly", gamma="scale")
support_vector_machine.fit(x_train, y_train)

svm_valid = support_vector_machine.predict(x_valid)
svm_test = support_vector_machine.predict(x_test)

print(f"SVM Accuracy of Validation Set Predictions: {accuracy_score(y_valid, svm_valid)}")
print(f"SVM F1 Score of Validation Set Predictions: {f1_score(y_valid, svm_valid, average='weighted')}")
print(f"SVM Accuracy of Testing Set Predictions: {accuracy_score(y_test, svm_test)}")
print(f"SVM F1 Score of Validation Set Predictions: {f1_score(y_valid, svm_valid, average='weighted')}\n")

linear_svm = svm.LinearSVC(dual="auto")
linear_svm.fit(x_train, y_train)

linear_valid = linear_svm.predict(x_valid)
linear_test = linear_svm.predict(x_test)

print(f"Linear Accuracy of Validation Set Predictions: {accuracy_score(y_valid, linear_valid)}")
print(f"Linear F1 Score of Validation Set Predictions: {f1_score(y_valid, linear_valid, average='weighted')}")
print(f"Linear Accuracy of Testing Set Predictions: {accuracy_score(y_test, linear_test)}")
print(f"Linear F1 Score of Validation Set Predictions: {f1_score(y_valid, linear_valid, average='weighted')}")
