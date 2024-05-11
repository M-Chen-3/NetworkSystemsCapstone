import os
from os import listdir
from os.path import isfile, join
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

TEST_SIZE = 0.2
VAL_SIZE = 0.2

def load_data():
    # Get all the file names
    direct = "data" + os.sep
    data_files = [f for f in listdir(direct) if isfile(join(direct, f))]
    data_list = [pd.DataFrame(), pd.DataFrame()]
    overall = pd.DataFrame()

    for file in data_files:
        print(f"Loading {file}...")
        df = pd.read_csv(direct + file)
        overall = pd.concat([overall, df], axis=0, ignore_index=True)

        # Get the labels and remove unwanted ones
        data_list[1] = pd.concat([data_list[1], df["website"]], axis=0)
        df = df.drop(["source", "dest", "website", "time", "description"], axis=1)

        ### Get the features ###
        # Standardize size column
        size_df = df["size"]
        df = df.drop("size", axis=1)
        size_array = size_df.values.reshape(-1, 1)
        scaler = MinMaxScaler()
        size_df = pd.DataFrame(scaler.fit_transform(size_array), columns=["size"])
        df = pd.concat([size_df, df], axis=1)

        # Select the columns we want to one-hot encode (as they aren't numeric)
        cat_columns = ["protocol"]

        # One-hot encode those columns and remove old ones
        encoder = OneHotEncoder(sparse_output=False)
        encoded_parts = encoder.fit_transform(df[cat_columns])
        df = df.drop(cat_columns, axis=1)

        # Merge one-hot encoded columns with old dataframe
        encoded_df = pd.DataFrame(encoded_parts, columns=encoder.get_feature_names_out(cat_columns))
        new_df = pd.concat([df, encoded_df], axis=1)
        data_list[0] = pd.concat([data_list[0], new_df], axis=0, ignore_index=True)
        # print(data_list[0])

    # print(overall.groupby(["website", "protocol"]).count().to_string())
    # print(overall.groupby(["protocol"]).count().to_string())
    data_list[0].drop(["protocol_HTTP", "protocol_ICMP", "protocol_MDNS", "protocol_SSDP", "protocol_SSL", "protocol_SSLv2"], axis=1)
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
print()
length = y.shape[0]
x = x.fillna(0)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=TEST_SIZE
)

# Split again into training set and validation set
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train, y_train, test_size=(1 - TEST_SIZE)*VAL_SIZE
)

def test_model(model, name):
    print(f"Running {name}...")
    model.fit(x_train, y_train.values.ravel())

    model_valid = model.predict(x_valid)
    model_test = model.predict(x_test)

    print(f"{name} Accuracy of Validation Set Predictions: {accuracy_score(y_valid, model_valid)}")
    print(f"{name} F1 Score of Validation Set Predictions: {f1_score(y_valid, model_valid, average='macro')}")
    print(f"{name} Accuracy of Testing Set Predictions: {accuracy_score(y_test, model_test)}")
    print(f"{name} F1 Score of Testing Set Predictions: {f1_score(y_test, model_test, average='macro')}\n")

    # cm = confusion_matrix(y_test, model_test, labels=model.classes_)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm,
    #                             display_labels=model.classes_)
    # disp.plot()
    # plt.title(name + " Confusion Matrix Results")
    # plt.xticks(rotation=45)
    # plt.savefig(name + "_results.png", bbox_inches='tight')

# linear_svm = svm.LinearSVC(dual="auto")
# test_model(linear_svm, "Linear SVM")

support_vector_machine = svm.SVC(kernel="rbf", degree=2, gamma="scale", max_iter=10)
test_model(support_vector_machine, "SVM")

# multi_nb = MultinomialNB()
# test_model(multi_nb, "Naive Bayes")

# decision_tree = DecisionTreeClassifier(random_state=0)
# test_model(decision_tree, "Decision Tree")

# random_forest = RandomForestClassifier(n_estimators=100)
# test_model(random_forest, "Random Forest")